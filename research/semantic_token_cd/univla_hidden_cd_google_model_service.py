"""UniVLA Hidden-CD service using the Fractal Google/SIMPLER action adapter."""
from __future__ import annotations

import json
import os
import socket
import traceback
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.univla_l14_google_model_service import (
    Policy as GooglePolicy,
    recv_message,
    send_message,
)
LAMBDAS = {
    "hidden_cd_010": 0.10,
    "hidden_cd_015": 0.15,
    "hidden_cd_025": 0.25,
    "hidden_cd_035": 0.35,
    "hidden_cd_05": 0.5,
}
HIDDEN_ALIGNMENT = os.environ.get("UNIVLA_HIDDEN_ALIGNMENT", "act1234").strip().lower()
if HIDDEN_ALIGNMENT not in {"act1234", "prompt_act123"}:
    raise ValueError(
        "UNIVLA_HIDDEN_ALIGNMENT must be 'act1234' or 'prompt_act123', "
        f"got {HIDDEN_ALIGNMENT!r}"
    )


class HiddenGooglePolicy(GooglePolicy):
    """Apply hidden guidance with Google setup, Fractal stats and grip adapter."""

    @torch.inference_mode()
    def _decode_hidden_trace(self, hidden_states):
        latent_tokens = torch.cat(hidden_states, dim=1)
        visual_embed = latent_tokens[:, :256]
        latent_action = latent_tokens[:, -4:]
        stats = self.vla.get_action_stats(self.model.unnorm_key)
        mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        decoder = self.model.action_decoder

        def unnormalize(x):
            return np.where(mask, 0.5 * (x + 1.0) * (high - low) + low, x)

        before_buffer = decoder.action_buffer.copy()
        before_mask = decoder.action_buffer_mask.copy()
        post = decoder(latent_action, visual_embed, mask, low, high)
        pre = unnormalize(decoder.action_buffer[0, 0].copy())
        return {
            "preagg": np.asarray(pre, dtype=np.float64).reshape(7),
            "postagg": np.asarray(post, dtype=np.float64).reshape(7),
            "buffer_before": before_buffer,
            "buffer_mask_before": before_mask,
        }

    def _action(self, raw):
        from transforms3d.euler import euler2axangle
        axis, angle = euler2axangle(*raw[3:6])
        action = {
            "world_vector": raw[:3].tolist(),
            "rot_axangle": (axis * angle).tolist(),
            "terminate_episode": [0.0],
        }
        current = np.asarray(raw[6:7], dtype=np.float32)
        previous = self.model.previous_gripper_action
        adapter = os.environ.get("UNIVLA_GRIPPER_ADAPTER", "legacy")
        if adapter == "absolute_binary_delta_v1":
            target_open = np.array([float(np.clip(current[0], 0.0, 1.0) > 0.5)], dtype=np.float32)
            if previous is None:
                raise RuntimeError("initial gripper state missing; reset must pass gripper_open")
            relative = np.asarray(previous, dtype=np.float32) - target_open
            self.model.previous_gripper_action = target_open
        else:
            relative = np.array([0.0], dtype=np.float32) if previous is None else np.asarray(previous) - current
            self.model.previous_gripper_action = current
        if np.abs(relative).max() > 0.5 and not self.model.sticky_action_is_on:
            self.model.sticky_action_is_on = True
            self.model.sticky_gripper_action = relative
        if self.model.sticky_action_is_on:
            self.model.gripper_action_repeat += 1
            relative = self.model.sticky_gripper_action
        if self.model.gripper_action_repeat == self.model.sticky_gripper_num_repeat:
            self.model.sticky_action_is_on = False
            self.model.gripper_action_repeat = 0
            self.model.sticky_gripper_action = 0.0
        action["gripper"] = np.asarray(relative).reshape(-1).tolist()
        return action

    @torch.inference_mode()
    def step(self, image: np.ndarray, arm: str, instruction: str):
        # Reuse the canonical UniVLA clean decoding path while the rollout
        # restores the exact same simulator snapshot for all paired arms.
        if arm == "vanilla":
            return super().step(image, "vanilla", instruction)
        if arm not in LAMBDAS:
            raise ValueError(f"expected vanilla or Hidden-CD arm in {tuple(LAMBDAS)}, got {arm!r}")
        lam = LAMBDAS[arm]
        inputs, queries = self._inputs(image, instruction)
        positive = self._forward_prefill(inputs, attentions=True)
        if positive.attentions is None or len(positive.attentions) != 32:
            raise RuntimeError("L14 attention missing; expected all 32 eager-attention layers")
        attn = positive.attentions[14][0, :, queries, 1:257].detach().float().mean(dim=(0, 1)).cpu().numpy()
        feature = positive.projector_features[0].detach().float().cpu().numpy()
        selected, matched = self._matched_region(feature)
        order = np.lexsort((np.arange(256), -attn.astype(np.float64)))
        selected = sorted(int(x) for x in order[:len(selected)])
        negative_features = feature.copy()
        negative_features[selected] = self._harmonic_reconstruct(feature, selected)
        negative = self._forward_prefill(inputs, replacement=(selected, negative_features[selected]))
        actual = negative.projector_features[0].detach().float().cpu().numpy()
        expected = torch.from_numpy(negative_features).to(torch.bfloat16).float().numpy()
        if not np.array_equal(expected, actual):
            raise RuntimeError("negative projector features differ from requested harmonic reconstruction")

        pos_past, neg_past = positive.past_key_values, negative.past_key_values
        pos_hidden = [positive.hidden_states[-1]]
        neg_hidden = [negative.hidden_states[-1]]
        pos_logits, neg_logits = positive.logits[:, -1, :], negative.logits[:, -1, :]
        action_tokens = []
        for index in range(4):
            token = self._sample(pos_logits)
            action_tokens.append(token)
            if index < 3:
                po = self._advance(token, pos_past)
                no = self._advance(token, neg_past)
                pos_past, neg_past = po.past_key_values, no.past_key_values
                pos_logits, neg_logits = po.logits[:, -1, :], no.logits[:, -1, :]
                pos_hidden.append(po.hidden_states[-1])
                neg_hidden.append(no.hidden_states[-1])

        # ``act1234`` uses the ActionDecoder's four ACT-token positions.
        # ``prompt_act123`` intentionally reproduces the earlier WidowX
        # Hidden-CD alignment for a controlled comparison on Google tasks.
        if HIDDEN_ALIGNMENT == "act1234":
            po = self._advance(action_tokens[-1], pos_past)
            no = self._advance(action_tokens[-1], neg_past)
            pos_hidden.append(po.hidden_states[-1])
            neg_hidden.append(no.hidden_states[-1])

        hidden_plus = torch.cat(pos_hidden, dim=1)[:, -4:]
        hidden_minus = torch.cat(neg_hidden, dim=1)[:, -4:]
        hidden_star = hidden_plus + lam * (hidden_plus - hidden_minus)
        guided_hidden = torch.cat(pos_hidden, dim=1)
        guided_hidden[:, -4:] = hidden_star

        decoder = self.model.action_decoder
        original_buffer, original_mask = decoder.action_buffer.copy(), decoder.action_buffer_mask.copy()
        clean = self._decode_hidden_trace(pos_hidden)
        decoder.action_buffer, decoder.action_buffer_mask = original_buffer.copy(), original_mask.copy()
        guided = self._decode_hidden_trace([guided_hidden])
        pre_delta, post_delta = guided["preagg"] - clean["preagg"], guided["postagg"] - clean["postagg"]
        raw = guided["postagg"]
        meta = {
            "arm": arm, "guidance_space": "last_hidden_action_states", "lambda_h": lam,
            "hidden_alignment": HIDDEN_ALIGNMENT,
            "action_token_ids": action_tokens,
            "action_tokens_in_learned_range": sum(int(x > 32000) for x in action_tokens),
            "selected_token_ids": selected, "m_t": len(selected),
            "attention_mass": float(attn[selected].sum()),
            "feature_change_norm": float(np.linalg.norm(negative_features - feature)),
            "clean_preagg_raw": clean["preagg"].tolist(), "guided_preagg_raw": guided["preagg"].tolist(),
            "clean_postagg_raw": clean["postagg"].tolist(), "guided_postagg_raw": raw.tolist(),
            "preagg_delta_xyz": pre_delta[:3].tolist(), "preagg_delta_rot": pre_delta[3:6].tolist(),
            "preagg_delta_grip": float(pre_delta[6]),
            "postagg_delta_xyz": post_delta[:3].tolist(), "postagg_delta_rot": post_delta[3:6].tolist(),
            "postagg_delta_grip": float(post_delta[6]),
            "grip_threshold_flip": bool((clean["postagg"][6] > 0.5) != (raw[6] > 0.5)),
            "preagg_delta_l2": float(np.linalg.norm(pre_delta)),
            "postagg_delta_l2": float(np.linalg.norm(post_delta)),
            "action_hidden_rel_l2": float((torch.linalg.vector_norm((hidden_star-hidden_plus).float()) /
                torch.linalg.vector_norm(hidden_plus.float()).clamp_min(1e-12)).item()),
            "action_hidden_norm_ratio": float((torch.linalg.vector_norm(hidden_star.float()) /
                torch.linalg.vector_norm(hidden_plus.float()).clamp_min(1e-12)).item()),
            **matched,
        }
        return {"action": self._action(raw),
                "raw": {"world_vector": raw[:3].tolist(), "rotation_delta": raw[3:6].tolist(),
                        "open_gripper": raw[6:7].tolist()}, "meta": meta}

    def _harmonic_reconstruct(self, feature, selected):
        from research.semantic_token_cd.univla_l14_google_model_service import harmonic_reconstruct
        return harmonic_reconstruct(feature, selected)


def main():
    sock = os.environ["UNIVLA_HIDDEN_SOCK"]
    policy = HiddenGooglePolicy()
    Path(sock).parent.mkdir(parents=True, exist_ok=True)
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock)
    server.listen(1)
    print(json.dumps({"model_ready": True, "service": "univla_hidden_cd_google",
                      "selected_layer": 14, "arms": LAMBDAS,
                      "hidden_alignment": HIDDEN_ALIGNMENT,
                      "checkpoint": str(policy.model.saved_model_path) if hasattr(policy.model, "saved_model_path") else os.environ.get("UNIVLA_CHECKPOINT"),
                      "action_stats_key": policy.model.unnorm_key,
                      "gripper_adapter": os.environ.get("UNIVLA_GRIPPER_ADAPTER")}), flush=True)
    conn, _ = server.accept()
    with conn:
        while True:
            try:
                head, payload = recv_message(conn)
            except EOFError:
                break
            try:
                cmd = head.get("cmd")
                if cmd == "quit":
                    send_message(conn, {"ok": True})
                    break
                if cmd == "reset":
                    result = policy.reset(head["instruction"], int(head["seed"]),
                                          None if "gripper_open" not in head else float(head["gripper_open"]))
                elif cmd == "step":
                    image = np.frombuffer(payload, dtype=np.uint8).reshape(head["shape"])
                    result = policy.step(image, head["arm"], head["instruction"])
                else:
                    raise ValueError(f"unknown command {cmd}")
                send_message(conn, {"ok": True, "result": result})
            except Exception as exc:
                traceback.print_exc()
                send_message(conn, {"ok": False, "error": repr(exc)})
    server.close()
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()

"""UniVLA L14-Matched hidden-state guidance service for the SIMPLER pilot.

This service is deliberately separate from the established L14/logit service.
For each step it samples only the clean positive latent-token prefix, advances
both cached branches with that same prefix, then guides the four action hidden
states before the official ActionDecoder.  The decoder's temporal buffer is
cloned so clean and guided raw actions are compared from identical history;
the guided decode is the one whose updated buffer is retained for execution.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import traceback
from pathlib import Path

import numpy as np
import torch

from research.semantic_token_cd.univla_l14_model_service import (
    Policy as L14Policy,
    harmonic_reconstruct,
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


class HiddenPolicy(L14Policy):
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
        pre_normalized = decoder.action_buffer[0, 0].copy()
        pre = unnormalize(pre_normalized)
        return {
            "preagg": np.asarray(pre, dtype=np.float64).reshape(7),
            "postagg": np.asarray(post, dtype=np.float64).reshape(7),
            "buffer_before": before_buffer,
            "buffer_mask_before": before_mask,
        }

    @staticmethod
    def _action(raw):
        from transforms3d.euler import euler2axangle
        roll, pitch, yaw = raw[3:6]
        axis, angle = euler2axangle(roll, pitch, yaw)
        return {
            "world_vector": raw[:3].tolist(),
            "rot_axangle": (axis * angle).tolist(),
            "gripper": [float(2.0 * (raw[6] > 0.5) - 1.0)],
            "terminate_episode": [0.0],
        }

    @torch.inference_mode()
    def step(self, image: np.ndarray, arm: str, instruction: str):
        if arm not in LAMBDAS:
            raise ValueError(f"HiddenPolicy only accepts {tuple(LAMBDAS)}, got {arm!r}")
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
        negative_features[selected] = harmonic_reconstruct(feature, selected)
        neg = self._forward_prefill(inputs, replacement=(selected, negative_features[selected]))
        nfeatures = neg.projector_features[0].detach().float().cpu().numpy()
        expected = torch.from_numpy(negative_features).to(torch.bfloat16).float().numpy()
        if not np.array_equal(expected, nfeatures):
            raise RuntimeError("negative projector features do not match harmonic reconstruction")

        pos_past, neg_past = positive.past_key_values, neg.past_key_values
        pos_hidden = [positive.hidden_states[-1]]
        neg_hidden = [neg.hidden_states[-1]]
        pos_logits, neg_logits = positive.logits[:, -1, :], neg.logits[:, -1, :]
        clean_tokens = []
        for index in range(4):
            # Sampling and both cache advances are always driven by the clean
            # positive branch.  Hidden guidance never changes token sampling.
            token = self._sample(pos_logits)
            clean_tokens.append(token)
            if index < 3:
                po = self._advance(token, pos_past)
                no = self._advance(token, neg_past)
                pos_past, neg_past = po.past_key_values, no.past_key_values
                pos_logits, neg_logits = po.logits[:, -1, :], no.logits[:, -1, :]
                pos_hidden.append(po.hidden_states[-1])
                neg_hidden.append(no.hidden_states[-1])
                del po, no

        pos_latent = torch.cat(pos_hidden, dim=1)
        neg_latent = torch.cat(neg_hidden, dim=1)
        hidden_plus, hidden_minus = pos_latent[:, -4:], neg_latent[:, -4:]
        hidden_star = hidden_plus + lam * (hidden_plus - hidden_minus)
        # The four action-related positions are the final four sequence states:
        # final prompt position plus the first three sampled latent tokens.
        combined = torch.cat(pos_hidden, dim=1)
        combined[:, -4:] = hidden_star
        guided_hidden = [combined]

        decoder = self.model.action_decoder
        original_buffer = decoder.action_buffer.copy()
        original_mask = decoder.action_buffer_mask.copy()
        clean_trace = self._decode_hidden_trace(pos_hidden)
        # Restore the exact incoming trajectory history before the guided call.
        decoder.action_buffer = original_buffer.copy()
        decoder.action_buffer_mask = original_mask.copy()
        guided_trace = self._decode_hidden_trace(guided_hidden)
        # _decode_hidden_trace leaves the guided buffer installed for execution.

        pre_delta = guided_trace["preagg"] - clean_trace["preagg"]
        post_delta = guided_trace["postagg"] - clean_trace["postagg"]
        clean_raw, guided_raw = clean_trace["postagg"], guided_trace["postagg"]
        meta = {
            "arm": arm,
            "guidance_space": "last_hidden_action_states",
            "lambda_h": lam,
            "action_token_ids": clean_tokens,
            "action_tokens_in_learned_range": sum(int(x > 32000) for x in clean_tokens),
            "selected_token_ids": selected,
            "m_t": len(selected),
            "attention_mass": float(attn[selected].sum()),
            "feature_change_norm": float(np.linalg.norm(negative_features - feature)),
            "clean_preagg_raw": clean_trace["preagg"].tolist(),
            "guided_preagg_raw": guided_trace["preagg"].tolist(),
            "clean_postagg_raw": clean_raw.tolist(),
            "guided_postagg_raw": guided_raw.tolist(),
            "preagg_delta_xyz": pre_delta[:3].tolist(),
            "preagg_delta_rot": pre_delta[3:6].tolist(),
            "preagg_delta_grip": float(pre_delta[6]),
            "postagg_delta_xyz": post_delta[:3].tolist(),
            "postagg_delta_rot": post_delta[3:6].tolist(),
            "postagg_delta_grip": float(post_delta[6]),
            "grip_threshold_flip": bool((clean_raw[6] > 0.5) != (guided_raw[6] > 0.5)),
            "preagg_delta_l2": float(np.linalg.norm(pre_delta)),
            "postagg_delta_l2": float(np.linalg.norm(post_delta)),
            "action_hidden_rel_l2": float((torch.linalg.vector_norm((hidden_star - hidden_plus).float()) /
                torch.linalg.vector_norm(hidden_plus.float()).clamp_min(1e-12)).item()),
            "action_hidden_norm_ratio": float((torch.linalg.vector_norm(hidden_star.float()) /
                torch.linalg.vector_norm(hidden_plus.float()).clamp_min(1e-12)).item()),
            **matched,
        }
        return {"action": self._action(guided_raw),
                "raw": {"world_vector": guided_raw[:3].tolist(),
                        "rotation_delta": guided_raw[3:6].tolist(),
                        "open_gripper": guided_raw[6:7].tolist()},
                "meta": meta}


def main():
    sock = os.environ["UNIVLA_HIDDEN_SOCK"]
    policy = HiddenPolicy()
    Path(sock).parent.mkdir(parents=True, exist_ok=True)
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock)
    server.listen(1)
    print(json.dumps({"model_ready": True, "service": "hidden_cd", "selected_layer": 14,
                      "arms": LAMBDAS}), flush=True)
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
                    result = policy.reset(head["instruction"], int(head["seed"]))
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

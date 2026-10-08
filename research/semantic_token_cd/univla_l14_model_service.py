"""UniVLA inference service for L14-Matched harmonic contrastive decoding.

The simulator worker runs under the SAPIEN2 environment; this process runs under
the UniVLA environment. Communication is a private Unix-domain socket per worker.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper

ROOT = Path("/home/leju-suzhou/zjt_ws")
UNIVLA_ROOT = ROOT / "UniVLA"
CKPT = ROOT / "checkpoints/univla-7b-224-sft-simpler-bridge"
SOCK = os.environ["UNIVLA_L14_SOCK"]
sys.path.insert(0, str(UNIVLA_ROOT))
sys.path.insert(0, str(UNIVLA_ROOT / "experiments/robot/simpler-bridge"))

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from simpler_env.policies.univla import univla_model as policy_module


def recv_exact(conn: socket.socket, count: int) -> bytes:
    buf = bytearray()
    while len(buf) < count:
        part = conn.recv(count - len(buf))
        if not part:
            raise EOFError
        buf.extend(part)
    return bytes(buf)


def recv_message(conn: socket.socket):
    n = struct.unpack("!I", recv_exact(conn, 4))[0]
    head = json.loads(recv_exact(conn, n))
    payload = recv_exact(conn, int(head.get("payload_nbytes", 0)))
    return head, payload


def send_message(conn: socket.socket, head: dict):
    body = json.dumps(head, default=str).encode()
    conn.sendall(struct.pack("!I", len(body)) + body)


def harmonic_reconstruct(features: np.ndarray, selected: list[int]) -> np.ndarray:
    selected = sorted(set(int(i) for i in selected))
    if not selected:
        raise ValueError("empty matched mask")
    n, d = len(selected), features.shape[1]
    row_of = {idx: row for row, idx in enumerate(selected)}
    matrix = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros((n, d), dtype=np.float64)
    values = features.astype(np.float64, copy=False)
    for row, token in enumerate(selected):
        r, c = divmod(token, 16)
        neighbors = []
        if r: neighbors.append(token - 16)
        if r < 15: neighbors.append(token + 16)
        if c: neighbors.append(token - 1)
        if c < 15: neighbors.append(token + 1)
        matrix[row, row] = len(neighbors)
        for nb in neighbors:
            col = row_of.get(nb)
            if col is None:
                rhs[row] += values[nb]
            else:
                matrix[row, col] -= 1.0
    out = np.linalg.solve(matrix, rhs).astype(np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite harmonic reconstruction")
    return out


def parse_entities(instruction: str) -> list[str]:
    words = instruction.lower().split()
    determiners = {"the", "a", "an"}
    prepositions = {"into", "in", "on", "onto", "near", "to", "next", "from", "under", "over", "behind", "beside", "above", "below", "inside", "at", "with"}
    rest = words[1:]
    prep = [i for i, word in enumerate(rest) if word in prepositions]
    if not prep:
        source = " ".join(w for w in rest if w not in determiners)
        return [source]
    source = " ".join(w for w in rest[:prep[0]] if w not in determiners)
    target = " ".join(w for w in rest[prep[-1] + 1:] if w not in determiners)
    return list(dict.fromkeys(x for x in (source, target) if x))


class Policy:
    def __init__(self):
        # UniVLA's layer scan used eager attention. Force the same backend here so
        # L14 prompt-query attention is measured by the identical implementation.
        original = policy_module.AutoModelForVision2Seq.from_pretrained
        def eager_load(*args, **kwargs):
            kwargs["attn_implementation"] = "eager"
            return original(*args, **kwargs)
        policy_module.AutoModelForVision2Seq.from_pretrained = eager_load
        self.model = policy_module.UniVLABridgeInference(
            saved_model_path=str(CKPT),
            decoder_path=str(CKPT / "action_decoder.pt"),
            policy_setup="widowx_bridge",
            action_scale=1.0,
            pred_action_horizon=10,
            exec_horizon=1,
        )
        self.vla = self.model.vla.eval()
        self.processor = self.model.processor
        self.device = next(self.vla.parameters()).device
        self.entities = []
        self.entity_emb = []
        self.prompt_offset_cache = {}
        self.temperature = TemperatureLogitsWarper(0.75)
        self.top_p = TopPLogitsWarper(0.9)

    def reset(self, instruction: str, seed: int):
        self.instruction = instruction
        self.entities = parse_entities(instruction)
        embed = self.vla.language_model.model.embed_tokens
        self.entity_emb = []
        for phrase in self.entities:
            ids = self.processor.tokenizer(phrase, add_special_tokens=False)["input_ids"]
            if not ids:
                ids = self.processor.tokenizer(phrase, add_special_tokens=True)["input_ids"]
            tok = torch.tensor([ids], dtype=torch.long, device=self.device)
            self.entity_emb.append(embed(tok)[0].mean(0).detach().float().cpu().numpy())
        self.model.reset(instruction)
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        return {"instruction": instruction, "entities": self.entities}

    def _inputs(self, image: np.ndarray, instruction: str):
        im = cv2.resize(image, tuple(self.model.image_size), interpolation=cv2.INTER_AREA)
        t = tf.convert_to_tensor(im)
        dtype = t.dtype
        t = tf.image.convert_image_dtype(t, tf.float32)
        t = policy_module.crop_and_resize(t, 0.9, 1)
        t = tf.clip_by_value(t, 0, 1)
        t = tf.image.convert_image_dtype(t, dtype, saturate=True)
        pil = Image.fromarray(t.numpy()).convert("RGB")
        text = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(text, pil, return_tensors="pt")
        offsets = self.processor.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")["offset_mapping"][0].tolist()
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        pixels = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)
        # Match UniVLA's predict_latent_action training prompt convention.
        if int(ids[0, -1]) != 29871:
            ids = torch.cat([ids, torch.tensor([[29871]], device=self.device, dtype=ids.dtype)], dim=1)
            mask = torch.cat([mask, torch.ones((1, 1), device=self.device, dtype=mask.dtype)], dim=1)
        char0 = text.index(instruction.lower())
        char1 = char0 + len(instruction.lower())
        special = set(int(x) for x in self.processor.tokenizer.all_special_ids)
        query_ids = []
        for pos, (start, end) in enumerate(offsets):
            if pos < ids.shape[1] - 1 and start < char1 and end > char0 and int(ids[0, pos]) not in special:
                if pos > 0:
                    query_ids.append(pos + 256)
        if not query_ids:
            raise RuntimeError(f"No instruction tokens identified in UniVLA prompt {text!r}")
        return {"input_ids": ids, "attention_mask": mask, "pixel_values": pixels}, query_ids

    @torch.inference_mode()
    def _matched_region(self, features: np.ndarray):
        if features.shape != (256, 4096):
            raise RuntimeError(f"expected projected visual features [256,4096], got {features.shape}")
        labels = KMeans(n_clusters=8, random_state=0, n_init=10).fit(features.astype(np.float64)).labels_
        centers = np.stack([features[labels == k].mean(0) for k in range(8)])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
        selected_groups, scores = [], []
        for entity in self.entity_emb:
            normalized = entity / (np.linalg.norm(entity) + 1e-8)
            sim = centers @ normalized
            group = int(np.argmax(sim))
            scores.append(float(sim[group]))
            if group not in selected_groups:
                selected_groups.append(group)
        selected = sorted(int(i) for group in selected_groups for i in np.flatnonzero(labels == group))
        if not selected:
            raise RuntimeError("matched SHR selector returned an empty mask")
        return selected, {"groups": selected_groups, "entity_scores": scores, "m_t": len(selected)}

    @torch.inference_mode()
    def _sample(self, logits: torch.Tensor) -> int:
        scores = self.temperature(None, logits.float())
        scores = self.top_p(None, scores)
        eos = self.vla.generation_config.eos_token_id
        if eos is not None:
            for token in ([eos] if isinstance(eos, int) else eos):
                if 0 <= int(token) < scores.shape[-1]:
                    scores[:, int(token)] = torch.finfo(scores.dtype).min
        probs = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, num_samples=1)[0, 0])

    @torch.inference_mode()
    def _forward_prefill(self, inputs, replacement=None, attentions=False):
        if replacement is None:
            return self.vla(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"], use_cache=True,
                output_attentions=attentions, output_hidden_states=True, return_dict=True,
            )
        handle = None
        def replace(_module, _args, output):
            h = output.clone()
            idx = torch.tensor(replacement[0], device=h.device, dtype=torch.long)
            vals = torch.tensor(replacement[1], device=h.device, dtype=h.dtype)
            h[:, idx] = vals
            return h
        handle = self.vla.projector.register_forward_hook(replace)
        try:
            return self.vla(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"], use_cache=True,
                output_attentions=False, output_hidden_states=True, return_dict=True,
            )
        finally:
            handle.remove()

    @torch.inference_mode()
    def _advance(self, token: int, past):
        ids = torch.tensor([[token]], device=self.device, dtype=torch.long)
        return self.vla.language_model(
            input_ids=ids, past_key_values=past, use_cache=True,
            output_hidden_states=True, output_attentions=False, return_dict=True,
        )

    @torch.inference_mode()
    def _decode_hidden(self, hidden_states):
        latent_tokens = torch.cat(hidden_states, dim=1)
        visual_embed = latent_tokens[:, :256]
        latent_tokens = latent_tokens[:, 256:]
        latent_action = latent_tokens[:, -4:]
        stats = self.vla.get_action_stats(self.model.unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        raw = self.model.action_decoder(
            latent_action, visual_embed, mask, np.asarray(stats["q01"]), np.asarray(stats["q99"])
        )
        raw_action = {
            "world_vector": np.asarray(raw[:3]),
            "rotation_delta": np.asarray(raw[3:6]),
            "open_gripper": np.asarray(raw[6:7]),
        }
        from transforms3d.euler import euler2axangle
        roll, pitch, yaw = raw_action["rotation_delta"]
        axis, angle = euler2axangle(roll, pitch, yaw)
        action = {
            "world_vector": raw_action["world_vector"].tolist(),
            "rot_axangle": (axis * angle).tolist(),
            "gripper": [float(2.0 * (raw_action["open_gripper"][0] > 0.5) - 1.0)],
            "terminate_episode": [0.0],
        }
        return {k: np.asarray(v).tolist() for k, v in raw_action.items()}, action

    @torch.inference_mode()
    def step(self, image: np.ndarray, arm: str, instruction: str):
        inputs, queries = self._inputs(image, instruction)
        positive = self._forward_prefill(inputs, attentions=(arm == "l14_matched"))
        pos_past = positive.past_key_values
        hidden_states = [positive.hidden_states[-1]]
        scores_meta = {"arm": arm, "selected_token_ids": [], "m_t": 0}
        neg_past = None
        if arm == "l14_matched":
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
            neg_past = neg.past_key_values
            nfeatures = neg.projector_features[0].detach().float().cpu().numpy()
            expected_features = torch.from_numpy(negative_features).to(torch.bfloat16).float().numpy()
            if not np.array_equal(expected_features, nfeatures):
                raise RuntimeError("negative projector features do not match the harmonic reconstruction")
            scores_meta.update(matched)
            scores_meta["selected_token_ids"] = selected
            scores_meta["attention_mass"] = float(attn[selected].sum())
            scores_meta["feature_change_norm"] = float(np.linalg.norm(negative_features - feature))
            neg_logits = neg.logits[:, -1, :]
            del neg
        elif arm == "vanilla":
            neg_logits = None
        else:
            raise ValueError(f"unknown arm {arm}")

        token_ids = []
        pos_logits = positive.logits[:, -1, :]
        final_hidden = hidden_states
        for action_token_idx in range(4):
            if arm == "l14_matched":
                guided_logits = 1.5 * pos_logits - 0.5 * neg_logits
                token = self._sample(guided_logits)
            else:
                token = self._sample(pos_logits)
            token_ids.append(token)
            if action_token_idx < 3:
                pos_out = self._advance(token, pos_past)
                pos_past = pos_out.past_key_values
                pos_logits = pos_out.logits[:, -1, :]
                final_hidden.append(pos_out.hidden_states[-1])
                del pos_out
                if arm == "l14_matched":
                    neg_out = self._advance(token, neg_past)
                    neg_past = neg_out.past_key_values
                    neg_logits = neg_out.logits[:, -1, :]
                    del neg_out

        raw_action, action = self._decode_hidden(final_hidden)
        scores_meta["action_token_ids"] = token_ids
        scores_meta["action_tokens_in_learned_range"] = sum(int(x > 32000) for x in token_ids)
        return {"action": action, "raw": raw_action, "meta": scores_meta}


def main():
    policy = Policy()
    Path(SOCK).parent.mkdir(parents=True, exist_ok=True)
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCK)
    server.listen(1)
    print(json.dumps({"model_ready": True, "attention": "eager", "layers": 32, "selected_layer": 14}), flush=True)
    conn, _ = server.accept()
    with conn:
        while True:
            try:
                head, payload = recv_message(conn)
            except EOFError:
                break
            cmd = head.get("cmd")
            try:
                if cmd == "quit":
                    send_message(conn, {"ok": True})
                    break
                if cmd == "reset":
                    result = policy.reset(head["instruction"], int(head["seed"]))
                    send_message(conn, {"ok": True, "result": result})
                elif cmd == "step":
                    image = np.frombuffer(payload, dtype=np.uint8).reshape(head["shape"])
                    result = policy.step(image, head["arm"], head["instruction"])
                    send_message(conn, {"ok": True, "result": result})
                else:
                    raise ValueError(f"unknown command {cmd}")
            except Exception as exc:
                traceback.print_exc()
                send_message(conn, {"ok": False, "error": repr(exc)})
    server.close()
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()

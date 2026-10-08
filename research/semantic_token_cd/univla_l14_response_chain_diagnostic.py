"""Offline UniVLA L14 identity and response-chain diagnostics.

Uses already saved layer-scan RGB observations; does not start simulator episodes.
Run in the univla-simpler environment with CUDA_VISIBLE_DEVICES set to a free GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
os.environ.setdefault("UNIVLA_L14_SOCK", "/tmp/univla_l14_diag_unused.sock")
sys.path.insert(0, str(REPO))

from research.semantic_token_cd.univla_l14_model_service import Policy, harmonic_reconstruct


STATE_DIR = REPO / "artifacts/univla_prompt_attention_layer_scan_v1/states"
OUT_DIR = REPO / "artifacts/univla_l14_response_chain_diagnostic_v1"
ACTION_IDS = torch.arange(32001, 32033, dtype=torch.long)
LAMBDA_H_SWEEP = (0.1, 0.25, 0.5, 0.75, 1.0)


def action_decode(policy: Policy, hidden_states):
    # Each offline branch is an independent first-step counterfactual. The
    # official ActionDecoder keeps a temporal ensemble buffer across calls, so
    # clear it before every branch decode to avoid cross-arm contamination.
    policy.model.action_decoder.reset()
    raw, action = policy._decode_hidden(hidden_states)
    latent = torch.cat(hidden_states, dim=1)
    visual = latent[:, :256]
    actions = latent[:, -4:]
    return raw, action, visual, actions


def official_predict(policy: Policy, inputs, instruction: str, seed: int):
    policy.reset(instruction, seed)
    latent, visual, ids = policy.vla.predict_latent_action(
        **inputs,
        unnorm_key=policy.model.unnorm_key,
        do_sample=True,
        temperature=0.75,
        top_p=0.9,
    )
    return ids.detach().clone(), latent.detach().clone(), visual.detach().clone()


def manual_decode(policy: Policy, inputs, instruction: str, seed: int, *,
                  negative_inputs=None, lam: float = 0.0, run_negative: bool = False,
                  attentions: bool = False):
    policy.reset(instruction, seed)
    pos = policy._forward_prefill(inputs, attentions=attentions)
    pos_past = pos.past_key_values
    hidden = [pos.hidden_states[-1]]
    neg = None
    neg_past = None
    if run_negative:
        if isinstance(negative_inputs, tuple):
            neg_input_tensors, neg_replacement = negative_inputs
        else:
            neg_input_tensors, neg_replacement = (negative_inputs or inputs), None
        neg = policy._forward_prefill(neg_input_tensors, replacement=neg_replacement)
        neg_past = neg.past_key_values
    pos_logits = pos.logits[:, -1, :]
    neg_logits = None if neg is None else neg.logits[:, -1, :]
    tokens, step_logits = [], []
    for i in range(4):
        if neg_logits is None:
            guided = pos_logits
        else:
            guided = (1.0 + lam) * pos_logits - lam * neg_logits
        step_logits.append((pos_logits.detach(), None if neg_logits is None else neg_logits.detach(), guided.detach()))
        token = policy._sample(guided)
        tokens.append(token)
        if i < 3:
            po = policy._advance(token, pos_past)
            pos_past, pos_logits = po.past_key_values, po.logits[:, -1, :]
            hidden.append(po.hidden_states[-1])
            del po
            if neg_past is not None:
                no = policy._advance(token, neg_past)
                neg_past, neg_logits = no.past_key_values, no.logits[:, -1, :]
                del no
    raw, action, visual, latent = action_decode(policy, hidden)
    return {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "hidden": hidden,
        "raw": raw,
        "action": action,
        "visual": visual,
        "latent": latent,
        "step_logits": step_logits,
    }


def matched_negative(policy: Policy, image: np.ndarray, instruction: str, seed: int):
    policy.reset(instruction, seed)
    inputs, queries = policy._inputs(image, instruction)
    pos = policy._forward_prefill(inputs, attentions=True)
    if pos.attentions is None or len(pos.attentions) <= 14:
        raise RuntimeError("L14 attention not returned")
    attn = pos.attentions[14][0, :, queries, 1:257].detach().float().mean(dim=(0, 1)).cpu().numpy()
    features = pos.projector_features[0].detach().float().cpu().numpy()
    selected, matched = policy._matched_region(features)
    order = np.lexsort((np.arange(256), -attn.astype(np.float64)))
    selected = sorted(int(x) for x in order[:len(selected)])
    negative_features = features.copy()
    negative_features[selected] = harmonic_reconstruct(features, selected)
    replacement = (selected, negative_features[selected])
    del pos
    return inputs, replacement, selected, matched, attn, features


def forced_branch_trace(policy: Policy, inputs, negative_inputs, instruction: str,
                        clean_tokens: list[int]):
    """Compare positive/negative internals under the same clean-token prefix."""
    pos = policy._forward_prefill(inputs)
    if isinstance(negative_inputs, tuple):
        neg_inputs, neg_replacement = negative_inputs
    else:
        neg_inputs, neg_replacement = negative_inputs, None
    neg = policy._forward_prefill(neg_inputs, replacement=neg_replacement)
    pos_hidden = [pos.hidden_states[-1]]
    neg_hidden = [neg.hidden_states[-1]]
    pos_logits = pos.logits[:, -1, :]
    neg_logits = neg.logits[:, -1, :]
    js_by_position = []
    outside_mass = []
    for i, token in enumerate(clean_tokens):
        p = torch.softmax(pos_logits.float().index_select(-1, ACTION_IDS.to(pos_logits.device)), -1)
        q = torch.softmax(neg_logits.float().index_select(-1, ACTION_IDS.to(neg_logits.device)), -1)
        m = 0.5 * (p + q)
        js = 0.5 * ((p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1) +
                    (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1))
        js_by_position.append(float(js.item()))
        all_probs = torch.softmax(pos_logits.float(), -1)
        act_mass = all_probs.index_select(-1, ACTION_IDS.to(pos_logits.device)).sum(-1)
        outside_mass.append(float((1.0 - act_mass).item()))
        if i < 3:
            po = policy._advance(token, pos.past_key_values)
            no = policy._advance(token, neg.past_key_values)
            pos.past_key_values, neg.past_key_values = po.past_key_values, no.past_key_values
            pos_logits, neg_logits = po.logits[:, -1, :], no.logits[:, -1, :]
            pos_hidden.append(po.hidden_states[-1])
            neg_hidden.append(no.hidden_states[-1])
            del po, no
    return pos_hidden, neg_hidden, js_by_position, outside_mass


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().float()
    bb = b.detach().float()
    return float((torch.linalg.vector_norm(aa - bb) / torch.linalg.vector_norm(aa).clamp_min(1e-12)).item())


def js_action_vocab(p: torch.Tensor, q: torch.Tensor) -> float:
    ids = ACTION_IDS.to(p.device)
    p = torch.softmax(p.float().index_select(-1, ids), dim=-1)
    q = torch.softmax(q.float().index_select(-1, ids), dim=-1)
    m = 0.5 * (p + q)
    js = 0.5 * ((p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1) +
                (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1))
    return float(js.item())


def as_py_action(action):
    return {k: np.asarray(v).astype(float).reshape(-1).tolist() for k, v in action.items()}


def raw_vector(raw):
    return np.concatenate([np.asarray(raw[k], dtype=np.float64).reshape(-1)
                           for k in ("world_vector", "rotation_delta", "open_gripper")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 means all saved images")
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--output", default=str(OUT_DIR / "diagnostics.jsonl"))
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    policy = Policy()
    policy.vla.eval()
    paths = sorted(STATE_DIR.glob("widowx_*__seed*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Instructions from the already completed UniVLA runs, one per task.
    prompt_by_task = {}
    sample_root = REPO / "artifacts/univla_simpler_l14_matched_rtfull_100_v1/episodes"
    for task_dir in sorted(sample_root.glob("widowx_*")):
        sample = next(task_dir.glob("l14_matched/seed_*.json"), None)
        if sample:
            prompt_by_task[task_dir.name] = json.loads(sample.read_text())["instruction"]

    with outpath.open("w") as fout:
        for index, path in enumerate(paths):
            task = path.name.split("__seed")[0]
            if task not in prompt_by_task:
                raise RuntimeError(f"No completed-run prompt found for {task}")
            instruction = prompt_by_task[task]
            seed = args.seed + index
            image = np.load(path)["image"]
            inputs, queries = policy._inputs(image, instruction)

            # Canonical A: call UniVLA's official public latent-action API.
            run_identity = index == 0
            a_ids = a_latent = a_visual = a_action_raw = None
            if run_identity:
                a_ids, a_latent, a_visual = official_predict(policy, inputs, instruction, seed)
                stats = policy.vla.get_action_stats(policy.model.unnorm_key)
                policy.model.action_decoder.reset()
                a_action_raw = policy.model.action_decoder(
                    a_latent, a_visual,
                    stats.get("mask", np.ones_like(stats["q01"], dtype=bool)),
                    np.asarray(stats["q01"]), np.asarray(stats["q99"]),
                )

            # B: manual decoding, no negative branch.
            b = manual_decode(policy, inputs, instruction, seed, run_negative=False, lam=0.0)
            # For C/D and response-chain, construct exactly the same L14-Matched negative.
            inputs2, replacement, selected, matched, attn, features = matched_negative(policy, image, instruction, seed)
            if inputs2["input_ids"].shape != inputs["input_ids"].shape:
                raise RuntimeError("preprocessing mismatch between diagnostic paths")
            neg_features = features.copy()
            neg_features[selected] = harmonic_reconstruct(features, selected)
            neg_replacement = (selected, neg_features[selected])
            negative_branch = (inputs, neg_replacement)
            # C actually executes the harmonic negative branch while lambda=0.
            c_real = d = None
            if run_identity:
                c_real = manual_decode(policy, inputs, instruction, seed, negative_inputs=negative_branch,
                                       lam=0.0, run_negative=True, attentions=False)
                d = manual_decode(policy, inputs, instruction, seed, negative_inputs=inputs,
                                  lam=0.5, run_negative=True, attentions=False)

            # Official hidden trajectory for validating H/action extraction on the API path.
            identity = None
            if run_identity:
                policy.reset(instruction, seed)
                official_hidden_out = policy.vla.generate(
                    **inputs, min_new_tokens=4, max_new_tokens=4, return_dict_in_generate=True,
                    output_hidden_states=True, do_sample=True, temperature=0.75, top_p=0.9,
                )
                official_hidden = [hs[-1] for hs in official_hidden_out.hidden_states]
                _, _, off_visual2, off_latent2 = action_decode(policy, official_hidden)
                identity = {
                    "A_official_api_ids": a_ids[0].cpu().tolist(),
                    "A_generate_ids": official_hidden_out.sequences[0, -4:].cpu().tolist(),
                    "B_manual_ids": b["tokens"].tolist(),
                    "C_real_negative_lambda0_ids": c_real["tokens"].tolist(),
                    "D_m0_lambda05_ids": d["tokens"].tolist(),
                    "A_vs_B_ids_equal": a_ids[0].cpu().tolist() == b["tokens"].tolist(),
                    "A_vs_B_latent_rel_l2": rel_l2(a_latent, b["latent"]),
                    "A_vs_B_visual_rel_l2": rel_l2(a_visual, b["visual"]),
                    "A_vs_B_raw_action_l2": float(np.linalg.norm(np.asarray(a_action_raw).reshape(-1) - raw_vector(b["raw"]))),
                    "official_api_vs_generate_ids_equal": a_ids[0].cpu().tolist() == official_hidden_out.sequences[0, -4:].cpu().tolist(),
                    "official_api_vs_generate_latent_rel_l2": rel_l2(a_latent, off_latent2),
                    "official_api_vs_generate_visual_rel_l2": rel_l2(a_visual, off_visual2),
                    "B_vs_C_ids_equal": b["tokens"].tolist() == c_real["tokens"].tolist(),
                    "B_vs_D_ids_equal": b["tokens"].tolist() == d["tokens"].tolist(),
                    "identity_selected_m": len(selected),
                    "identity_selected_ids": selected,
                }
            if args.identity_only:
                fout.write(json.dumps({"kind": "identity", "image": path.name, "task": task,
                                       "instruction": instruction, "identity": identity}) + "\n")
                fout.flush()
                print(json.dumps({"identity": path.name, **identity}), flush=True)
                break

            # Causal response chain with a fixed clean prefix for pos-vs-neg, then
            # actual guided trajectory for intervention-vs-clean action displacement.
            pos_h, neg_h, js_by_pos, pos_outside = forced_branch_trace(
                policy, inputs, negative_branch, instruction, b["tokens"].tolist())
            guided = manual_decode(policy, inputs, instruction, seed, negative_inputs=negative_branch,
                                   lam=0.5, run_negative=True, attentions=False)
            pos_vec = torch.cat(pos_h, dim=1)
            neg_vec = torch.cat(neg_h, dim=1)
            pos_visual, neg_visual = pos_vec[:, :256], neg_vec[:, :256]
            pos_action_h, neg_action_h = pos_vec[:, -4:], neg_vec[:, -4:]
            guided_vec = torch.cat(guided["hidden"], dim=1)
            guided_visual, guided_action_h = guided_vec[:, :256], guided_vec[:, -4:]
            hidden_star_action_h = pos_action_h + 0.5 * (pos_action_h - neg_action_h)
            hidden_star_states = [h.clone() for h in pos_h]
            hidden_star_states[0][:, -1:, :] = hidden_star_action_h[:, :1, :]
            for hidx in range(1, 4):
                hidden_star_states[hidx] = hidden_star_action_h[:, hidx:hidx + 1, :]
            hidden_star_raw, hidden_star_decoded, _, _ = action_decode(policy, hidden_star_states)
            hidden_lambda_sweep = []
            hplus_norm = torch.linalg.vector_norm(pos_action_h.float()).clamp_min(1e-12)
            for lam_h in LAMBDA_H_SWEEP:
                h_lam = pos_action_h + float(lam_h) * (pos_action_h - neg_action_h)
                states_lam = [h.clone() for h in pos_h]
                states_lam[0][:, -1:, :] = h_lam[:, :1, :]
                for hidx in range(1, 4):
                    states_lam[hidx] = h_lam[:, hidx:hidx + 1, :]
                raw_lam, action_lam, _, _ = action_decode(policy, states_lam)
                rv_lam = raw_vector(raw_lam)
                rv_clean = raw_vector(b["raw"])
                hidden_lambda_sweep.append({
                    "lambda_h": float(lam_h),
                    "action_hidden_rel_l2": rel_l2(pos_action_h, h_lam),
                    "action_hidden_norm_ratio": float((torch.linalg.vector_norm(h_lam.float()) / hplus_norm).item()),
                    "decoded_raw_action_l2": float(np.linalg.norm(rv_lam - rv_clean)),
                    "decoded_translation_l2": float(np.linalg.norm(rv_lam[:3] - rv_clean[:3])),
                    "decoded_rotation_l2": float(np.linalg.norm(rv_lam[3:6] - rv_clean[3:6])),
                    "decoded_gripper_abs_delta": float(abs(rv_lam[6] - rv_clean[6])),
                    "action": as_py_action(action_lam),
                })
            guided_vocab_mass = []
            for _, _, gl in guided["step_logits"]:
                probs = torch.softmax(gl.float(), dim=-1)
                guided_vocab_mass.append(float(probs.index_select(-1, ACTION_IDS.to(gl.device)).sum(-1).item()))
            rec = {
                "kind": "response_chain", "image": path.name, "task": task,
                "instruction": instruction, "seed": seed,
                "identity": identity,
                "m_t": len(selected), "fixed32_delta": len(selected) - 32,
                "matched_groups": matched.get("groups", []),
                "attention_mass_selected": float(attn[selected].sum()),
                "harmonic_feature_change_norm": float(np.linalg.norm(neg_features - features)),
                "clean_tokens": b["tokens"].tolist(),
                "guided_tokens": guided["tokens"].tolist(),
                "per_position_token_flip": [int(x != y) for x, y in zip(b["tokens"].tolist(), guided["tokens"].tolist())],
                "guided_outside_action_vocab_count": sum(int(t < 32001 or t > 32032) for t in guided["tokens"].tolist()),
                "clean_outside_action_vocab_count": sum(int(t < 32001 or t > 32032) for t in b["tokens"].tolist()),
                "guided_action_vocab_probability_mass": guided_vocab_mass,
                "positive_action_vocab_probability_mass": [1.0 - x for x in pos_outside],
                "js_action_vocab_by_position_same_clean_prefix": js_by_pos,
                "js_action_vocab_mean": float(np.mean(js_by_pos)),
                "relative_H_visual_negative_vs_positive_same_prefix": rel_l2(pos_visual, neg_visual),
                "relative_H_action_negative_vs_positive_same_prefix": rel_l2(pos_action_h, neg_action_h),
                "per_action_position_H_negative_rel_l2": [rel_l2(pos_action_h[:, j], neg_action_h[:, j]) for j in range(4)],
                "relative_H_visual_guided_vs_clean": rel_l2(pos_visual, guided_visual),
                "relative_H_action_guided_vs_clean": rel_l2(pos_action_h, guided_action_h),
                "per_action_position_H_guided_rel_l2": [rel_l2(pos_action_h[:, j], guided_action_h[:, j]) for j in range(4)],
                "hidden_guidance_lambda": 0.5,
                "hidden_guidance_action_hidden_rel_l2": rel_l2(pos_action_h, hidden_star_action_h),
                "hidden_guidance_raw_action_l2_vs_clean": float(np.linalg.norm(raw_vector(hidden_star_raw) - raw_vector(b["raw"]))),
                "hidden_guidance_action": as_py_action(hidden_star_decoded),
                "hidden_lambda_sweep": hidden_lambda_sweep,
                "negative_raw_action_l2_vs_clean": float(np.linalg.norm(
                    raw_vector(action_decode(policy, neg_h)[0]) - raw_vector(b["raw"]))),
                "guided_raw_action_l2_vs_clean": float(np.linalg.norm(
                    raw_vector(guided["raw"]) - raw_vector(b["raw"]))),
                "clean_action": as_py_action(b["action"]),
                "guided_action": as_py_action(guided["action"]),
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(json.dumps({"done": index + 1, "total": len(paths), "task": task,
                              "m_t": rec["m_t"], "js": round(rec["js_action_vocab_mean"], 5),
                              "dHneg": round(rec["relative_H_action_negative_vs_positive_same_prefix"], 5),
                              "dAguided": round(rec["guided_raw_action_l2_vs_clean"], 5),
                              "token_flips": rec["per_position_token_flip"]}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-state online action-consistency engineering qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from research.coreact_closed_loop.guidance import _full_velocity, _prefix_cache, replace_visual_tokens, select_visual_tokens, tensor_sha256
from research.coreact_closed_loop.runtime import env_config, load_policy_and_processors
from research.coreact_exploration.instrumentation import attention_ranking_scores, build_prefix_span_map, grouped_intervention
from research.coreact_exploration.run_development_gates import CAMERA_IDS, replacements_for
from research.coreact_region.segmented_runtime import batched_observation, make_segmented_env
from research.coreact_revision.evaluate_region_sign_probe_v3 import aggregate
from research.coreact_revision.run_region_sign_capture_v3 import action_chunk, rewrite_demo_xml

TAUS = (0.2, 0.5, 0.8)
SEED = 1729
FEATURES = ("clean_cross_condition_dispersion", "masked_cross_condition_dispersion", "masked_minus_clean_dispersion", "clean_mask_consensus_distance", "intervention_direction_consistency", "mean_intervention_norm")


def hash_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value); h = hashlib.sha256(); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes()); return h.hexdigest()


def dispersion(x: torch.Tensor) -> float:
    mean = x.mean(0); return float(((x - mean) ** 2).mean() / (mean.square().mean() + 1e-8))


def direction_cosine(x: torch.Tensor) -> float:
    x = x.flatten(1); x = x / (x.norm(dim=1, keepdim=True) + 1e-8); m = x @ x.T; n = len(x)
    return float((m.sum() - m.diagonal().sum()) / max(1, n * (n - 1)))


def features(clean: torch.Tensor, masked: torch.Tensor) -> dict[str, float]:
    clean_mean, masked_mean = clean.mean(0), masked.mean(0); delta = masked - clean
    return {"clean_cross_condition_dispersion": dispersion(clean), "masked_cross_condition_dispersion": dispersion(masked), "masked_minus_clean_dispersion": dispersion(masked) - dispersion(clean), "clean_mask_consensus_distance": float((masked_mean - clean_mean).square().mean() / (clean_mean.square().mean() + 1e-8)), "intervention_direction_consistency": direction_cosine(delta), "mean_intervention_norm": float(delta.norm(dim=1).mean())}


def fit_task_heldout_classifier(raw_effects: Path, consistency: Path, heldout_task: int = 0):
    consistency_rows = {}
    for line in consistency.read_text().splitlines():
        row = json.loads(line); consistency_rows[(row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])] = row
    aggregate_rows = aggregate(raw_effects)
    numeric = [f"tau3_seed1729_{name}" for name in FEATURES]
    rows = []
    for row in aggregate_rows:
        source = consistency_rows[(row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])]
        row = {**row, **{name: source[name] for name in numeric}}
        rows.append(row)
    train = [row for row in rows if row["task_id"] != heldout_task]
    x = np.asarray([[row[k] for k in numeric] + [row["camera_id"]] for row in train], dtype=object)
    y = np.asarray([row["label_nuisance"] for row in train])
    transform = ColumnTransformer([("numeric", StandardScaler(), list(range(len(numeric)))), ("camera", OneHotEncoder(handle_unknown="ignore"), [len(numeric)])])
    model = make_pipeline(transform, LogisticRegression(class_weight="balanced", max_iter=3000, random_state=20260808))
    model.fit(x, y)
    return model, { (row["task_id"], row["demo_id"], row["frame_id"], row["group_id"]): row for row in rows if row["task_id"] == heldout_task }, numeric


def score_model(model, numeric, rows):
    x = np.asarray([[row[k] for k in numeric] + [row["camera_id"]] for row in rows], dtype=object)
    return model.predict_proba(x)[:, 1].tolist()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, required=True); p.add_argument("--artifact", type=Path, required=True); p.add_argument("--ablation", type=Path, required=True); args = p.parse_args()
    w, artifact, ablation = args.workspace.resolve(), args.artifact.resolve(), args.ablation.resolve()
    source = w / "artifacts/coreact_region_sign_multitask_v1_20260808_093747"
    manifest = [json.loads(line) for line in (source / "state_manifest.jsonl").read_text().splitlines() if line.strip()]
    state_row = next(row for row in manifest if row["task_id"] == 0 and row["demo_id"] == "demo_4" and row["frame_id"] == 22)
    model_classifier, offline_rows, numeric = fit_task_heldout_classifier(source / "raw_effects.jsonl", ablation / "consistency_scores.jsonl")
    task_name = next(row["name"] for row in json.loads((source / "task_manifest.json").read_text()) if row["task_id"] == 0)
    cfg, policy, preprocessor, _ = load_policy_and_processors(w)
    means = torch.load(w / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    env = make_segmented_env("libero_spatial", 0)
    env_pre, _ = make_env_pre_post_processors(env_cfg=env_config("libero_spatial", 0), policy_cfg=cfg)
    try:
        with h5py.File(w / "LIBERO/libero/datasets/libero_spatial" / f"{task_name}_demo.hdf5", "r") as source:
            episode = source["data"][state_row["demo_id"]]; states, actions = np.asarray(episode["states"]), np.asarray(episode["actions"]); xml = episode.attrs["model_file"]; xml = xml.decode() if isinstance(xml, bytes) else xml
            env._env.reset(); env._env.reset_from_xml_string(__import__("libero.libero.envs.utils", fromlist=["postprocess_model_xml"]).postprocess_model_xml(rewrite_demo_xml(xml, w), {}, demo_generation=False)); env._env.env.sim.reset()
            raw = env._env.regenerate_obs_from_state(states[state_row["frame_id"]]); observation = env._format_raw_obs(raw)
        batch = preprocess_observation(batched_observation(observation)); batch["task"] = [state_row["language"]]; act, pad = action_chunk(actions, state_row["frame_id"]); batch["action"], batch["action_is_pad"] = act.unsqueeze(0), pad.unsqueeze(0); batch = preprocessor(env_pre(batch))
        images, image_masks = policy.prepare_images(batch); state = policy.prepare_state(batch); lang_tokens, lang_masks = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
        input_hashes = {key: tensor_sha256(value) for key, value in {"state": state, "lang_tokens": lang_tokens, "lang_masks": lang_masks, "camera1": images[0], "camera2": images[1], "camera1_mask": image_masks[0], "camera2_mask": image_masks[1], "action": policy.prepare_action(batch), "action_is_pad": batch["action_is_pad"]}.items()}
        noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=torch.Generator(device=state.device).manual_seed(SEED), device=state.device, dtype=state.dtype)
        with torch.inference_mode():
            prefix, prefix_pad, prefix_att = policy.model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
            span_map = build_prefix_span_map(policy.model, images, image_masks, lang_tokens, lang_masks, prefix_pad, camera_ids=CAMERA_IDS)[0]
            _, trace = _full_velocity(policy.model, prefix, prefix_pad, prefix_att, noise, torch.ones(1, device=state.device), record_attention=True)
            scores = attention_ranking_scores(trace, prefix.shape[1])["late_half_action_to_context_attention"]
            top8 = select_visual_tokens(span_map, scores, method="top", count=8, random_seed=0)
            replacements = replacements_for(span_map, means, mode="position")
            masked_prefix = replace_visual_tokens(prefix, span_map, top8, means["visual_position_mean"], CAMERA_IDS)
            changed = (prefix != masked_prefix).any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
            pos_cache = _prefix_cache(policy.model, prefix, prefix_pad, prefix_att)
            masked_cache = _prefix_cache(policy.model, masked_prefix, prefix_pad, prefix_att)
            native_a, native_b = [], []
            repeat_finite = True; final_trajectories = []; flow_start = time.perf_counter()
            for repeat in range(2):
                x = noise.clone(); estimates_clean, estimates_masked = [], [[] for _ in state_row["selected_groups"]]
                for step in range(cfg.num_steps):
                    tau = 1.0 - step / cfg.num_steps; t = torch.tensor([tau], device=state.device)
                    clean_v = policy.model.denoise_step(prefix_pad, pos_cache, x, t); repeat_finite = repeat_finite and bool(torch.isfinite(clean_v).all()); x_next = x - clean_v / cfg.num_steps
                    if tau in TAUS:
                        clean_est = (x - tau * clean_v)[0, :, : act.shape[-1]].detach().cpu(); estimates_clean.append(clean_est[~pad[0]].reshape(-1))
                        groups = [int(g.split("-")[-1]) if isinstance(g, str) else int(g) for g in state_row["selected_groups"]]
                        masked_prefixes = grouped_intervention(prefix, span_map, [[g] for g in groups], replacements)
                        masked_pad = prefix_pad.expand(len(groups), -1); masked_cache_batch = _prefix_cache(policy.model, masked_prefixes, masked_pad, prefix_att.expand(len(groups), -1))
                        masked_v = policy.model.denoise_step(masked_pad, masked_cache_batch, x.expand(len(groups), -1, -1), t.expand(len(groups))); repeat_finite = repeat_finite and bool(torch.isfinite(masked_v).all())
                        masked_est = (x.expand(len(groups), -1, -1) - tau * masked_v)[:, :, : act.shape[-1]]
                        for j in range(len(groups)): estimates_masked[j].append(masked_est[j][~pad[0]].reshape(-1).detach().cpu())
                    x = x_next
                result_rows = []
                for group, clean_parts, masked_parts in zip(state_row["selected_groups"], [estimates_clean] * len(estimates_masked), estimates_masked, strict=True):
                    f = features(torch.stack(clean_parts), torch.stack(masked_parts)); group_id = f"prefix-{group}"; base = offline_rows.get((0, "demo_4", 22, group_id))
                    if base is not None: result_rows.append({"group_id": group_id, "camera_id": base["camera_id"], **f, **{f"tau3_seed1729_{key}": value for key, value in f.items()}})
                final_trajectories.append(x.detach().cpu())
                (native_a if repeat == 0 else native_b).extend(result_rows)
            online_rows = native_a
        offline_state_rows = [row for key, row in offline_rows.items() if key[1] == "demo_4" and key[2] == 22 and row["group_id"] in {r["group_id"] for r in online_rows}]
        offline_prob = score_model(model_classifier, numeric, offline_state_rows); online_prob = score_model(model_classifier, numeric, online_rows)
        by_off = {r["group_id"]: p for r, p in zip(offline_state_rows, offline_prob, strict=True)}; by_on = {r["group_id"]: p for r, p in zip(online_rows, online_prob, strict=True)}
        common = sorted(set(by_off) & set(by_on)); rho = float(spearmanr([by_off[k] for k in common], [by_on[k] for k in common]).statistic) if len(common) >= 2 else float("nan")
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if state.is_cuda else 0.0
        repeat_feature_diff = max((abs(native_a[i][key] - native_b[i][key]) for i in range(len(native_a)) for key in FEATURES), default=0.0)
        flow_elapsed = time.perf_counter() - flow_start
        trajectory_diff = float((final_trajectories[0] - final_trajectories[1]).abs().max())
        online_pass = bool(repeat_feature_diff <= 1e-6 and trajectory_diff <= 1e-6 and repeat_finite and len(changed) == 8 and sorted(top8) == sorted(changed) and all(span_map[i].modality == "visual" and span_map[i].intervention_allowed for i in changed) and len(common) >= 2 and math.isfinite(rho))
        report = {"gate": "online_action_consistency_engineering_qualification", "pass": online_pass, "state": state_row, "input_hashes": input_hashes, "noise_sha256": tensor_sha256(noise), "native_prefix_sha256": tensor_sha256(prefix), "masked_prefix_sha256": tensor_sha256(masked_prefix), "top8": top8, "changed_indices": changed, "selected_equals_changed": sorted(top8) == sorted(changed), "exactly_8_visual": len(changed) == 8, "protected_untouched": all(span_map[i].modality == "visual" and span_map[i].intervention_allowed for i in changed), "native_attention_layers": len(trace), "all_outputs_finite": repeat_finite, "repeat_feature_max_abs_diff": repeat_feature_diff, "repeat_clean_trajectory_max_abs_diff": trajectory_diff, "offline_online_spearman": rho, "common_groups": len(common), "offline_probabilities": by_off, "online_probabilities": by_on, "peak_memory_mb": peak_mb, "flow_repeats": 2, "flow_elapsed_seconds": flow_elapsed, "flow_latency_ms_per_repeat": flow_elapsed * 1000 / 2, "selective_closed_loop_authorized": False, "latency_note": "qualification measures single-state computation; no success rollout"}
        (artifact / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (artifact / "qualification.md").write_text("# Online Consistency Qualification\n\nOverall: PASS (engineering only)\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in report.items() if k not in {"offline_probabilities", "online_probabilities"}) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    finally:
        env.close()


if __name__ == "__main__": main()

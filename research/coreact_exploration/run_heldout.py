#!/usr/bin/env python3
"""Append-only two-suite held-out grouped-intervention runner."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from research.coreact_exploration.instrumentation import (
    attention_ranking_scores,
    grouped_intervention,
    predict_teacher_forced_velocity,
)
from research.coreact_exploration.metrics import intervention_metrics
from research.coreact_exploration.qualify_single_suite import CHECKPOINT_REVISION, Paths, load_preprocessor
from research.coreact_exploration.run_development_gates import (
    CAMERA_IDS,
    SnapshotRows,
    load_policy,
    processed_state,
    replacements_for,
)


BATCH_SIZE = 16


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_within(values, indices):
    output = torch.zeros_like(values, dtype=torch.float32)
    selected = values[indices].float()
    std = selected.std(unbiased=False).clamp_min(1e-12)
    output[indices] = (selected - selected.mean()) / std
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    lock = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    manifest = [json.loads(line) for line in (artifact / "sample_manifest.jsonl").read_text().splitlines()]
    heldout = [row for row in manifest if row["split"] == "heldout"]
    if len(heldout) != 120:
        raise ValueError("locked held-out manifest does not contain 120 states")
    raw_path = artifact / "raw_effects.jsonl"
    completed = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            row = json.loads(line)
            completed.add(
                (
                    row["episode_id"], row["frame_id"], row["tau"], row["noise_seed"],
                    row["group_id"], row["replacement_type"],
                )
            )
    dataset = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero"
    paths = Paths(
        workspace,
        dataset,
        workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots" / CHECKPOINT_REVISION,
        workspace / "counterfactual-flow-vla/pi05_svcpd_p0/verified_calibration_download_manifest.json",
    )
    store, preprocessor, policy = SnapshotRows(dataset), load_preprocessor(paths), load_policy(paths)
    means = torch.load(artifact / "modality_means.pt", map_location="cpu", weights_only=True)
    conditions = [(tau, seed) for tau in lock["tau_values"] for seed in lock["noise_seeds"]]
    expected_rows = 0
    started_all = time.perf_counter()
    for state_index, state_row in enumerate(heldout):
        batch = processed_state(store, preprocessor, state_row)
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        actions = policy.prepare_action(batch)
        base_by_condition, score_by_condition = {}, []
        for tau_value, noise_seed in conditions:
            generator = torch.Generator(device=actions.device).manual_seed(noise_seed)
            noise = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
            tau = torch.tensor([tau_value], device=actions.device, dtype=actions.dtype)
            with torch.inference_mode():
                base = predict_teacher_forced_velocity(
                    policy.model,
                    images,
                    image_masks,
                    batch["observation.language.tokens"],
                    batch["observation.language.attention_mask"],
                    state,
                    actions,
                    tau,
                    noise,
                    record_attention=True,
                    camera_ids=CAMERA_IDS,
                )
            base_by_condition[(tau_value, noise_seed)] = (base, noise, tau)
            score_by_condition.append(attention_ranking_scores(base["attention_trace"], len(base["prefix_span_map"][0])))
        span_map = base_by_condition[conditions[0]][0]["prefix_span_map"][0]
        visual = [token["index"] for token in span_map if token["intervention_allowed"] and token["modality"] == "visual"]
        language = [token["index"] for token in span_map if token["intervention_allowed"] and token["modality"] == "language"]
        aggregate_last = torch.stack([score["raw_last_expert_layer_attention"] for score in score_by_condition]).median(0).values
        aggregate_late = torch.stack([score["late_half_action_to_context_attention"] for score in score_by_condition]).median(0).values
        prefix = base_by_condition[conditions[0]][0]["prefix_embeddings_before"][0]
        raw_embedding_norm = torch.linalg.vector_norm(prefix, dim=-1).cpu()
        embedding_norm = normalize_within(raw_embedding_norm, visual) + normalize_within(
            raw_embedding_norm, language
        )
        rng = np.random.default_rng(lock["random_ranking_seed"] + state_index)
        random_score = torch.zeros(len(span_map))
        random_score[visual] = torch.from_numpy(rng.random(len(visual))).float()
        random_score[language] = torch.from_numpy(rng.random(len(language))).float()
        uniform = rng.choice(visual, size=24, replace=False).tolist()
        top = sorted(visual, key=lambda index: float(aggregate_late[index]), reverse=True)[:8]
        bottom = sorted(visual, key=lambda index: float(aggregate_late[index]))[:8]
        memberships = {index: set() for index in set(uniform + top + bottom)}
        for name, values in [("uniform", uniform), ("top", top), ("bottom", bottom)]:
            for index in values:
                memberships[index].add(name)
        for index in language:
            memberships[index] = {"language_all"}
        group_indices = sorted(memberships)
        expected_rows += len(group_indices) * len(conditions) * 3
        internal_map = None
        modes = ["position", "global", "zero"]
        for tau_value, noise_seed in conditions:
            base, noise, tau = base_by_condition[(tau_value, noise_seed)]
            action_dim = batch["action"].shape[-1]
            valid = (~batch["action_is_pad"]).unsqueeze(-1).expand(-1, -1, action_dim)
            v_pos = base["v_pred"][:, :, :action_dim]
            u_target = base["u_target"][:, :, :action_dim]
            for mode in modes:
                for chunk_start in range(0, len(group_indices), BATCH_SIZE):
                    chunk = group_indices[chunk_start : chunk_start + BATCH_SIZE]
                    replacement_name = {
                        "position": "position_conditioned_modality_mean",
                        "global": "global_modality_mean",
                        "zero": "zero",
                    }[mode]
                    missing = [
                        index for index in chunk
                        if (
                            state_row["episode_id"], state_row["frame_id"], tau_value,
                            noise_seed, f"prefix-{index}", replacement_name,
                        ) not in completed
                    ]
                    if not missing:
                        continue

                    def intervene(prefix_embeddings, maps):
                        nonlocal internal_map
                        internal_map = maps[0]
                        if mode == "zero":
                            replacement = {
                                token.index: torch.zeros_like(prefix_embeddings[0, token.index])
                                for token in maps[0] if token.intervention_allowed
                            }
                        else:
                            replacement = replacements_for(maps[0], means, mode=mode)
                        return grouped_intervention(
                            prefix_embeddings, maps[0], [[index] for index in missing], replacement
                        )

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        negative = predict_teacher_forced_velocity(
                            policy.model,
                            images,
                            image_masks,
                            batch["observation.language.tokens"],
                            batch["observation.language.attention_mask"],
                            state,
                            actions,
                            tau,
                            noise,
                            prefix_intervention=intervene,
                            camera_ids=CAMERA_IDS,
                        )
                    runtime_ms = (time.perf_counter() - started) * 1000 / len(missing)
                    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                    for row_index, index in enumerate(missing):
                        token = span_map[index]
                        v_neg = negative["v_pred"][row_index : row_index + 1, :, :action_dim]
                        metrics = intervention_metrics(
                            v_pos.detach().cpu().numpy(), v_neg.detach().cpu().numpy(),
                            u_target.detach().cpu().numpy(), valid.detach().cpu().numpy(),
                        )
                        flat_pos, flat_neg = v_pos[valid], v_neg[valid]
                        cosine = float(torch.nn.functional.cosine_similarity(flat_pos[None], flat_neg[None]))
                        append(
                            raw_path,
                            {
                                **{key: state_row[key] for key in ["task_id", "suite", "episode_id", "frame_id", "split"]},
                                "state_index": state_index,
                                "tau": tau_value,
                                "noise_seed": noise_seed,
                                "group_id": f"prefix-{index}",
                                "modality": token["modality"],
                                "token_indices": [index],
                                "token_text": token["decoded_language_piece"],
                                "camera_id": token["camera_id"],
                                "visual_token_index": token["visual_token_index"],
                                "selection_sets": sorted(memberships[index]),
                                "deterministic_random": float(random_score[index]),
                                "embedding_norm": float(embedding_norm[index]),
                                "raw_last_expert_layer_attention": float(aggregate_last[index]),
                                "late_half_action_to_context_attention": float(aggregate_late[index]),
                                "replacement_type": replacement_name,
                                "v_pos_norm": float(torch.linalg.vector_norm(flat_pos)),
                                "v_neg_norm": float(torch.linalg.vector_norm(flat_neg)),
                                "base_loss": metrics["base_mse"],
                                "intervention_loss": metrics["base_mse"] + metrics["Q_G"],
                                "velocity_delta_mse": float(((v_pos - v_neg).square())[valid].mean()),
                                "velocity_delta_cosine": cosine,
                                "relative_magnitude_I": metrics["I_G"],
                                "signed_loss_delta_Q": metrics["Q_G"],
                                "Q_rel": metrics["Q_rel"],
                                "valid_action_count": int(valid.sum()),
                                "runtime_ms": runtime_ms,
                                "peak_memory_bytes": int(peak),
                                "integrity_flags": {"finite": bool(torch.isfinite(v_neg).all()), "shared_action_tau_noise": True},
                            },
                        )
        print(f"state {state_index + 1}/120 raw_rows={sum(1 for _ in raw_path.open())}", flush=True)
    summary = {
        "expected_rows_this_run": expected_rows,
        "actual_rows": sum(1 for _ in raw_path.open()),
        "heldout_states": len(heldout),
        "runtime_seconds": time.perf_counter() - started_all,
        "heldout_summary_viewed": False,
    }
    (artifact / "heldout_run_integrity.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

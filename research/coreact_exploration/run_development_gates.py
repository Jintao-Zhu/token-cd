#!/usr/bin/env python3
"""Run CoreAct calibration and mandatory development integrity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from research.coreact_exploration.instrumentation import (
    attention_ranking_scores,
    grouped_intervention,
    predict_teacher_forced_velocity,
)
from research.coreact_exploration.metrics import intervention_metrics, sha256_file
from research.coreact_exploration.qualify_single_suite import (
    CHECKPOINT_REVISION,
    Paths,
    build_action_chunk,
    build_raw_batch,
    load_preprocessor,
    run_preprocessor,
)


FP32_TOLERANCE = 1e-6
BATCH_PARITY_TOLERANCE = 1e-4
CAMERA_IDS = ["camera1", "camera2"]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


class SnapshotRows:
    def __init__(self, dataset: Path):
        self.dataset = dataset
        self.locator: dict[tuple[int, int], tuple[Path, int]] = {}
        self.actions: dict[int, list[float]] = {}
        self.episode_rows: dict[int, dict] = {
            row["episode_index"]: row
            for row in pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        }
        self._cached_path: Path | None = None
        self._cached_rows: list[dict] | None = None
        for path in sorted((dataset / "data/chunk-000").glob("*.parquet")):
            table = pq.read_table(
                path, columns=["episode_index", "frame_index", "index", "action"]
            ).to_pylist()
            for offset, row in enumerate(table):
                self.locator[(row["episode_index"], row["frame_index"])] = (path, offset)
                self.actions[row["index"]] = row["action"]

    def row(self, episode: int, frame: int) -> dict:
        path, offset = self.locator[(episode, frame)]
        if path != self._cached_path:
            self._cached_rows = pq.read_table(path).to_pylist()
            self._cached_path = path
        return self._cached_rows[offset]

    def action_chunk(self, episode: int, frame: int, chunk_size: int = 50):
        meta = self.episode_rows[episode]
        indices, padding, _ = build_action_chunk_from_map(
            self.actions, meta, episode, frame, chunk_size
        )
        return indices, padding


def build_action_chunk_from_map(actions, episode_meta, episode, frame, chunk_size):
    from research.coreact_exploration.qualify_single_suite import official_action_query

    absolute = episode_meta["dataset_from_index"] + frame
    query_indices, padding = official_action_query(episode_meta, absolute, chunk_size)
    chunk = torch.tensor([actions[index] for index in query_indices], dtype=torch.float32)
    return chunk, padding, query_indices


def processed_state(store, preprocessor, manifest_row):
    episode, frame = manifest_row["episode_id"], manifest_row["frame_id"]
    raw_row = store.row(episode, frame)
    actions, padding, _ = build_action_chunk_from_map(
        store.actions, store.episode_rows[episode], episode, frame, 50
    )
    raw = build_raw_batch(raw_row, manifest_row["instruction"], actions, padding)
    return run_preprocessor(preprocessor, raw)


def combine_processed(items: list[dict]) -> dict:
    keys = [
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
        "action",
        "action_is_pad",
    ]
    return {key: torch.cat([item[key] for item in items], dim=0) for key in keys}


def load_policy(paths: Paths):
    config = PreTrainedConfig.from_pretrained(paths.checkpoint, local_files_only=True)
    policy = SmolVLAPolicy.from_pretrained(
        paths.checkpoint, config=config, local_files_only=True, strict=True
    )
    policy.eval().requires_grad_(False)
    policy.to(dtype=torch.float32)
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("policy freeze/eval gate failed")
    return policy


def calibration_means(policy, preprocessor, store, rows, output: Path, batch_size=8):
    visual_sums = None
    visual_counts = None
    language_sum = None
    language_count = 0
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    special_ids = set(tokenizer.all_special_ids)
    for start in range(0, len(rows), batch_size):
        processed = combine_processed(
            [processed_state(store, preprocessor, row) for row in rows[start : start + batch_size]]
        )
        images, masks = policy.prepare_images(processed)
        state = policy.prepare_state(processed)
        tokens = processed["observation.language.tokens"]
        token_masks = processed["observation.language.attention_mask"]
        with torch.inference_mode():
            prefix, _, _ = policy.model.embed_prefix(images, masks, tokens, token_masks, state=state)
        if visual_sums is None:
            with torch.inference_mode():
                visual_token_count = policy.model.vlm_with_expert.embed_image(images[0][:1]).shape[1]
            visual_sums = torch.zeros(
                len(images), visual_token_count, prefix.shape[-1], dtype=torch.float64
            )
            visual_counts = torch.zeros(len(images), visual_token_count, dtype=torch.int64)
            language_sum = torch.zeros(prefix.shape[-1], dtype=torch.float64)
        offset = 0
        for camera in range(len(images)):
            if policy.model.add_image_special_tokens:
                offset += 2
            values = prefix[:, offset : offset + visual_sums.shape[1]].detach().double().cpu()
            valid = masks[camera].detach().cpu().bool()
            visual_sums[camera] += values[valid].sum(dim=0)
            visual_counts[camera] += int(valid.sum())
            offset += visual_sums.shape[1]
            if policy.model.add_image_special_tokens:
                offset += 1
        language_values = prefix[:, offset : offset + tokens.shape[1]].detach().double().cpu()
        valid_language = token_masks.detach().cpu().bool()
        for special in special_ids:
            valid_language &= tokens.detach().cpu() != special
        language_sum += language_values[valid_language].sum(dim=0)
        language_count += int(valid_language.sum())

    means = {
        "visual_position_mean": (visual_sums / visual_counts[:, :, None]).float(),
        "visual_global_mean": (visual_sums.sum(dim=(0, 1)) / visual_counts.sum()).float(),
        "language_mean": (language_sum / language_count).float(),
        "visual_counts": visual_counts,
        "language_count": language_count,
        "calibration_state_count": len(rows),
        "camera_ids": CAMERA_IDS,
    }
    torch.save(means, output / "modality_means.pt")
    write_json(
        output / "modality_means_metadata.json",
        {
            "calibration_state_count": len(rows),
            "camera_ids": CAMERA_IDS,
            "visual_position_mean_shape": list(means["visual_position_mean"].shape),
            "visual_counts_shape": list(visual_counts.shape),
            "visual_count_min": int(visual_counts.min()),
            "visual_count_max": int(visual_counts.max()),
            "language_mean_shape": list(means["language_mean"].shape),
            "language_token_count": language_count,
            "stored_dtype": str(means["visual_position_mean"].dtype),
            "sha256": sha256_file(output / "modality_means.pt"),
            "source_split": "mean_calibration only",
            "actions_or_effects_used": False,
        },
    )
    return means


def replacements_for(span_map, means, mode="position"):
    replacements = {}
    for token in span_map:
        if not token.intervention_allowed:
            continue
        if token.modality == "visual":
            camera = CAMERA_IDS.index(token.camera_id)
            replacements[token.index] = (
                means["visual_position_mean"][camera, token.visual_token_index]
                if mode == "position"
                else means["visual_global_mean"]
            )
        elif token.modality == "language":
            replacements[token.index] = means["language_mean"]
    return replacements


def valid_arrays(batch, value):
    action_dim = batch["action"].shape[-1]
    mask = (~batch["action_is_pad"]).unsqueeze(-1).expand(-1, -1, action_dim)
    return value[:, :, :action_dim], mask


def max_trace_diff(left, right):
    if len(left) != len(right):
        return float("inf")
    return max(
        float((a["probabilities"] - b["probabilities"]).abs().max())
        for a, b in zip(left, right, strict=True)
    )


def run_development(policy, preprocessor, store, rows, means, output: Path, protocol):
    records_path = output / "development_integrity_rows.jsonl"
    effects_path = output / "development_effects.jsonl"
    if records_path.exists() or effects_path.exists():
        raise FileExistsError("development outputs are append-only and already exist")
    failures = []
    no_op_values = []
    effect_values = defaultdict(list)
    prefix_examples = []
    diagnostic = None
    for state_index, manifest_row in enumerate(rows):
        started = time.perf_counter()
        batch = processed_state(store, preprocessor, manifest_row)
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        actions = policy.prepare_action(batch)
        generator = torch.Generator(device=actions.device).manual_seed(protocol["noise_seeds"][0])
        noise = torch.randn(
            actions.shape, generator=generator, device=actions.device, dtype=actions.dtype
        )
        tau = torch.tensor([0.5], device=actions.device, dtype=actions.dtype)
        probe_args = (
            policy.model,
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            state,
            actions,
            tau,
            noise,
        )
        with torch.inference_mode():
            base = predict_teacher_forced_velocity(
                *probe_args, record_attention=True, camera_ids=CAMERA_IDS
            )
            repeat = predict_teacher_forced_velocity(
                *probe_args, record_attention=True, camera_ids=CAMERA_IDS
            )
            policy_loss, _ = policy(batch, noise=noise, time=tau, reduction="none")
        span_map = base["prefix_span_map"][0]
        if state_index < 3:
            prefix_examples.append(
                {
                    "state": manifest_row,
                    "prefix_shape": base["prefix_summary"]["before_shape"],
                    "tokens": span_map,
                }
            )
        eligible_visual = [
            item["index"] for item in span_map if item["intervention_allowed"] and item["modality"] == "visual"
        ]
        eligible_language = [
            item["index"] for item in span_map if item["intervention_allowed"] and item["modality"] == "language"
        ]
        internal_map = []
        # The intervention callback receives dataclasses; retain it for validation and replacements.
        def identity_intervention(prefix, maps):
            nonlocal internal_map
            internal_map = maps[0]
            selected = eligible_visual[0]
            return grouped_intervention(prefix, maps[0], [[selected]], {selected: prefix[0, selected]})

        with torch.inference_mode():
            identity = predict_teacher_forced_velocity(
                *probe_args,
                prefix_intervention=identity_intervention,
                record_attention=True,
                camera_ids=CAMERA_IDS,
            )
            empty = predict_teacher_forced_velocity(
                *probe_args,
                prefix_intervention=lambda prefix, maps: grouped_intervention(
                    prefix, maps[0], [[]], {}
                ),
                record_attention=True,
                camera_ids=CAMERA_IDS,
            )
        velocity_repeat_diff = float((base["v_pred"] - repeat["v_pred"]).abs().max())
        attention_repeat_diff = max_trace_diff(base["attention_trace"], repeat["attention_trace"])
        no_op_velocity_diff = max(
            float((identity["v_pred"] - base["v_pred"]).abs().max()),
            float((empty["v_pred"] - base["v_pred"]).abs().max()),
        )
        no_op_attention_diff = max(
            max_trace_diff(identity["attention_trace"], base["attention_trace"]),
            max_trace_diff(empty["attention_trace"], base["attention_trace"]),
        )

        v_valid, valid_mask = valid_arrays(batch, base["v_pred"])
        u_valid, _ = valid_arrays(batch, base["u_target"])
        manual_loss = ((v_valid - u_valid).square()[valid_mask]).mean()
        forward_diff = float((manual_loss - policy_loss[0]).abs())
        for no_op_result in (identity, empty):
            no_op_valid, _ = valid_arrays(batch, no_op_result["v_pred"])
            no_op_values.append(
                intervention_metrics(
                    v_valid.detach().cpu().numpy(),
                    no_op_valid.detach().cpu().numpy(),
                    u_valid.detach().cpu().numpy(),
                    valid_mask.detach().cpu().numpy(),
                )["I_G"]
            )
        trace_row_sum_diff = max(
            float((trace["row_probability_sums"] - 1).abs().max())
            for trace in base["attention_trace"]
        )
        trace_masked_max = 0.0
        for trace in base["attention_trace"]:
            expanded_mask = trace["key_mask"][:, None].expand_as(trace["probabilities"])
            if (~expanded_mask).any():
                trace_masked_max = max(
                    trace_masked_max,
                    float(trace["probabilities"][~expanded_mask].abs().max()),
                )

        replacement_map = replacements_for(internal_map, means)
        parity_groups = [[eligible_visual[0]], [eligible_language[0]]]

        def batch_intervention(prefix, maps):
            return grouped_intervention(prefix, maps[0], parity_groups, replacement_map)

        with torch.inference_mode():
            batched = predict_teacher_forced_velocity(
                *probe_args, prefix_intervention=batch_intervention, camera_ids=CAMERA_IDS
            )
            serial = []
            for group in parity_groups:
                serial_out = predict_teacher_forced_velocity(
                    *probe_args,
                    prefix_intervention=lambda prefix, maps, group=group: grouped_intervention(
                        prefix, maps[0], [group], replacement_map
                    ),
                    camera_ids=CAMERA_IDS,
                )
                serial.append(serial_out["v_pred"][0])
        batch_diff = float((batched["v_pred"] - torch.stack(serial)).abs().max())

        rng = np.random.default_rng(protocol["random_ranking_seed"] + state_index)
        sampled_visual = rng.choice(eligible_visual, size=24, replace=False).tolist()
        effect_groups = [[index] for index in sampled_visual + eligible_language]

        def effect_intervention(prefix, maps):
            return grouped_intervention(prefix, maps[0], effect_groups, replacement_map)

        with torch.inference_mode():
            effects = predict_teacher_forced_velocity(
                *probe_args, prefix_intervention=effect_intervention, camera_ids=CAMERA_IDS
            )
        ranking = attention_ranking_scores(base["attention_trace"], len(span_map))
        prefix_norm = torch.linalg.vector_norm(base["prefix_embeddings_before"][0], dim=-1).cpu()
        for effect_index, group in enumerate(effect_groups):
            token = span_map[group[0]]
            vn_valid, _ = valid_arrays(batch, effects["v_pred"][effect_index : effect_index + 1])
            metrics = intervention_metrics(
                v_valid.detach().cpu().numpy(),
                vn_valid.detach().cpu().numpy(),
                u_valid.detach().cpu().numpy(),
                valid_mask.detach().cpu().numpy(),
            )
            effect_values[token["modality"]].append(metrics["I_G"])
            append_jsonl(
                effects_path,
                {
                    **{key: manifest_row[key] for key in ["task_id", "episode_id", "frame_id"]},
                    "state_index": state_index,
                    "group_id": f"prefix-{group[0]}",
                    "modality": token["modality"],
                    "camera_id": token["camera_id"],
                    "token_index": group[0],
                    "token_text": token["decoded_language_piece"],
                    "replacement_type": "position_conditioned_modality_mean",
                    "embedding_norm": float(prefix_norm[group[0]]),
                    "raw_last_expert_layer_attention": float(
                        ranking["raw_last_expert_layer_attention"][group[0]]
                    ),
                    "late_half_action_to_context_attention": float(
                        ranking["late_half_action_to_context_attention"][group[0]]
                    ),
                    **metrics,
                },
            )

        finite_trace = all(torch.isfinite(trace["probabilities"]).all() for trace in base["attention_trace"])
        gate_values = {
            "determinism_velocity_max_abs_diff": velocity_repeat_diff,
            "determinism_attention_max_abs_diff": attention_repeat_diff,
            "forward_parity_abs_diff": forward_diff,
            "no_op_velocity_max_abs_diff": no_op_velocity_diff,
            "no_op_attention_max_abs_diff": no_op_attention_diff,
            "batch_serial_velocity_max_abs_diff": batch_diff,
            "attention_row_sum_max_abs_diff": trace_row_sum_diff,
            "attention_masked_probability_max": trace_masked_max,
            "attention_finite": bool(finite_trace),
            "prefix_map_valid": bool(eligible_visual and eligible_language),
        }
        strict_values = [
            velocity_repeat_diff,
            attention_repeat_diff,
            forward_diff,
            no_op_velocity_diff,
            no_op_attention_diff,
            trace_row_sum_diff,
            trace_masked_max,
        ]
        row_pass = (
            all(value <= FP32_TOLERANCE for value in strict_values)
            and batch_diff <= BATCH_PARITY_TOLERANCE
            and gate_values["attention_finite"]
            and gate_values["prefix_map_valid"]
        )
        if not row_pass:
            failures.append({"state_index": state_index, "values": gate_values})
        append_jsonl(
            records_path,
            {
                **manifest_row,
                **gate_values,
                "valid_action_count": int(valid_mask.sum()),
                "runtime_seconds": time.perf_counter() - started,
                "pass": row_pass,
            },
        )

        if state_index == 0:
            zero_map = {
                token.index: torch.zeros_like(base["prefix_embeddings_before"][0, token.index])
                for token in internal_map
                if token.intervention_allowed
            }
            diagnostic_groups = [eligible_language, [
                item["index"]
                for item in span_map
                if item["intervention_allowed"] and item["camera_id"] == CAMERA_IDS[0]
            ]]
            with torch.inference_mode():
                diag = predict_teacher_forced_velocity(
                    *probe_args,
                    prefix_intervention=lambda prefix, maps: grouped_intervention(
                        prefix, maps[0], diagnostic_groups, zero_map
                    ),
                    camera_ids=CAMERA_IDS,
                )
            diagnostic = {
                "all_language_zero_velocity_mse": float(
                    (diag["v_pred"][0] - base["v_pred"][0]).square().mean()
                ),
                "camera1_zero_velocity_mse": float(
                    (diag["v_pred"][1] - base["v_pred"][0]).square().mean()
                ),
                "interpretation": "diagnostic only; zero is OOD and is not primary evidence",
            }
        if failures:
            break

    write_json(output / "prefix_map_examples.json", prefix_examples)
    no_op_threshold = float(np.quantile(no_op_values, 0.999))
    thresholds = {
        modality: max(no_op_threshold, float(np.quantile(values, 0.75)))
        for modality, values in effect_values.items()
    }
    return {
        "pass": not failures and len(rows) == 24,
        "processed_development_states": state_index + 1,
        "expected_development_states": 24,
        "tolerance": FP32_TOLERANCE,
        "batch_parity_tolerance": BATCH_PARITY_TOLERANCE,
        "failures": failures,
        "diagnostic_positive_controls": diagnostic,
        "no_op_99_9_percentile": no_op_threshold,
        "development_effect_thresholds": thresholds,
        "effect_counts": {key: len(value) for key, value in effect_values.items()},
    }


def git_output(workspace: Path, *args):
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={workspace / 'lerobot'}",
            "-C",
            str(workspace / "lerobot"),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.artifact.resolve()
    protocol = yaml.safe_load((output / "protocol.yaml").read_text())
    tolerances = protocol["integrity_tolerances"]
    if tolerances["determinism_forward_noop"] != FP32_TOLERANCE:
        raise ValueError("protocol/code strict tolerance mismatch")
    if tolerances["batch_serial_velocity"] != BATCH_PARITY_TOLERANCE:
        raise ValueError("protocol/code batch tolerance mismatch")
    manifest = [json.loads(line) for line in (output / "sample_manifest.jsonl").read_text().splitlines()]
    calibration = [row for row in manifest if row["split"] == "mean_calibration"]
    development = [row for row in manifest if row["split"] == "development"]
    dataset = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero"
    paths = Paths(
        workspace,
        dataset,
        workspace
        / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots"
        / CHECKPOINT_REVISION,
        workspace
        / "counterfactual-flow-vla/pi05_svcpd_p0/verified_calibration_download_manifest.json",
    )
    started = time.perf_counter()
    store = SnapshotRows(dataset)
    preprocessor = load_preprocessor(paths)
    policy = load_policy(paths)
    with torch.inference_mode():
        means = calibration_means(policy, preprocessor, store, calibration, output)
        report = run_development(
            policy, preprocessor, store, development, means, output, protocol
        )
    write_json(output / "integrity_report.json", report)
    status = "DEVELOPMENT_INTEGRITY_PASS" if report["pass"] else "FAILED_INTEGRITY"
    write_json(
        output / "decision.json",
        {
            "status": status,
            "heldout_viewed": False,
            "heldout_created": False,
            "single_suite_only": True,
            "cross_suite_proceed_eligible": False,
        },
    )
    write_json(
        output / "environment.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "precision": "fp32",
            "policy_eval": not policy.training,
            "all_parameters_frozen": not any(p.requires_grad for p in policy.parameters()),
            "lerobot_commit": git_output(workspace, "rev-parse", "HEAD"),
            "lerobot_dirty_status": git_output(workspace, "status", "--short").splitlines(),
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps({"artifact": str(output), "decision": status}, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

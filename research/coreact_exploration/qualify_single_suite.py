#!/usr/bin/env python3
"""Qualify the fixed LIBERO-Object snapshot for CoreAct v3."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline


DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
CHECKPOINT_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
PARITY_TOLERANCE = 1e-3
EPISODE_INDEX = 807
MODEL_FRAME_INDEX = 70
TAU = 0.5
NOISE_SEED = 1729


@dataclass(frozen=True)
class Paths:
    workspace: Path
    dataset: Path
    checkpoint: Path
    download_manifest: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(items: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(items):
        digest.update(key.encode("utf-8"))
        value = items[key]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def decode_image(value: dict[str, Any]) -> torch.Tensor:
    image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    array = np.array(image, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)


def audit_source_shards(paths: Paths) -> dict[str, Any]:
    manifest = json.loads(paths.download_manifest.read_text())
    records = manifest["records"]
    results = []
    for record in records:
        path = paths.dataset / "data" / "chunk-000" / f"file-{record['index']:03d}.parquet"
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        parquet = pq.ParquetFile(path)
        results.append(
            {
                "index": record["index"],
                "path": str(path.relative_to(paths.workspace)),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "size_matches": actual_bytes == record["bytes"],
                "sha256_matches": actual_sha256 == record["sha256"],
            }
        )
    return {
        "repo": "HuggingFaceVLA/libero",
        "evaluation_snapshot_revision": DATASET_REVISION,
        "claimed_as_training_revision": False,
        "shard_count": len(results),
        "total_bytes": sum(row["bytes"] for row in results),
        "total_rows": sum(row["rows"] for row in results),
        "all_size_matches": all(row["size_matches"] for row in results),
        "all_sha256_matches": all(row["sha256_matches"] for row in results),
        "shards": results,
    }


def load_episode(paths: Paths) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    episode_table = pq.read_table(paths.dataset / "meta/episodes/chunk-000/file-000.parquet")
    episode_meta = next(row for row in episode_table.to_pylist() if row["episode_index"] == EPISODE_INDEX)

    columns = [
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    rows = []
    for path in sorted((paths.dataset / "data/chunk-000").glob("*.parquet")):
        metadata = pq.read_metadata(path)
        # Min/max statistics avoid decoding image columns in unrelated shards.
        schema_names = metadata.schema.names
        ep_column = schema_names.index("episode_index")
        stats = metadata.row_group(0).column(ep_column).statistics
        if stats.min <= EPISODE_INDEX <= stats.max:
            table = pq.read_table(path, columns=columns)
            rows.extend(row for row in table.to_pylist() if row["episode_index"] == EPISODE_INDEX)

    rows.sort(key=lambda row: row["frame_index"])
    task_rows = pq.read_table(paths.dataset / "meta/tasks.parquet").to_pylist()
    task_map = {row["task_index"]: row["__index_level_0__"] for row in task_rows}
    instruction = task_map[rows[0]["task_index"]]
    return rows, episode_meta, instruction


def verify_episode(rows: list[dict[str, Any]], episode_meta: dict[str, Any], instruction: str) -> dict[str, Any]:
    frame_indices = [row["frame_index"] for row in rows]
    global_indices = [row["index"] for row in rows]
    return {
        "episode_index": EPISODE_INDEX,
        "row_count": len(rows),
        "metadata_length": episode_meta["length"],
        "frame_indices_contiguous": frame_indices == list(range(len(rows))),
        "global_indices_contiguous": global_indices
        == list(range(episode_meta["dataset_from_index"], episode_meta["dataset_to_index"])),
        "canonical_instruction": instruction,
        "metadata_instruction": episode_meta["tasks"][0],
        "instruction_matches_episode_metadata": instruction == episode_meta["tasks"][0],
        "state_dim": len(rows[0]["observation.state"]),
        "action_dim": len(rows[0]["action"]),
        "real_camera_keys": ["observation.images.image", "observation.images.image2"],
    }


def official_action_query(
    episode_meta: dict[str, Any], absolute_index: int, chunk_size: int
) -> tuple[list[int], torch.Tensor]:
    reader = object.__new__(DatasetReader)
    reader._meta = SimpleNamespace(episodes={EPISODE_INDEX: episode_meta})
    reader.delta_indices = {"action": list(range(chunk_size))}
    indices, padding = DatasetReader._get_query_indices(reader, absolute_index, EPISODE_INDEX)
    return indices["action"], padding["action_is_pad"]


def build_action_chunk(
    rows: list[dict[str, Any]], episode_meta: dict[str, Any], frame_index: int, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    absolute_index = episode_meta["dataset_from_index"] + frame_index
    query_indices, is_pad = official_action_query(episode_meta, absolute_index, chunk_size)
    by_absolute_index = {row["index"]: row for row in rows}
    actions = torch.tensor([by_absolute_index[index]["action"] for index in query_indices], dtype=torch.float32)
    return actions, is_pad, query_indices


def load_preprocessor(paths: Paths) -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline.from_pretrained(
        paths.checkpoint,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
    )


def build_raw_batch(
    row: dict[str, Any], instruction: str, actions: torch.Tensor, action_is_pad: torch.Tensor
) -> dict[str, Any]:
    return {
        "observation.images.image": decode_image(row["observation.images.image"]),
        "observation.images.image2": decode_image(row["observation.images.image2"]),
        "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
        "action": actions,
        "action_is_pad": action_is_pad,
        "task": instruction,
    }


def run_preprocessor(preprocessor: PolicyProcessorPipeline, raw: dict[str, Any]) -> dict[str, Any]:
    processed = preprocessor(raw)
    processed["action"] = processed["action"].unsqueeze(0)
    processed["action_is_pad"] = processed["action_is_pad"].unsqueeze(0)
    return processed


def describe_batch(batch: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in sorted(batch.items()):
        if isinstance(value, torch.Tensor):
            output[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "finite": bool(torch.isfinite(value).all()) if value.is_floating_point() else None,
            }
        else:
            output[key] = {"type": type(value).__name__, "value": value}
    return output


def capture_velocity(
    policy: SmolVLAPolicy,
    images: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    batch: dict[str, Any],
    noise: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach().float().cpu())

    handle = policy.model.action_out_proj.register_forward_hook(hook)
    try:
        losses = policy.model.forward(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            policy.prepare_state(batch),
            policy.prepare_action(batch),
            noise=noise,
            time=tau,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one action_out_proj call, got {len(captured)}")
    return captured[0], losses.detach().float().cpu()


def velocity_parity(paths: Paths, batch: dict[str, Any]) -> dict[str, Any]:
    config = PreTrainedConfig.from_pretrained(paths.checkpoint, local_files_only=True)
    policy = SmolVLAPolicy.from_pretrained(
        paths.checkpoint, config=config, local_files_only=True, strict=True
    )
    policy.eval()
    policy.requires_grad_(False)
    # The preregistered mechanism fallback is fp32 when bf16 parity exceeds 1e-3.
    policy.to(dtype=torch.float32)
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("Policy freeze/eval gate failed")

    images, image_masks = policy.prepare_images(batch)
    actions = policy.prepare_action(batch)
    generator = torch.Generator(device=actions.device).manual_seed(NOISE_SEED)
    noise = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
    tau = torch.tensor([TAU], device=actions.device, dtype=actions.dtype)

    with torch.inference_mode():
        native_velocity, native_loss = capture_velocity(
            policy, images, image_masks, batch, noise, tau
        )
        repeated_native_velocity, repeated_native_loss = capture_velocity(
            policy, images, image_masks, batch, noise, tau
        )
        empty_image = torch.full_like(images[0], -1.0)
        empty_mask = torch.zeros_like(image_masks[0], dtype=torch.bool)
        masked_velocity, masked_loss = capture_velocity(
            policy,
            [*images, empty_image],
            [*image_masks, empty_mask],
            batch,
            noise,
            tau,
        )

    valid = (~batch["action_is_pad"].detach().cpu()).unsqueeze(-1).expand(-1, -1, 7)
    velocity_diff = (native_velocity[:, :, :7] - masked_velocity[:, :, :7]).abs()
    loss_diff = (native_loss[:, :, :7] - masked_loss[:, :, :7]).abs()
    repeat_velocity_diff = (
        native_velocity[:, :, :7] - repeated_native_velocity[:, :, :7]
    ).abs()
    repeat_loss_diff = (native_loss[:, :, :7] - repeated_native_loss[:, :, :7]).abs()
    max_velocity_diff = float(velocity_diff[valid].max())
    max_loss_diff = float(loss_diff[valid].max())
    return {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "policy_eval": not policy.training,
        "all_parameters_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
        "native_real_view_count": len(images),
        "explicit_view_count": len(images) + 1,
        "explicit_empty_mask_all_false": not bool(empty_mask.any()),
        "fixed_tau": TAU,
        "fixed_noise_seed": NOISE_SEED,
        "same_actions_tau_noise": True,
        "compute_precision": "fp32",
        "comparison_unit": "valid action elements (7 original dimensions, non-padded timesteps)",
        "native_repeat_max_abs_velocity_diff": float(repeat_velocity_diff[valid].max()),
        "native_repeat_max_abs_loss_diff": float(repeat_loss_diff[valid].max()),
        "max_abs_velocity_diff": max_velocity_diff,
        "max_abs_loss_diff": max_loss_diff,
        "tolerance": PARITY_TOLERANCE,
        "pass": all(
            value <= PARITY_TOLERANCE
            for value in [
                max_velocity_diff,
                max_loss_diff,
                float(repeat_velocity_diff[valid].max()),
                float(repeat_loss_diff[valid].max()),
            ]
        ),
        "native_velocity_norm": float(torch.linalg.vector_norm(native_velocity)),
        "masked_velocity_norm": float(torch.linalg.vector_norm(masked_velocity)),
    }


def git_output(workspace: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={workspace / 'lerobot'}",
        "-C",
        str(workspace / "lerobot"),
        *args,
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = args.output_root or workspace / "artifacts" / f"coreact_exploration_v3_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    paths = Paths(
        workspace=workspace,
        dataset=workspace
        / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero",
        checkpoint=workspace
        / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots"
        / CHECKPOINT_REVISION,
        download_manifest=workspace
        / "counterfactual-flow-vla/pi05_svcpd_p0/verified_calibration_download_manifest.json",
    )

    started = time.perf_counter()
    shard_audit = audit_source_shards(paths)
    write_json(output / "source_shard_audit.json", shard_audit)

    rows, episode_meta, instruction = load_episode(paths)
    episode_check = verify_episode(rows, episode_meta, instruction)
    write_json(output / "episode_qualification.json", episode_check)

    end_actions, end_padding, end_indices = build_action_chunk(
        rows, episode_meta, len(rows) - 1, chunk_size=50
    )
    padding_gate = {
        "official_helper": "lerobot.datasets.dataset_reader.DatasetReader._get_query_indices",
        "episode_index": EPISODE_INDEX,
        "frame_index": len(rows) - 1,
        "chunk_size": 50,
        "query_indices": end_indices,
        "padding_mask": end_padding.tolist(),
        "valid_action_count": int((~end_padding).sum()),
        "padded_action_count": int(end_padding.sum()),
        "first_step_valid": not bool(end_padding[0]),
        "remaining_steps_padded": bool(end_padding[1:].all()),
        "padded_actions_repeat_terminal_action": bool(
            torch.equal(end_actions[1:], end_actions[0].expand_as(end_actions[1:]))
        ),
    }
    padding_gate["pass"] = all(
        [
            padding_gate["first_step_valid"],
            padding_gate["remaining_steps_padded"],
            padding_gate["padded_actions_repeat_terminal_action"],
        ]
    )
    write_json(output / "action_padding_gate.json", padding_gate)

    actions, action_is_pad, query_indices = build_action_chunk(
        rows, episode_meta, MODEL_FRAME_INDEX, chunk_size=50
    )
    raw = build_raw_batch(rows[MODEL_FRAME_INDEX], instruction, actions, action_is_pad)
    preprocessor = load_preprocessor(paths)
    processed = run_preprocessor(preprocessor, raw)
    padded_state = torch.nn.functional.pad(processed["observation.state"], (0, 24))
    preprocessor_gate = {
        "input": describe_batch(raw),
        "output": describe_batch(processed),
        "canonical_instruction": instruction,
        "language_nonpadding_tokens": int(processed["observation.language.attention_mask"].sum()),
        "query_indices": query_indices,
        "raw_batch_sha256": tensor_sha256(raw),
        "processed_batch_sha256": tensor_sha256(processed),
        "state_runtime_input_dim": int(raw["observation.state"].shape[-1]),
        "state_normalized_dim": int(processed["observation.state"].shape[-1]),
        "state_model_padded_dim": int(padded_state.shape[-1]),
        "two_real_cameras": all(
            key in processed
            for key in ["observation.images.camera1", "observation.images.camera2"]
        ),
    }
    preprocessor_gate["pass"] = all(
        [
            preprocessor_gate["state_runtime_input_dim"] == 8,
            preprocessor_gate["state_normalized_dim"] == 8,
            preprocessor_gate["state_model_padded_dim"] == 32,
            preprocessor_gate["two_real_cameras"],
            preprocessor_gate["language_nonpadding_tokens"] > 0,
        ]
    )
    write_json(output / "preprocessor_gate.json", preprocessor_gate)

    parity_gate = velocity_parity(paths, processed)
    write_json(output / "masked_view_velocity_parity.json", parity_gate)

    gates = {
        "source_shards": shard_audit["all_size_matches"] and shard_audit["all_sha256_matches"],
        "episode_integrity": all(
            [
                episode_check["row_count"] == episode_check["metadata_length"],
                episode_check["frame_indices_contiguous"],
                episode_check["global_indices_contiguous"],
                episode_check["instruction_matches_episode_metadata"],
            ]
        ),
        "action_padding": padding_gate["pass"],
        "published_preprocessor": preprocessor_gate["pass"],
        "masked_empty_view_velocity_parity": parity_gate["pass"],
    }
    all_pass = all(gates.values())
    decision = {
        "status": "QUALIFIED_DATA_READY" if all_pass else "INCONCLUSIVE_DATA_OR_RESOURCES",
        "qualification_scope": "single-suite LIBERO-Object only",
        "gates": gates,
        "all_gates_pass": all_pass,
        "heldout_viewed": False,
        "development_started": False,
        "cross_suite_proceed_eligible": False,
        "reason": "The fixed snapshot currently provides qualification coverage for one suite only.",
    }
    write_json(output / "decision.json", decision)
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "lerobot_commit": git_output(workspace, "rev-parse", "HEAD"),
        "lerobot_dirty_status": git_output(workspace, "status", "--short").splitlines(),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "evaluation_snapshot_revision": DATASET_REVISION,
        "runtime_seconds": time.perf_counter() - started,
        "exact_command": " ".join(subprocess.list2cmdline([part]) for part in __import__("sys").argv),
    }
    write_json(output / "environment.json", environment)
    print(output)
    print(json.dumps(decision, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

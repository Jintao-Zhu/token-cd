"""Canonical four-arm SIMPLER Geometry-Mask/Protect selector experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import subprocess
import time
from pathlib import Path


TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ARMS = ("l11_matched", "geometry_mask", "geometry_protect", "random_matched")
PROTOCOL = "SIMPLER_L11_GEOMETRY_MASK_PROTECT_RANDOM_V1"
SEEDS = tuple(range(100, 200))
LAMBDA = 0.5
RANDOM_SALT = 0x50A77E11


def parse_seeds(spec: str) -> list[int]:
    result: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(part))
    values = sorted(set(result))
    if not values or any(seed not in SEEDS for seed in values):
        raise ValueError("the locked SIMPLER comparison uses seeds 100..199")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _write_arm_arrays(path: Path, records: list[dict], actions) -> None:
    import numpy as np

    if not records:
        raise RuntimeError("cannot write empty arm arrays")
    required = ("positive", "negative", "selected_mask", "prompt_attention", "geometry_scores")
    missing = [key for key in required if any(key not in record for record in records)]
    if missing:
        raise RuntimeError(f"missing per-step selector arrays: {missing}")
    np.savez_compressed(
        path,
        positive_logits=np.stack([record["positive"] for record in records]),
        negative_logits=np.stack([record["negative"] for record in records]),
        selected_mask=np.stack([record["selected_mask"] for record in records]),
        l11_scores=np.stack([record["prompt_attention"] for record in records]),
        geometry_scores=np.stack([record["geometry_scores"] for record in records]),
        executed_actions=actions,
    )


def _arm_policy(base, task: str, mask_selector: str):
    from research.semantic_token_cd.prompt_attn_l11_count_rollout import LAMBDA as SHR_LAMBDA
    from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
    from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX, _init_common

    policy = copy.copy(base)
    policy.__class__ = PromptAttentionSHRInference
    _init_common(policy, SHR_LAMBDA)
    policy.beta = 0.0
    policy.selector_mode = "prompt_attention"
    policy.mask_selector = "l11" if mask_selector == "l11_matched" else mask_selector
    policy.task_index = TASK_INDEX[task]
    policy.attention_layers = (11,)
    policy.selection_count = None
    policy.selection_top_p = None
    policy.selection_budget_schedule = None
    policy.selection_budget_source = "matched"
    policy.save_prompt_attention = True
    policy.geometry_diagnostics = True
    return policy


def _load_existing_arm(artifact: Path, task: str, arm: str, seed: int):
    import numpy as np

    arm_root = artifact / "episodes" / task / arm
    summary_path = arm_root / f"episode_{seed:03d}_summary.json"
    arrays_path = arm_root / f"episode_{seed:03d}_arrays.npz"
    if not (summary_path.is_file() and arrays_path.is_file()):
        return None
    return json.loads(summary_path.read_text()), np.load(arrays_path)


def _save_preflight_figure(path: Path, rgb224, arms_data: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("SIMPLER seed 100: processor-aligned geometry and first-step masks")
    axes[0, 0].imshow(rgb224)
    axes[0, 0].set_title("224x224 OpenVLA RGB")
    geo = arms_data["l11_matched"]["geometry_scores"][0].reshape(16, 16)
    axes[0, 1].imshow(rgb224)
    axes[0, 1].imshow(geo, cmap="magma", alpha=0.55, interpolation="nearest",
                      extent=(0, 224, 224, 0))
    axes[0, 1].set_title("Sobel mean per 14x14 patch")
    for column, (arm, title) in enumerate((
        ("l11_matched", "A: L11 Top-m"),
        ("geometry_mask", "B: Geometry Top-m"),
        ("geometry_protect", "C: Geometry-Protect"),
        ("random_matched", "D: Random-m"),
    )):
        mask = arms_data[arm]["selected_mask"][0].reshape(16, 16)
        axes[1, column].imshow(rgb224)
        axes[1, column].imshow(mask, cmap="winter", alpha=0.65, interpolation="nearest",
                               extent=(0, 224, 224, 0))
        axes[1, column].set_title(title)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_pair_arrays(path: Path, arms_data: dict[str, dict], budget: list[int], rgb224=None) -> dict:
    import numpy as np

    lengths = {arm: len(data["selected_mask"]) for arm, data in arms_data.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"arms have different closed-loop lengths: {lengths}")
    steps = next(iter(lengths.values()))
    if len(budget) != steps:
        raise RuntimeError(f"matched budget tape length {len(budget)} != episode steps {steps}")

    masks = np.stack([arms_data[arm]["selected_mask"] for arm in ARMS]).astype(np.uint8)
    l11 = np.stack([arms_data[arm]["l11_scores"] for arm in ARMS]).astype(np.float32)
    geometry = np.stack([arms_data[arm]["geometry_scores"] for arm in ARMS]).astype(np.float32)
    m_by_arm = np.stack([arms_data[arm]["m_t"] for arm in ARMS]).astype(np.int16)
    if not np.all(m_by_arm == np.asarray(budget, dtype=np.int16)[None, :]):
        raise RuntimeError("m_t is not identical across all four arms at every step")
    expected_counts = np.asarray(budget, dtype=np.int16)
    if not np.array_equal(masks.sum(axis=-1).astype(np.int16), np.broadcast_to(expected_counts, (4, steps))):
        raise RuntimeError("at least one selected mask differs from shared matched m_t")
    jaccard = np.zeros((4, steps), dtype=np.float32)
    for arm_index in range(4):
        for step in range(steps):
            a = masks[0, step].astype(bool)
            b = masks[arm_index, step].astype(bool)
            jaccard[arm_index, step] = float(np.logical_and(a, b).sum()) / max(
                1, int(np.logical_or(a, b).sum())
            )
    metrics = {
        key: np.stack([arms_data[arm][key] for arm in ARMS]).astype(np.float32)
        for key in ("D_feat", "D_res", "D_action")
    }
    payload = {
        "arm_names": np.asarray(ARMS),
        "matched_budget_m": expected_counts,
        "selected_masks": masks,
        "l11_scores": l11,
        "geometry_scores": geometry,
        "jaccard_vs_l11": jaccard,
        **metrics,
    }
    if rgb224 is not None:
        payload["processor_rgb_224_first_step"] = np.asarray(rgb224, dtype=np.uint8)
    np.savez_compressed(path, **payload)
    return {
        "control_steps": steps,
        "matched_budget_min": int(expected_counts.min()),
        "matched_budget_max": int(expected_counts.max()),
        "all_four_arms_share_m_t": True,
        "mask_jaccard_mean_vs_l11": {
            arm: float(jaccard[index].mean()) for index, arm in enumerate(ARMS)
        },
        "mean_dose": {
            arm: {
                key: float(metrics[key][index].mean())
                for key in metrics
            }
            for index, arm in enumerate(ARMS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", default="manual")
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if args.phase == "preflight" and any(seed > 104 for seed in seeds):
        raise ValueError("preflight is locked to seeds 100..104")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import cv2
    import numpy as np
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import (
        PCD_SOURCE, array_sha256, jsonable, restore_snapshot, snapshot_sha,
    )
    from research.semantic_token_cd.prompt_attn_shr_policy import geometry_patch_scores
    from research.semantic_token_cd.prompt_attn_layer_rollout import make_environment
    from research.semantic_token_cd.rollout_pilot import capture_snapshot
    from research.semantic_token_cd.spatial_grid_rollout import get_image_from_maniskill2_obs_dict, run_episode

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not (PCD_SOURCE / "pretrained/openvla-7b/config.json").is_file():
        raise FileNotFoundError(f"OpenVLA base checkpoint is missing: {PCD_SOURCE}")

    source = args.snapshot_artifact.resolve()
    artifact = args.artifact.resolve()
    canonical_episode_root = source / "episodes" / args.task / "vanilla"
    snapshot_root = source / "snapshots" / args.task
    if not (source / "SNAPSHOT_MANIFEST.jsonl").is_file():
        raise FileNotFoundError(f"canonical snapshot manifest missing under {source}")

    repo = Path(os.environ.get("TOKEN_CD", Path(__file__).resolve().parents[2])).resolve()
    script_path = Path(__file__).resolve()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True,
    )
    commit_label = git_commit.stdout.strip() if git_commit.returncode == 0 else "working_tree_hashes_only"
    lock = {
        "protocol_id": PROTOCOL,
        "task_universe": list(TASKS),
        "seeds_by_task": list(SEEDS),
        "preflight_seeds": list(range(100, 105)),
        "arms": list(ARMS),
        "canonical_snapshot_artifact": str(source),
        "checkpoint": str(PCD_SOURCE / "pretrained/openvla-7b"),
        "checkpoint_config_sha256": file_sha256(PCD_SOURCE / "pretrained/openvla-7b/config.json"),
        "geometry": {
            "processor_resize": "OpenVLAInference._resize_image, cv2.INTER_AREA to 224x224 RGB",
            "grayscale": "cv2.COLOR_RGB2GRAY",
            "edge": "cv2.Sobel float32 dx=1/dy=1 ksize=3, sqrt(Gx^2+Gy^2)",
            "patch_map": "16x16 row-major; exact non-overlapping 14x14 pixel mean",
            "tie_break": "descending score, ascending visual-token index",
        },
        "shared_downstream": {
            "attention_layers": [11], "lambda": LAMBDA, "beta": 0.0,
            "harmonic": "existing 16x16 four-neighbor Dirichlet implementation",
            "clean_branch": "same shared policy implementation and processor inputs",
            "negative_decode": "shared clean greedy teacher-forced prefix",
            "guided_dimensions": [0, 1, 2, 3, 4, 5],
            "gripper": "clean positive branch", "sampling": False,
        },
        "m_t_coupling": "compute the L11-Matched own-state m_t tape on arm A; freeze and reuse it by control-step index in B/C/D",
        "random": "SeedSequence([task_index, episode_seed, control_step, 0x50A77E11])",
        "code": {
            "git_commit": commit_label,
            "runner_sha256": file_sha256(script_path),
            "selector_sha256": file_sha256(repo / "research/semantic_token_cd/prompt_attn_shr_policy.py"),
        },
    }
    artifact.mkdir(parents=True, exist_ok=True)
    config_path = artifact / "CONFIG_LOCK.json"
    if config_path.exists() and json.loads(config_path.read_text()) != lock:
        raise RuntimeError(f"config lock mismatch; use a new artifact directory: {config_path}")
    atomic_json(config_path, lock)

    env, environment_id = make_environment(args.task, args.gpu)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, args.task, {}, False))
    policies = {arm: _arm_policy(base, args.task, arm) for arm in ARMS}
    task_out = artifact / "episodes" / args.task
    task_out.mkdir(parents=True, exist_ok=True)
    (artifact / "preflight_diagnostics").mkdir(parents=True, exist_ok=True)
    completed = []
    try:
        for seed in seeds:
            canonical_path = snapshot_root / f"seed_{seed:03d}.pkl"
            reference_path = canonical_episode_root / f"episode_{seed:03d}_summary.json"
            if not (canonical_path.is_file() and reference_path.is_file()):
                raise FileNotFoundError(f"missing canonical state/reference for {args.task} seed {seed}")
            reference = json.loads(reference_path.read_text())
            with canonical_path.open("rb") as stream:
                snapshot = pickle.load(stream)
            canonical_sha = snapshot_sha(snapshot)
            expected = (
                reference["canonical_snapshot_sha256"],
                reference["initial_state_sha256"],
                reference["initial_rgb_sha256"],
            )
            if canonical_sha != expected[0]:
                raise RuntimeError(f"canonical snapshot SHA mismatch for {args.task} seed {seed}")

            pair_path = task_out / f"episode_{seed:03d}_pair_arrays.npz"
            all_existing = {
                arm: _load_existing_arm(artifact, args.task, arm, seed) for arm in ARMS
            }
            if all(all_existing.values()) and pair_path.is_file():
                import numpy as np
                pair = np.load(pair_path)
                if not np.all(pair["matched_budget_m"] > 0):
                    raise RuntimeError(f"invalid saved m_t tape for {args.task} seed {seed}")
                completed.append(seed)
                print(json.dumps({"skip_complete": True, "task": args.task, "seed": seed}), flush=True)
                continue

            arm_data: dict[str, dict] = {}
            summaries: dict[str, dict] = {}
            rgb224_for_figure = None
            budget_tape: list[int] | None = None
            for arm in ARMS:
                policy = policies[arm]
                if arm == "l11_matched":
                    policy.selection_budget_schedule = None
                else:
                    if budget_tape is None:
                        prior = all_existing["l11_matched"]
                        if prior is not None:
                            budget_tape = [int(row["m_t"]) for row in json.loads(
                                (task_out / arm / f"episode_{seed:03d}_summary.json").read_text()
                            )["selector_trace"]]
                        else:
                            raise RuntimeError("L11-Matched arm must run first to establish shared m_t")
                    policy.selection_budget_schedule = tuple(budget_tape)

                existing = all_existing[arm]
                if existing is not None:
                    summary, loaded = existing
                    records = {
                        "selected_mask": loaded["selected_mask"],
                        "l11_scores": loaded["l11_scores"],
                        "geometry_scores": loaded["geometry_scores"],
                        "D_feat": np.asarray([x["D_feat"] for x in summary["selector_trace"]], dtype=np.float32),
                        "D_res": np.asarray([x["D_res"] for x in summary["selector_trace"]], dtype=np.float32),
                        "D_action": np.asarray([x["D_action"] for x in summary["selector_trace"]], dtype=np.float32),
                        "m_t": np.asarray([x["m_t"] for x in summary["selector_trace"]], dtype=np.int16),
                    }
                    arm_data[arm] = records
                    summaries[arm] = summary
                    if arm == "l11_matched":
                        budget_tape = records["m_t"].astype(int).tolist()
                    continue

                obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
                if state_sha != expected[1]:
                    raise RuntimeError(
                        f"restored canonical state mismatch for {args.task} seed {seed} arm {arm}: "
                        f"actual={state_sha} expected={expected[1]}"
                    )
                instruction = env.unwrapped.get_language_instruction()
                if instruction != reference["instruction"]:
                    raise RuntimeError(f"canonical instruction mismatch for {args.task} seed {seed}")
                policy.reset(instruction, seed=seed)
                policy._episode_trace = []
                policy._episode_logits = []
                if args.phase == "preflight":
                    image = get_image_from_maniskill2_obs_dict(env, obs)
                    _score, rgb224_for_figure = geometry_patch_scores(policy, image)

                started = time.monotonic()
                result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
                elapsed = time.monotonic() - started
                trace = jsonable(policy._episode_trace)
                if not trace or any(not row.get("technical_pass", True) for row in trace):
                    raise RuntimeError(f"empty/failed selector trace for {args.task} seed {seed} arm {arm}")
                if any(row.get("lambda") != LAMBDA or row.get("beta") != 0.0 for row in trace):
                    raise RuntimeError(f"lambda/beta drift for {args.task} seed {seed} arm {arm}")
                if any(row.get("guided_prefix") is not True or row.get("non_target_bit_identical") is not True for row in trace):
                    raise RuntimeError(f"shared downstream audit failed for {args.task} seed {seed} arm {arm}")

                arm_root = task_out / arm
                arm_root.mkdir(parents=True, exist_ok=True)
                arrays_path = arm_root / f"episode_{seed:03d}_arrays.npz"
                _write_arm_arrays(arrays_path, policy._episode_logits, actions)
                values = {
                    "selected_mask": np.stack([record["selected_mask"] for record in policy._episode_logits]),
                    "l11_scores": np.stack([record["prompt_attention"] for record in policy._episode_logits]),
                    "geometry_scores": np.stack([record["geometry_scores"] for record in policy._episode_logits]),
                    "D_feat": np.asarray([row["D_feat"] for row in trace], dtype=np.float32),
                    "D_res": np.asarray([row["D_res"] for row in trace], dtype=np.float32),
                    "D_action": np.asarray([row["D_action"] for row in trace], dtype=np.float32),
                    "m_t": np.asarray([row["m_t"] for row in trace], dtype=np.int16),
                }
                arm_data[arm] = values
                if arm == "l11_matched":
                    budget_tape = values["m_t"].astype(int).tolist()

                summary = {
                    "protocol_id": PROTOCOL, "phase": args.phase,
                    "task": args.task, "environment_id": environment_id,
                    "seed": seed, "arm": arm, "instruction": instruction,
                    "success": bool(result["success"]), "result": jsonable(result),
                    "failure_reason": reason, "control_steps": steps,
                    "runtime_seconds": elapsed, "action_jitter_index": jitter,
                    "canonical_snapshot_sha256": canonical_sha,
                    "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                    "attention_layers": [11], "lambda": LAMBDA, "beta": 0.0,
                    "budget_source": (
                        "own_state_l11_matched" if arm == "l11_matched"
                        else "same_episode_l11_matched_m_t_tape"
                    ),
                    "arrays_file": arrays_path.name,
                    "selector_trace": trace,
                }
                summary_path = arm_root / f"episode_{seed:03d}_summary.json"
                atomic_json(summary_path, summary)
                summaries[arm] = summary
                print(json.dumps({
                    "task": args.task, "seed": seed, "arm": arm,
                    "success": summary["success"], "steps": steps,
                    "runtime_seconds": round(elapsed, 2),
                    "mean_m": float(values["m_t"].mean()),
                }), flush=True)

            if budget_tape is None:
                raise RuntimeError("L11-Matched failed to produce the shared budget tape")
            initial_hashes = {
                arm: (summaries[arm]["initial_state_sha256"], summaries[arm]["initial_rgb_sha256"])
                for arm in ARMS
            }
            if any(pair[0] != expected[1] for pair in initial_hashes.values()):
                raise RuntimeError(f"arms failed canonical initial-state pairing: {initial_hashes}")
            if len({pair[1] for pair in initial_hashes.values()}) != 1:
                raise RuntimeError(f"arms do not share one current-renderer initial RGB: {initial_hashes}")
            pairing_metrics = _write_pair_arrays(
                pair_path, arm_data, budget_tape,
                rgb224_for_figure if args.phase == "preflight" else None,
            )
            pair_summary = {
                "protocol_id": PROTOCOL, "task": args.task, "seed": seed,
                "arms": list(ARMS), "canonical_snapshot_sha256": canonical_sha,
                "initial_state_sha256": expected[1], "initial_rgb_sha256": initial_hashes["l11_matched"][1],
                "canonical_reference_rgb_sha256": expected[2],
                "historical_rgb_matches_current_renderer": initial_hashes["l11_matched"][1] == expected[2],
                "all_current_arms_same_rgb": len({value[1] for value in initial_hashes.values()}) == 1,
                "vanilla_reference_success": reference["success"],
                "success_by_arm": {arm: bool(summaries[arm]["success"]) for arm in ARMS},
                "pair_arrays": pair_path.name, **pairing_metrics,
            }
            atomic_json(task_out / f"episode_{seed:03d}_pair_summary.json", pair_summary)
            if args.phase == "preflight" and seed == 100 and rgb224_for_figure is not None:
                _save_preflight_figure(
                    artifact / "preflight_diagnostics" / f"{args.task}_seed100_selector_maps.png",
                    rgb224_for_figure, arm_data,
                )
            completed.append(seed)
    finally:
        env.close()

    print(json.dumps({
        "task": args.task, "phase": args.phase, "completed": len(completed),
        "seeds": completed, "artifact": str(artifact),
    }), flush=True)


if __name__ == "__main__":
    main()

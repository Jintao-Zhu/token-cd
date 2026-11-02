"""Offline all-layer prompt-attention screen for pi0.5 on LIBERO-10 states.

Runs in the LIBERO environment (rendering + instance segmentation) and asks the
pi0.5 websocket server for the 18 PaliGemma attention maps (`pi05_arm=capture`).

Judging protocol matches the pi0-FAST screen:

* **swap probe** — replace the instruction's target object phrase with another
  object in the same scene; a good layer moves its top-K attended patches from
  the original object's ground-truth mask to the new object's mask
  (`mask_response = 0.5 * (gain_new + drop_old)`).
* **paraphrase probe** — LIBERO-Plus ships LLM-rewritten instructions for the
  same task (`*_language_N.bddl`); a good layer keeps its top-K set stable
  (`synonym_mask_jaccard`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
sys.path.insert(0, str(REPO))
from research.semantic_token_cd.pi05_libero_plus_paired_rollout import (  # noqa: E402
    clean_instruction,
)
GRID = 16
N_VISUAL = GRID * GRID
DEFAULT_OUT = REPO / "artifacts/pi05_layer_selection_v1"


def axis_angle(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] * value[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros((3,), dtype=np.float64)
    return value[:3] * 2.0 * math.acos(value[3]) / denominator


def object_phrase(instance: str) -> str:
    words = [part for part in instance.split("_") if not part.isdigit()]
    return " ".join(words)


def pick_probe_objects(instruction: str, instances: list[str]) -> tuple[str, str] | None:
    text = instruction.lower()
    present = [name for name in instances if len(name) > 3]
    in_text = [name for name in present if object_phrase(name).lower() in text]
    if in_text:
        target = in_text[0]
        others = [
            name
            for name in present
            if name != target and object_phrase(name).lower() not in text
        ]
        others.sort(key=len)
        if others:
            return target, others[0]
    if len(present) >= 2:
        present.sort(key=len)
        return present[0], present[1]
    return None


def resize_mask(mask: np.ndarray, grid: int = GRID) -> np.ndarray:
    height, width = mask.shape
    block_h, block_w = height // grid, width // grid
    trimmed = mask[: block_h * grid, : block_w * grid]
    return trimmed.reshape(grid, block_h, grid, block_w).mean(axis=(1, 3)) > 0.35


def topk(scores: np.ndarray, count: int) -> list[int]:
    order = np.lexsort((np.arange(scores.size), -np.asarray(scores)))
    return [int(index) for index in order[:count]]


def load_paraphrases(bddl_root: Path, task) -> list[str]:
    import libero.libero.envs.bddl_utils as BDDLUtils

    folder = bddl_root / task.problem_folder
    stem = re.sub(
        r"(_view_.*|_noise_\d+|_table_\d+|_tb_\d+|_add_\d+|_light_\d+|_language_\d+)$",
        "",
        re.sub(r"\.bddl$", "", task.bddl_file),
    )
    out: list[str] = []
    for path in sorted(folder.glob(f"{stem}_language_*.bddl")):
        try:
            info = BDDLUtils.get_problem_info(str(path))
        except Exception:  # noqa: BLE001
            continue
        text = str(info.get("language_instruction", "")).strip()
        if text and text not in out and text.lower() != str(task.language).strip().lower():
            out.append(text)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--states-per-task", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from openpi_client import image_tools, websocket_client_policy

    args.out.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    bddl_root = Path(get_libero_path("bddl_files"))

    # One representative instance per base skill (the suite is ordered by
    # perturbation variant, so the first N indices are all the same skill).
    seen: set[str] = set()
    task_indices: list[int] = []
    for index in range(suite.n_tasks):
        key = clean_instruction(suite.get_task(index).name)
        if key in seen:
            continue
        seen.add(key)
        task_indices.append(index)
        if len(task_indices) >= args.tasks:
            break
    print(f"probe tasks: {task_indices}", flush=True)

    def scores_for(base_image, wrist_image, state, prompt: str) -> np.ndarray:
        response = client.infer(
            {
                "observation/image": base_image,
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "prompt": prompt,
                "pi05_arm": "capture",
            }
        )
        return np.asarray(response["attention_scores"], dtype=np.float32)

    records = []
    for task_index in task_indices:
        task = suite.get_task(task_index)
        init_states = suite.get_task_init_states(task_index)
        instruction = clean_instruction(task.name)
        env = SegmentationRenderEnv(
            bddl_file_name=str(bddl_root / task.problem_folder / task.bddl_file),
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(7)
        env.reset()
        instances = sorted(env.instance_to_id)
        chosen = pick_probe_objects(instruction, instances)
        if chosen is None:
            print(f"[skip] task {task_index}: no probeable objects", flush=True)
            env.close()
            continue
        target, swap = chosen
        swap_instruction = instruction.replace(object_phrase(target), object_phrase(swap))
        if swap_instruction == instruction:
            swap_instruction = f"{instruction} {object_phrase(swap)}"
        paraphrases = load_paraphrases(bddl_root, task)
        print(
            f"[task {task_index}] target={target} swap={swap} "
            f"paras={len(paraphrases)} | {instruction[:70]}",
            flush=True,
        )
        print(f"    swap prompt: {swap_instruction[:80]}", flush=True)

        for state_index in range(min(args.states_per_task, len(init_states))):
            observation = env.set_init_state(init_states[state_index])
            for _ in range(10):
                observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            segmentation = np.asarray(observation["agentview_segmentation_instance"])
            segmentation = segmentation[..., 0] if segmentation.ndim == 3 else segmentation
            # the policy sees the image rotated by 180 degrees, so the mask must
            # be transformed identically
            segmentation = segmentation[::-1, ::-1]
            masks = {
                name: resize_mask(segmentation == env.instance_to_id[name])
                for name in (target, swap)
            }
            base_image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
            wrist_image = np.ascontiguousarray(
                observation["robot0_eye_in_hand_image"][::-1, ::-1]
            )
            base_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(base_image, 224, 224)
            )
            wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_image, 224, 224)
            )
            state = np.concatenate(
                (
                    observation["robot0_eef_pos"],
                    axis_angle(observation["robot0_eef_quat"]),
                    observation["robot0_gripper_qpos"],
                )
            ).astype(np.float32)

            original = scores_for(base_image, wrist_image, state, instruction)
            swapped = scores_for(base_image, wrist_image, state, swap_instruction)
            paraphrase = (
                scores_for(base_image, wrist_image, state, paraphrases[0])
                if paraphrases
                else None
            )

            for layer in range(original.shape[0]):
                original_top = set(topk(original[layer], args.top_k))
                swapped_top = set(topk(swapped[layer], args.top_k))
                target_mask = masks[target].reshape(-1)
                swap_mask = masks[swap].reshape(-1)
                gain = (
                    len(swapped_top & set(np.flatnonzero(swap_mask)))
                    - len(original_top & set(np.flatnonzero(swap_mask)))
                ) / args.top_k
                drop = (
                    len(original_top & set(np.flatnonzero(target_mask)))
                    - len(swapped_top & set(np.flatnonzero(target_mask)))
                ) / args.top_k
                record = {
                    "task_index": task_index,
                    "task_name": task.name,
                    "state_index": state_index,
                    "layer": layer,
                    "target_instance": target,
                    "swap_instance": swap,
                    "new_mask_gain": float(gain),
                    "old_mask_drop": float(drop),
                    "mask_response": float(0.5 * (gain + drop)),
                    "correct_both": bool(gain > 0 and drop > 0),
                    "target_mask_coverage": float(
                        len(original_top & set(np.flatnonzero(target_mask))) / args.top_k
                    ),
                    "target_mask_fraction": float(target_mask.mean()),
                }
                if paraphrase is not None:
                    para_top = set(topk(paraphrase[layer], args.top_k))
                    record["synonym_mask_jaccard"] = float(
                        len(original_top & para_top) / max(1, len(original_top | para_top))
                    )
                records.append(record)
        env.close()

    write_summary(records, args.out, args.top_k)


def write_summary(records: list[dict], out: Path, top_k: int) -> None:
    if not records:
        raise SystemExit("no records collected")
    summary = []
    for layer in sorted({row["layer"] for row in records}):
        rows = [row for row in records if row["layer"] == layer]
        jaccard = [row["synonym_mask_jaccard"] for row in rows if "synonym_mask_jaccard" in row]
        summary.append(
            {
                "layer": layer,
                "n_records": len(rows),
                "mask_response": float(np.mean([row["mask_response"] for row in rows])),
                "target_mask_coverage": float(
                    np.mean([row["target_mask_coverage"] for row in rows])
                ),
                "correct_both_fraction": float(
                    np.mean([row["correct_both"] for row in rows])
                ),
                "synonym_mask_jaccard": float(np.mean(jaccard)) if jaccard else None,
            }
        )
    best = max(summary, key=lambda row: row["mask_response"])
    payload = {
        "protocol": "PI05_LAYER_SELECTION_OFFLINE_V1",
        "top_k": top_k,
        "n_states": len({(row["task_index"], row["state_index"]) for row in records}),
        "per_layer": summary,
        "best_layer_by_mask_response": best["layer"],
        "caveat": (
            "Offline attention-layer candidate screen only, not closed-loop "
            "validation; it nominates a layer, it does not prove a success-rate gain."
        ),
        "records": records,
    }
    (out / "LAYER_SCAN_RESULTS.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["layer,mask_response,target_mask_coverage,correct_both_fraction,synonym_mask_jaccard"]
    for row in summary:
        jaccard = "" if row["synonym_mask_jaccard"] is None else f"{row['synonym_mask_jaccard']:.6f}"
        lines.append(
            f"{row['layer']},{row['mask_response']:.6f},{row['target_mask_coverage']:.6f},"
            f"{row['correct_both_fraction']:.6f},{jaccard}"
        )
    (out / "per_layer.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"best layer by mask_response: {best['layer']}", flush=True)


if __name__ == "__main__":
    main()

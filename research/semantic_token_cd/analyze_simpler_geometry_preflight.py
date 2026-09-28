"""Fail-closed structural/data audit for the four-arm SIMPLER preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from research.semantic_token_cd.prompt_attn_shr_policy import stable_top_m
from research.semantic_token_cd.semantic_recon_rollout import TASK_INDEX


TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ARMS = ("l11_matched", "geometry_mask", "geometry_protect", "random_matched")
SEEDS = range(100, 105)
RANDOM_SALT = 0x50A77E11


def sobel_scores(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise AssertionError(f"bad OpenVLA processor RGB: {rgb.shape}/{rgb.dtype}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag.reshape(16, 14, 16, 14).mean(axis=(1, 3)).reshape(256).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--mark-gate", action="store_true",
                        help="write the full-run gate after all structural checks pass")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    source = args.snapshot_artifact.resolve()
    checks = []
    errors = []

    for task in TASKS:
        for seed in SEEDS:
            task_root = artifact / "episodes" / task
            pair_summary_path = task_root / f"episode_{seed:03d}_pair_summary.json"
            pair_arrays_path = task_root / f"episode_{seed:03d}_pair_arrays.npz"
            if not pair_summary_path.is_file() or not pair_arrays_path.is_file():
                errors.append(f"missing pair outputs: {task} seed {seed}")
                continue
            pair_summary = json.loads(pair_summary_path.read_text())
            pair = np.load(pair_arrays_path)
            try:
                masks = pair["selected_masks"].astype(bool)
                l11 = pair["l11_scores"]
                geo = pair["geometry_scores"]
                budget = pair["matched_budget_m"].astype(int)
                rgb = pair["processor_rgb_224_first_step"]
                if masks.shape != l11.shape or masks.shape != geo.shape or masks.shape[0] != 4 or masks.shape[2] != 256:
                    raise AssertionError(f"score/mask dimensions invalid: {masks.shape}, {l11.shape}, {geo.shape}")
                if rgb.shape != (224, 224, 3):
                    raise AssertionError(f"224 RGB mapping invalid: {rgb.shape}")
                expected_geo = sobel_scores(rgb)
                if not np.allclose(geo[:, 0], expected_geo[None, :], rtol=0, atol=1e-5):
                    raise AssertionError("first-step Sobel scores do not match 14x14 patch recomputation")
                if not np.allclose(l11[:, 0], l11[0, 0][None, :], rtol=0, atol=1e-7):
                    raise AssertionError("first-step L11 scores differ despite identical input RGB")
                if not np.allclose(geo[:, 0], geo[0, 0][None, :], rtol=0, atol=1e-7):
                    raise AssertionError("first-step geometry scores differ across arms")
                if not np.array_equal(masks.sum(axis=-1).astype(int), np.broadcast_to(budget, (4, len(budget)))):
                    raise AssertionError("mask count differs from shared matched m_t")
                if not np.all(pair["jaccard_vs_l11"][0] == 1.0):
                    raise AssertionError("A-vs-A Jaccard is not 1")

                ref = json.loads((source / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json").read_text())
                if pair_summary["canonical_snapshot_sha256"] != ref["canonical_snapshot_sha256"]:
                    raise AssertionError("canonical snapshot does not match the reference artifact")
                if pair_summary["initial_state_sha256"] != ref["initial_state_sha256"]:
                    raise AssertionError("initial simulator state does not match the canonical reference")
                actual_rgbs = {
                    json.loads((task_root / arm / f"episode_{seed:03d}_summary.json").read_text())["initial_rgb_sha256"]
                    for arm in ARMS
                }
                if len(actual_rgbs) != 1 or pair_summary["initial_rgb_sha256"] not in actual_rgbs:
                    raise AssertionError("the four arms do not share one current-renderer initial RGB")
                if pair_summary.get("all_current_arms_same_rgb") is not True:
                    raise AssertionError("current-renderer RGB pairing flag false")
                if not pair_summary["all_four_arms_share_m_t"]:
                    raise AssertionError("shared m_t flag false")

                for arm_index, arm in enumerate(ARMS):
                    summary_path = task_root / arm / f"episode_{seed:03d}_summary.json"
                    arm_arrays_path = task_root / arm / f"episode_{seed:03d}_arrays.npz"
                    summary = json.loads(summary_path.read_text())
                    arm_arrays = np.load(arm_arrays_path)
                    trace = summary["selector_trace"]
                    if len(trace) != masks.shape[1] or len(trace) != len(budget):
                        raise AssertionError(f"trace length mismatch: {arm}")
                    if [int(x["m_t"]) for x in trace] != budget.tolist():
                        raise AssertionError(f"m_t tape mismatch: {arm}")
                    if any(x["lambda"] != 0.5 or x["beta"] != 0.0 for x in trace):
                        raise AssertionError(f"lambda/beta changed: {arm}")
                    if any(x["guided_prefix"] is not True or x["non_target_bit_identical"] is not True for x in trace):
                        raise AssertionError(f"harmonic/prefix audit failed: {arm}")
                    for step, row in enumerate(trace):
                        selected = np.flatnonzero(masks[arm_index, step]).tolist()
                        m = int(budget[step])
                        if arm == "l11_matched":
                            expected = stable_top_m(l11[arm_index, step], m)
                        elif arm == "geometry_mask":
                            expected = stable_top_m(geo[arm_index, step], m)
                        elif arm == "geometry_protect":
                            protected = set(stable_top_m(geo[arm_index, step], m))
                            order = np.lexsort((np.arange(256), -l11[arm_index, step].astype(np.float64)))
                            expected = sorted(int(x) for x in order if int(x) not in protected)[:m]
                            if len(expected) != m:
                                raise AssertionError(f"Geometry-Protect impossible for m={m}")
                        else:
                            task_idx = TASK_INDEX[task]
                            expected_seed = int(np.random.SeedSequence([
                                task_idx, seed, step, RANDOM_SALT,
                            ]).generate_state(1, dtype=np.uint32)[0])
                            expected = sorted(int(x) for x in np.random.default_rng(expected_seed).choice(
                                256, size=m, replace=False
                            ))
                            if row["random_seed"] != expected_seed:
                                raise AssertionError("Random-Matched seed rule changed")
                        if selected != expected:
                            raise AssertionError(f"selector mask mismatch at {arm} step {step}")
                    for key in ("positive_logits", "negative_logits", "selected_mask", "l11_scores", "geometry_scores"):
                        if key not in arm_arrays:
                            raise AssertionError(f"missing required per-step record {key}: {arm}")
                    checks.append({"task": task, "seed": seed, "pair_exact": True,
                                   "steps": len(budget), "m_min": int(budget.min()),
                                   "m_max": int(budget.max())})
            except Exception as exc:
                errors.append(f"{task} seed {seed}: {type(exc).__name__}: {exc}")

    report = {
        "protocol_id": "SIMPLER_L11_GEOMETRY_MASK_PROTECT_RANDOM_PREFLIGHT_V1",
        "expected_pairs": 20,
        "passed_pairs": len(checks),
        "checks": checks,
        "errors": errors,
        "structural_pass": not errors and len(checks) == 20,
        "visual_review_required": True,
    }
    report_path = artifact / "PREFLIGHT_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if errors or len(checks) != 20:
        raise SystemExit(f"preflight failed; see {report_path}: {errors[:10]}")
    print(json.dumps({"structural_pass": True, "passed_pairs": len(checks), "report": str(report_path)}))
    if args.mark_gate:
        gate = {
            "protocol_id": "SIMPLER_L11_GEOMETRY_MASK_PROTECT_RANDOM_PREFLIGHT_V1",
            "pass": True,
            "structural_pass": True,
            "visual_maps_reviewed": True,
            "pairs": len(checks),
            "report": str(report_path),
        }
        (artifact / "PREFLIGHT_GATE.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

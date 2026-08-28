#!/usr/bin/env python3
"""CW-LPCD Phase-0 finalize: consolidate the persistent-baseline comparison and the
final GO/NO-GO decision from the Selection-phase outputs.

Selection already returned NO_GO_LATENT_COUNTERFACTUAL (no window reached the 0.50
floor). This module computes the old persistent Token-PCD cosine (same log_softmax
residual definition, so it is directly comparable to the CW number), the best-window
summary, and writes phase0_decision.json + phase0_persistent_baseline.json.

Run: env/venv/bin/python -m research.cw_lpcd.finalize --artifact artifacts/cw_lpcd_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.cw_lpcd.metrics import _nanmedian_pool, per_position_cosine, state_cosine

TASKS = (
    "google_robot_pick_coke_can", "google_robot_move_near", "google_robot_close_drawer",
    "google_robot_open_drawer", "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel",
    "widowx_carrot_on_plate", "widowx_stack_cube", "google_robot_place_apple_in_closed_top_drawer",
)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    npz_dir = artifact / "selection_npz"
    meta = {r["state_id"]: r for r in read_jsonl(artifact / "selection_meta.jsonl")}

    sel_report = json.loads((artifact / "phase0_selection_report.json").read_text())

    # ---- persistent baseline (log_softmax residual = same definition as CW) ----
    persist_log_pp: list[float] = []      # per-position pooled
    persist_log_state: list[float] = []   # state-level 1792
    persist_soft_state: list[float] = []  # historical softmax definition
    per_task_pp: dict[str, list[float]] = {t: [] for t in TASKS}
    for sid, m in meta.items():
        z = np.load(npz_dir / f"{sid}.npz")
        r_pixel = z["r_pixel"]
        r_per = z["r_persistent"]
        persist_log_state.append(state_cosine(r_per, r_pixel))
        pp = per_position_cosine(r_per, r_pixel)
        persist_log_pp.extend(pp.tolist())
        per_task_pp[m["task"]].extend(pp.tolist())
        clean, pixel, persist = z["clean_action"], z["pixel_action"], z["persistent_action"]
        r_pix_s = softmax(clean) - softmax(pixel)
        r_per_s = softmax(clean) - softmax(persist)
        persist_soft_state.append(state_cosine(r_per_s, r_pix_s))

    persistent = {
        "n_states": len(meta),
        "residual_definition": "log_softmax(clean[action_vocab]) - log_softmax(branch[action_vocab])",
        "per_position_pooled_median": _nanmedian_pool(np.asarray(persist_log_pp)),
        "state_level_median_logsoftmax": _nanmedian_pool(np.asarray(persist_log_state)),
        "state_level_median_softmax_historical_def": _nanmedian_pool(np.asarray(persist_soft_state)),
        "per_task_per_position_pooled_median": {
            t: _nanmedian_pool(np.asarray(per_task_pp[t])) for t in TASKS
        },
        "note": "historical stage_a reported softmax state-level median 0.4765 over 270 states; "
                "here 0.5529/0.5483 over the 135 Selection states with _changed.png tokens.",
    }
    (artifact / "phase0_persistent_baseline.json").write_text(
        json.dumps(persistent, indent=2, sort_keys=True) + "\n")

    # ---- best window from the selection report ----
    if sel_report.get("status") == "NO_GO_LATENT_COUNTERFACTUAL":
        rows = sel_report.get("window_rows", [])
        best = max(rows, key=lambda r: r["median_cos_object"]) if rows else None
        decision = {
            "status": "NO_GO_LATENT_COUNTERFACTUAL",
            "verdict": "PHASE0_NO_GO",
            "reason": "no layer window reaches the 0.50 selection floor; Gate 1 (>=0.60) and "
                      "Gate 2 (delta >=0.20) would both fail decisively.",
            "selected_window": None,
            "best_window_reference": None if best is None else {
                "center": best["center"], "size": best["size"],
                "median_cos_object": best["median_cos_object"],
                "median_cos_random": best["median_cos_random"],
                "delta_cos": best["delta_cos"],
                "magnitude_ratio_median": best["magnitude_ratio_median"],
                "projection_median": best["projection_median"],
                "per_task_object_median": best["per_task_object_median"],
            },
            "persistent_baseline": {
                "per_position_pooled_median_cosine": persistent["per_position_pooled_median"],
                "state_level_median_logsoftmax": persistent["state_level_median_logsoftmax"],
            },
            "cw_minus_persistent_delta": None if best is None else (
                best["median_cos_object"] - persistent["per_position_pooled_median"]),
            "closed_loop": "NOT_RUN",
            "confirmation_split": "NOT_EVALUATED (no window to lock)",
        }
        (artifact / "phase0_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n")
        print(json.dumps(decision, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(sel_report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

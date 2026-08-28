#!/usr/bin/env python3
"""CW-LPCD STEP 3-6 + STEP 8-9 + STEP 10: Selection-phase residual computation,
21-window search, deterministic window lock.

For every Selection state (135) this computes clean / pixel / persistent / and all
21 latent + 21 random CW-window logits under a single shared Vanilla prefix, saves
the 256-dim action-vocab logits + log_softmax residuals, then aggregates the STEP 9
metrics over the Selection split and applies the deterministic window-selection rule
to write CW_LPCD_LOCK.yaml.

Run:
  official-reproductions/.../env/venv/bin/python -m research.cw_lpcd.step3_9_selection \
      --artifact artifacts/cw_lpcd_v1 [--limit N] [--aggregate-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from research.cw_lpcd.core import (
    PCD_ROOT, TASKS, WINDOW_CENTERS, WINDOW_SIZES, enumerate_windows, load_model,
)
from research.cw_lpcd.engine import compute_state, load_position_mean, matched_random_ids, random_seed_for
from research.cw_lpcd.metrics import (
    _nanmedian_pool, norm_ratio, per_position_cosine, projection, sign_agreement,
    state_cosine, top1_shift_agreement,
)

STATES_LOCK = Path(
    "artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl"
)
MEAN_PATH = Path(
    "artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/position_conditioned_visual_mean.pt"
)

SELECTION_GATE = 0.50  # Median_Cos_Object floor before window selection (spec STEP 13)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def select_window(rows: list[dict]) -> tuple[dict | None, list[dict]]:
    """Deterministic STEP 13 rule. Returns (selected, all_rows)."""
    surviving = [r for r in rows if r["median_cos_object"] >= SELECTION_GATE]
    if not surviving:
        return None, rows
    max_delta = max(r["delta_cos"] for r in surviving)
    tier = [r for r in surviving if abs(r["delta_cos"] - max_delta) <= 1e-9]
    max_cos = max(r["median_cos_object"] for r in tier)
    tier = [r for r in tier if abs(r["median_cos_object"] - max_cos) <= 1e-9]
    min_size = min(r["size"] for r in tier)
    tier = [r for r in tier if r["size"] == min_size]
    tier = sorted(tier, key=lambda r: r["center"])
    return tier[0], rows


def aggregate(selection_ids: list[str], meta: dict, npz_dir: Path, windows: list[dict]) -> list[dict]:
    window_rows = []
    for wi, w in enumerate(windows):
        obj_pool, rnd_pool = [], []
        state_cos_obj, state_cos_rnd = [], []
        mag_pool, proj_pool, sign_pool, top1_pool = [], [], [], []
        per_task_obj: dict[str, list[float]] = {t: [] for t in TASKS}
        for sid in selection_ids:
            z = np.load(npz_dir / f"{sid}.npz")
            r_pixel = z["r_pixel"]                       # [7, 256]
            r_latent = z["r_latent"][wi]                 # [7, 256]
            r_random = z["r_random"][wi]
            obj_pool.extend(per_position_cosine(r_latent, r_pixel).tolist())
            rnd_pool.extend(per_position_cosine(r_random, r_pixel).tolist())
            state_cos_obj.append(state_cosine(r_latent, r_pixel))
            state_cos_rnd.append(state_cosine(r_random, r_pixel))
            mag_pool.extend(norm_ratio(r_latent, r_pixel).tolist())
            proj_pool.extend(projection(r_latent, r_pixel).tolist())
            sign_pool.extend(sign_agreement(r_latent, r_pixel).tolist())
            top1_pool.extend(top1_shift_agreement(z["latent_action"][wi], z["pixel_action"]).tolist())
            per_task_obj[meta[sid]["task"]].extend(per_position_cosine(r_latent, r_pixel).tolist())
        med_obj = _nanmedian_pool(np.asarray(obj_pool))
        med_rnd = _nanmedian_pool(np.asarray(rnd_pool))
        window_rows.append({
            "center": w["center"], "size": w["size"], "start": w["start"], "end": w["end"],
            "median_cos_object": med_obj,
            "median_cos_random": med_rnd,
            "delta_cos": med_obj - med_rnd,
            "state_cos_object_median": _nanmedian_pool(np.asarray(state_cos_obj)),
            "state_cos_random_median": _nanmedian_pool(np.asarray(state_cos_rnd)),
            "magnitude_ratio_median": _nanmedian_pool(np.asarray(mag_pool)),
            "projection_median": _nanmedian_pool(np.asarray(proj_pool)),
            "sign_agreement_median": _nanmedian_pool(np.asarray(sign_pool)),
            "top1_shift_agreement_median": _nanmedian_pool(np.asarray(top1_pool)),
            "per_task_object_median": {t: _nanmedian_pool(np.asarray(per_task_obj[t])) for t in TASKS},
        })
    return window_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, default=PCD_ROOT)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--states-lock", type=Path, default=STATES_LOCK)
    parser.add_argument("--mean-path", type=Path, default=MEAN_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    pcd_root = args.pcd_root.resolve()
    artifact = args.artifact.resolve()
    npz_dir = artifact / "selection_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads((artifact / "FROZEN_SPLIT.json").read_text())
    selection_ids = split["selection_state_ids"]
    windows = enumerate_windows()

    if not args.aggregate_only:
        states = {r["state_id"]: r for r in read_jsonl(args.states_lock.resolve())}
        manifest = {r["state_id"]: r for r in read_jsonl(artifact / "object_token_manifest.jsonl")}
        torch.manual_seed(20260822)
        torch.cuda.manual_seed_all(20260822)
        sys.path.insert(0, str(pcd_root / "source/PCD"))
        model, processor = load_model()
        mean = load_position_mean(args.mean_path.resolve())
        mean = mean.to(model.device, dtype=torch.bfloat16)
        print(json.dumps({"model_loaded": True, "vocab_size": int(model.vocab_size),
                          "n_selection": len(selection_ids), "n_windows": len(windows)}), flush=True)

        todo = selection_ids[: args.limit] if args.limit else selection_ids
        meta_rows = []
        for ordinal, sid in enumerate(todo):
            out_npz = npz_dir / f"{sid}.npz"
            if out_npz.exists():
                continue
            row = states[sid]
            object_ids = manifest[sid]["object_token_ids"]
            random_ids = matched_random_ids(object_ids, random_seed_for(sid))
            result = compute_state(model, processor, pcd_root, row, object_ids, random_ids, mean, windows)
            np.savez_compressed(
                out_npz,
                clean_ids=np.asarray(result["clean_ids"], dtype=np.int64),
                object_ids=np.asarray(result["object_ids"], dtype=np.int64),
                random_ids=np.asarray(result["random_ids"], dtype=np.int64),
                clean_action=result["clean_action"].astype(np.float32),
                pixel_action=result["pixel_action"].astype(np.float32),
                persistent_action=result["persistent_action"].astype(np.float32),
                latent_action=result["latent_action"].astype(np.float32),
                random_action=result["random_action"].astype(np.float32),
                r_pixel=result["r_pixel"].astype(np.float32),
                r_persistent=result["r_persistent"].astype(np.float32),
                r_latent=result["r_latent"].astype(np.float32),
                r_random=result["r_random"].astype(np.float32),
            )
            meta_rows.append({
                "state_id": sid, "task": row["task"], "seed": row["seed"],
                "npz": str(out_npz.relative_to(artifact)),
                "n_object": len(object_ids), "n_random": len(random_ids),
                "random_seed": random_seed_for(sid), "clean_ids": result["clean_ids"],
            })
            if ordinal % 10 == 0:
                print(json.dumps({"done": ordinal, "state_id": sid, "n_object": len(object_ids)}), flush=True)

        meta_path = artifact / "selection_meta.jsonl"
        existing = {r["state_id"]: r for r in read_jsonl(meta_path)} if meta_path.exists() else {}
        for r in meta_rows:
            existing[r["state_id"]] = r
        with meta_path.open("w") as handle:
            for sid in selection_ids:
                if sid in existing:
                    handle.write(json.dumps(existing[sid], sort_keys=True) + "\n")
        print(json.dumps({"compute_done": len(todo)}), flush=True)

    # ---------------- aggregate + select + lock ----------------
    if args.limit and not args.aggregate_only:
        print(json.dumps({"note": "limit set; skipping aggregate"}), flush=True)
        return

    meta = {r["state_id"]: r for r in read_jsonl(artifact / "selection_meta.jsonl")}
    missing = [sid for sid in selection_ids if sid not in meta or not (npz_dir / f"{sid}.npz").exists()]
    if missing:
        print(json.dumps({"error": "missing selection states", "n_missing": len(missing),
                          "first": missing[:5]}), flush=True)
        sys.exit(2)

    window_rows = aggregate(selection_ids, meta, npz_dir, windows)
    selected, all_rows = select_window(window_rows)

    # write phase0_selection.csv (one row per window)
    import csv
    csv_path = artifact / "phase0_selection.csv"
    fieldnames = ["center", "size", "start", "end", "median_cos_object", "median_cos_random",
                  "delta_cos", "state_cos_object_median", "state_cos_random_median",
                  "magnitude_ratio_median", "projection_median", "sign_agreement_median",
                  "top1_shift_agreement_median"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in fieldnames})
    # per-task csv
    pt_path = artifact / "phase0_selection_per_task.csv"
    with pt_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["center", "size"] + list(TASKS))
        for r in all_rows:
            writer.writerow([r["center"], r["size"]] + [r["per_task_object_median"][t] for t in TASKS])

    if selected is None:
        report = {"status": "NO_GO_LATENT_COUNTERFACTUAL",
                  "reason": "no window with Median_Cos_Object >= 0.50",
                  "selection_gate": SELECTION_GATE, "window_rows": all_rows}
        (artifact / "phase0_selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        sys.exit(0)

    # freeze the window (STEP 10)
    lock = {
        "protocol_id": "CW-LPCD-PHASE0-LOCK-R1",
        "selected_window": {k: selected[k] for k in ("center", "size", "start", "end")},
        "object_patch_threshold": 0.25,
        "random_rule": "same count, sampled without replacement from non-object tokens; seed=hash(state_uid)=int(sha256(state_id)[:8],16)",
        "pixel_pcd_alpha": 0.8,
        "action_vocab_slice": [31744, 31999],
        "residual_definition": "log_softmax(z_clean[action_vocab]) - log_softmax(z_branch[action_vocab]); per-position 256-dim cosine, pooled median",
        "window_centers": list(WINDOW_CENTERS),
        "window_sizes": list(WINDOW_SIZES),
        "split_sha256": (artifact / "FROZEN_SPLIT.sha256").read_text().split()[0],
        "selection_gate": SELECTION_GATE,
        "selection_decision": {"median_cos_object": selected["median_cos_object"],
                               "median_cos_random": selected["median_cos_random"],
                               "delta_cos": selected["delta_cos"]},
    }
    (artifact / "CW_LPCD_LOCK.yaml").write_text(_to_yaml(lock))
    report = {"status": "SELECTED", "selected_window": lock["selected_window"],
              "median_cos_object": selected["median_cos_object"],
              "median_cos_random": selected["median_cos_random"],
              "delta_cos": selected["delta_cos"],
              "state_cos_object_median": selected["state_cos_object_median"],
              "state_cos_random_median": selected["state_cos_random_median"],
              "magnitude_ratio_median": selected["magnitude_ratio_median"],
              "projection_median": selected["projection_median"],
              "sign_agreement_median": selected["sign_agreement_median"],
              "top1_shift_agreement_median": selected["top1_shift_agreement_median"],
              "per_task_object_median": selected["per_task_object_median"],
              "lock_path": "CW_LPCD_LOCK.yaml"}
    (artifact / "phase0_selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def _to_yaml(obj) -> str:
    lines = []
    def emit(key, value, indent=0):
        pad = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            for k, v in value.items():
                emit(k, v, indent + 1)
        elif isinstance(value, (list, tuple)):
            lines.append(f"{pad}{key}:")
            for v in value:
                lines.append(f"{pad}  - {v}")
        else:
            lines.append(f"{pad}{key}: {value}")
    for k, v in obj.items():
        emit(k, v)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

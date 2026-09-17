"""Analyze the offline four-branch decode for the cross-instruction question.

Reads offline_branches_{top,middle}.json (per emitted frame the four branches
instruction x region) and reports:
  1. within-frame effect size of target vs other region under each instruction;
  2. same-mask instruction sensitivity (mask top under 'open top' vs under
     'open middle', and mirror for middle);
  3. on the 20 strictly paired step-0 frames: whether guided translation is
     aligned with the *target* handle for the target branch (uses eef from the
     paired snapshot), i.e. does the same mask's effect track the instruction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.xdrawer_protocol import (
    DRAWERS, INSTRUCTION_BY_DRAWER, PROTOCOL,
)


def load(path):
    return json.loads(Path(path).read_text())


def eff(rec):
    """SHR-guided change norm over action dims 0..5 (gripper dim is clean)."""
    c = np.asarray(rec["clean_action"], dtype=np.float64)
    g = np.asarray(rec["guided_action"], dtype=np.float64)
    delta = g[:6] - c[:6]
    return float(np.linalg.norm(delta)), delta


def geometry_from_emitted(root: Path, origin: str, seed: int, step: int) -> dict:
    """Return {drawer: handle_p} read from the matching emitted step meta json.

    The offline branch json stores only token ids per region; the emitted step
    json keeps the simulator geometry (front_p/handle_p) for the same frame.
    """
    path = root / origin / f"seed_{seed:03d}" / f"step_{step:04d}.json"
    try:
        meta = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    out = {}
    for d in DRAWERS:
        reg = meta.get("regions", {}).get(d) or {}
        hp = reg.get("handle_p")
        if isinstance(hp, (list, tuple)) and len(hp) == 3:
            out[d] = np.asarray(hp, dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", type=Path, required=True, help="dir with offline_branches_*.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshots", type=Path, default=None)
    ap.add_argument("--emitted", type=Path, default=None,
                    help="dir with emitted_states/{top,middle}/seed_*/step_*.json for handle geometry")
    ap.add_argument("--gpu", type=int, default=3)
    args = ap.parse_args()
    emitted_root = (args.emitted or offline_dir.parent / "emitted_states").resolve()
    offline_dir = args.offline.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for d in DRAWERS:
        data[d] = load(offline_dir / f"offline_branches_{d}.json")["records"]
    frames = data["top"] + data["middle"]
    print(f"loaded frames: top={len(data['top'])} middle={len(data['middle'])} total={len(frames)}")

    # ---------- effect-size analysis over all frames ----------
    rows = []  # one row per (frame, instruction), with target & other branch effect
    for fr in frames:
        for instr_d in DRAWERS:
            t = fr["branches"][f"{instr_d}/target_region"]
            o = fr["branches"][f"{instr_d}/other_region"]
            et, _ = eff(t)
            eo, _ = eff(o)
            rows.append({
                "seed": fr["seed"], "control_step": fr["control_step"],
                "rgb": fr["rgb_sha256"], "instruction": instr_d,
                "effect_target": et, "effect_other": eo,
                "perturb_target": float(t["feature_perturbation_norm"]),
                "perturb_other": float(o["feature_perturbation_norm"]),
                "region_size_target": int(t["region_size"]),
                "region_size_other": int(o["region_size"]),
            })
    diff = np.asarray([r["effect_target"] - r["effect_other"] for r in rows])
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    tstat = mean_diff / se if se else float("nan")
    all_effect = {
        "n_rows": len(rows),
        "mean_effect_target": float(np.mean([r["effect_target"] for r in rows])),
        "mean_effect_other": float(np.mean([r["effect_other"] for r in rows])),
        "mean_target_minus_other": mean_diff,
        "paired_se": se, "paired_t": tstat,
        "frac_target_larger": float(np.mean([r["effect_target"] > r["effect_other"] for r in rows])),
        "n_target_larger": int(np.sum([r["effect_target"] > r["effect_other"] for r in rows])),
        "mean_perturb_target": float(np.mean([r["perturb_target"] for r in rows])),
        "mean_perturb_other": float(np.mean([r["perturb_other"] for r in rows])),
        "mean_size_target": float(np.mean([r["region_size_target"] for r in rows])),
        "mean_size_other": float(np.mean([r["region_size_other"] for r in rows])),
    }

    # per-instruction breakdown
    per_instr = {}
    for instr_d in DRAWERS:
        sub = [r for r in rows if r["instruction"] == instr_d]
        d = np.asarray([r["effect_target"] - r["effect_other"] for r in sub])
        per_instr[instr_d] = {
            "n": len(sub),
            "mean_effect_target": float(np.mean([r["effect_target"] for r in sub])),
            "mean_effect_other": float(np.mean([r["effect_other"] for r in sub])),
            "mean_target_minus_other": float(d.mean()),
            "frac_target_larger": float(np.mean([r["effect_target"] > r["effect_other"] for r in sub])),
        }

    # ---------- same-mask instruction sensitivity (all frames) ----------
    # For a fixed frame+mask, compare guided effect under instr top vs middle.
    same_mask = {"top": [], "middle": []}
    for fr in frames:
        for mask_d in DRAWERS:
            et, _ = eff(fr["branches"][f"top/{'target_region' if mask_d == 'top' else 'other_region'}"])
            em, _ = eff(fr["branches"][f"middle/{'target_region' if mask_d == 'middle' else 'other_region'}"])
            same_mask[mask_d].append({"effect_instr_top": et, "effect_instr_middle": em,
                                      "abs_diff": abs(et - em)})
    same_mask_stats = {}
    for mask_d in DRAWERS:
        arr = same_mask[mask_d]
        diffs = np.asarray([a["abs_diff"] for a in arr])
        same_mask_stats[mask_d] = {
            "n": len(arr),
            "mean_abs_effect_difference_across_instructions": float(diffs.mean()),
            "median_abs_effect_difference": float(np.median(diffs)),
            "frac_changed_gt_0_01": float(np.mean(diffs > 0.01)),
        }

    # ---------- strict step-0 pairing + direction alignment ----------
    step0 = {}
    for fr in frames:
        if fr["control_step"] != 0:
            continue
        step0.setdefault(fr["rgb_sha256"], fr)  # keep one record per unique frame
    step0_frames = list(step0.values())
    print(f"strict step-0 unique frames: {len(step0_frames)}")

    # Handle positions for each step-0 frame come from the emitted state meta
    # (offline branch json stores only token ids).
    geo_cache = {}
    for fr in step0_frames:
        key = (fr["drawer_origin"], fr["seed"], fr["control_step"])
        geo_cache[key] = geometry_from_emitted(emitted_root, *key)
    n_geo = sum(1 for g in geo_cache.values() if len(g) == len(DRAWERS))
    print(f"step-0 frames with full handle geometry: {n_geo}/{len(geo_cache)}")

    eef_cache = {}
    if args.snapshots is not None and len(step0_frames):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        from research.semantic_token_cd.xdrawer_protocol import (
            load_snapshot, make_drawer_env, restore_snapshot,
        )
        env, _ = make_drawer_env("top", args.gpu)
        for fr in step0_frames:
            seed = fr["seed"]
            if seed not in eef_cache:
                snap = load_snapshot(args.snapshots.resolve(), "top", seed)
                obs = restore_snapshot(env, seed, snap)
                eef_cache[seed] = np.asarray(obs["agent"]["eef_pos"][:3], dtype=np.float64)
        env.close()
    print(f"eef cached seeds: {len(eef_cache)}")

    alignment = []
    missing_geo = 0
    for fr in step0_frames:
        eef = eef_cache.get(fr["seed"])
        handles = geo_cache.get((fr["drawer_origin"], fr["seed"], fr["control_step"]), {})
        for instr_d in DRAWERS:
            row = {"seed": fr["seed"], "instruction": instr_d}
            for branch_name in ("target_region", "other_region"):
                rec = fr["branches"][f"{instr_d}/{branch_name}"]
                _, delta = eff(rec)
                row[f"effect_{branch_name}_norm"] = float(np.linalg.norm(delta[:3]))
                if eef is not None and len(handles) == len(DRAWERS):
                    proj = {}
                    for d in DRAWERS:
                        vec = handles[d] - eef
                        n = np.linalg.norm(vec)
                        proj[d] = float(delta[:3] @ (vec / n)) if n > 1e-6 else 0.0
                    row[f"proj_target_{branch_name}"] = proj[instr_d]
                    row[f"proj_other_{branch_name}"] = proj["middle" if instr_d == "top" else "top"]
            if "proj_target_target_region" not in row:
                missing_geo += 1
            alignment.append(row)
    print(f"alignment rows: {len(alignment)} (rows lacking geo/eef: {missing_geo})")
    align_stats = {}
    if alignment:
        for instr_d in DRAWERS:
            sub = [a for a in alignment if a["instruction"] == instr_d]
            tgt_larger = [a for a in sub if a.get("proj_target_target_region", 0) > a.get("proj_other_target_region", 0)]
            align_stats[instr_d] = {
                "n": len(sub),
                "n_with_direction": int(np.sum([1 for a in sub if "proj_target_target_region" in a])),
                "mean_proj_target_target_branch": float(np.mean([a.get("proj_target_target_region", 0) for a in sub])),
                "mean_proj_other_target_branch": float(np.mean([a.get("proj_other_target_region", 0) for a in sub])),
                "mean_proj_target_other_branch": float(np.mean([a.get("proj_target_other_region", 0) for a in sub])),
                "mean_proj_other_other_branch": float(np.mean([a.get("proj_other_other_region", 0) for a in sub])),
                "frac_target_branch_aligned_to_target": float(len(tgt_larger) / max(1, len(sub))),
            }
    report = {
        "protocol_id": PROTOCOL,
        "n_total_frames": len(frames),
        "n_step0_unique": len(step0_frames),
        "all_frames_role_effect": all_effect,
        "per_instruction_role_effect": per_instr,
        "same_mask_instruction_sensitivity": same_mask_stats,
        "step0_direction_alignment": align_stats,
        "note": "effect = ||guided_action[0:6]-clean_action[0:6]||; gripper dim excluded. "
                "proj_* = SHR delta translation projected on direction from eef to drawer handle.",
    }
    (out / "offline_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

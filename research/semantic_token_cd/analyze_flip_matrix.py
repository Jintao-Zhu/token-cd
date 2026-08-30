"""Phase 1B per-seed flip matrix + first-divergence mechanism export.

Exports, for every task/seed across three arms (vanilla from Phase 1A,
semantic_attn_k8_l8_15 and semantic_merge_k8_eta100 from Phase 1B), the
per-seed rescue/harm classification and the first-divergence / state-regime
diagnostics needed to answer:

    Why does move_near need the semantic-token access path preserved, while
    close_drawer benefits from blocking it?

No env replay is needed: everything below is read from the already-written
summaries (selector_trace, result.first_*) and arrays (positive/negative_logits,
executed_actions). Spatial distances (gripper<->source, source<->target) are NOT
recorded and are left to a follow-up replay step.

Usage:
  <venv>/bin/python research/semantic_token_cd/analyze_flip_matrix.py \
    --artifact artifacts/attn_semantic_merge_k8_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)
ATTN_ARM = "semantic_attn_k8_l8_15"
MERGE_ARM = "semantic_merge_k8_eta100"
VANILLA_ARM = "vanilla"
SEED_START, SEED_END = 200, 299
ATOL = 1e-4  # action comparison tolerance (float32 continuous)


def _load_summaries(artifact: Path, task: str, arm: str) -> dict[int, dict]:
    d = {}
    adir = artifact / "episodes" / task / arm
    if not adir.exists():
        return d
    for f in adir.glob("episode_*_summary.json"):
        import re
        mm = re.search(r"episode_(\d+)_", f.name)
        if not mm:
            continue
        d[int(mm.group(1))] = json.loads(f.read_text())
    return d


def _load_arrays(artifact: Path, task: str, arm: str, seed: int) -> dict | None:
    p = artifact / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=False))


def _logsoftmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(e.sum(axis=-1, keepdims=True))


def _residual(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return (_logsoftmax(pos) - _logsoftmax(neg)).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def _first_divergence(a: np.ndarray, b: np.ndarray, atol: float = ATOL):
    """First timestep t where the 7-dim actions differ; None if identical over overlap."""
    T = min(a.shape[0], b.shape[0])
    for t in range(T):
        if not np.allclose(a[t], b[t], atol=atol, rtol=0.0):
            return t, (a[t] - b[t]).astype(np.float32)
    return None, None


def _token_ids(arr: np.ndarray, t: int) -> list[int]:
    return [int(x) for x in np.argmax(arr[t], axis=-1)]


def _exec_token(pos: np.ndarray, neg: np.ndarray | None, t: int, lambd: float = 0.5) -> list[int]:
    """CD arm's executed action token in ACTION space (0..255), not LLM vocab.

    The trace's final_token_ids are LLM vocabulary ids (action_idx + 31744); the
    [7,256] action logits are the action-token slice. Combine mirrors the policy:
    dims 0..5 = (1+lambda)*clean - lambda*negative, dim 6 (gripper) = clean.
    """
    if neg is None:
        return _token_ids(pos, t)
    comb = pos[t].astype(np.float64).copy()
    comb[:-1] = (1.0 + lambd) * comb[:-1] - lambd * neg[t].astype(np.float64)[:-1]
    return [int(x) for x in np.argmax(comb, axis=-1)]


def _outcome(van_succ: bool, arm_succ: bool) -> str:
    if van_succ and not arm_succ:
        return "harm"
    if not van_succ and arm_succ:
        return "rescue"
    if van_succ and arm_succ:
        return "same_succ"
    return "same_fail"


def _joint_class(a_out: str, m_out: str) -> str:
    if a_out == "harm" and m_out == "rescue":
        return "attn_harm_merge_rescue"
    if a_out == "rescue" and m_out == "harm":
        return "attn_rescue_merge_harm"
    if a_out == "rescue" and m_out == "rescue":
        return "both_rescue"
    if a_out == "harm" and m_out == "harm":
        return "both_harm"
    return "other"


def _phase_flags(result: dict) -> dict:
    """Extract first_* (timestep of event) + final_* + success + numeric qpos."""
    out = {}
    for k, v in result.items():
        if k.startswith("first_") or k in ("success", "qpos"):
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ref_artifact", type=Path,
                        default=Path("artifacts/attn_global_merge_v1"))
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/attn_semantic_merge_k8_v1/flip_matrix"))
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    ref = args.ref_artifact.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_range = range(SEED_START, SEED_END + 1)
    rows: list[dict] = []
    failures: list[str] = []

    for task in TASKS:
        vsum = _load_summaries(ref, task, VANILLA_ARM)
        asum = _load_summaries(artifact, task, ATTN_ARM)
        msum = _load_summaries(artifact, task, MERGE_ARM)
        paired = [s for s in seed_range if s in vsum and s in asum and s in msum]

        for s in paired:
            vs = vsum[s]; aa = asum[s]; mm = msum[s]
            # ---- non-determinism guard: hash match vs vanilla ----
            non_det = not (
                aa["canonical_snapshot_sha256"] == vs["canonical_snapshot_sha256"]
                and aa["initial_state_sha256"] == vs["initial_state_sha256"]
                and aa["initial_rgb_sha256"] == vs["initial_rgb_sha256"]
                and mm["canonical_snapshot_sha256"] == vs["canonical_snapshot_sha256"]
            )
            if non_det:
                failures.append(f"{task} seed {s}")

            v_succ = bool(vs["success"]); a_succ = bool(aa["success"]); m_succ = bool(mm["success"])
            a_out = _outcome(v_succ, a_succ)
            m_out = _outcome(v_succ, m_succ)

            va = _load_arrays(ref, task, VANILLA_ARM, s)
            aa_arr = _load_arrays(artifact, task, ATTN_ARM, s)
            mm_arr = _load_arrays(artifact, task, MERGE_ARM, s)
            v_act = va["executed_actions"] if va is not None else None
            a_act = aa_arr["executed_actions"] if aa_arr is not None else None
            m_act = mm_arr["executed_actions"] if mm_arr is not None else None

            a_trace = aa.get("selector_trace", [])
            m_trace = mm.get("selector_trace", [])

            def _div_row(arm_act, van_act, van_arr, arm_arr, arm_trace):
                if arm_act is None or van_act is None:
                    return {"first_div_timestep": None, "note": "missing arrays"}
                t, delta = _first_divergence(arm_act, van_act)
                if t is None:
                    return {"first_div_timestep": None,
                            "episode_len_diff": int(arm_act.shape[0] - van_act.shape[0]),
                            "identical_over_overlap": True}
                dims = [int(i) for i in np.flatnonzero(~np.isclose(delta, 0.0, atol=ATOL))]
                v_tok = _token_ids(van_arr["positive_logits"], t) if van_arr is not None else None
                a_tok = (_exec_token(arm_arr["positive_logits"], arm_arr.get("negative_logits"), t)
                         if arm_arr is not None else None)
                flipped = ([i for i in range(7) if v_tok[i] != a_tok[i]]
                           if v_tok is not None and a_tok is not None else None)
                return {
                    "first_div_timestep": t,
                    "div_action_dims": dims,
                    "div_continuous_delta": [float(x) for x in delta],
                    "vanilla_token_ids_at_div": v_tok,
                    "arm_token_ids_at_div": a_tok,
                    "flipped_token_indices": flipped,
                    "residual_norm_at_div": (
                        float(arm_trace[t]["residual_norm"]) if t < len(arm_trace) else None
                    ),
                }

            a_vs_v = _div_row(a_act, v_act, va, aa_arr, a_trace)
            m_vs_v = _div_row(m_act, v_act, va, mm_arr, m_trace)
            # merge vs attn (same-process head-to-head)
            if a_act is not None and m_act is not None:
                t, delta = _first_divergence(m_act, a_act)
                ma_vs_a = {
                    "first_div_timestep": t,
                    "div_action_dims": (
                        [int(i) for i in np.flatnonzero(~np.isclose(delta, 0.0, atol=ATOL))]
                        if t is not None else []
                    ),
                }
            else:
                ma_vs_a = {"first_div_timestep": None}

            # cos(r_attn, r_merge) at step 0 (shared initial state, clean attribution)
            cos_step0 = None
            if (aa_arr is not None and mm_arr is not None
                    and "negative_logits" in aa_arr and "negative_logits" in mm_arr
                    and aa_arr["positive_logits"].shape[0] > 0
                    and mm_arr["positive_logits"].shape[0] > 0):
                r_a = _residual(aa_arr["positive_logits"][0], aa_arr["negative_logits"][0])
                r_m = _residual(mm_arr["positive_logits"][0], mm_arr["negative_logits"][0])
                cos_step0 = _cos(r_a, r_m)

            # selector coverage (step 0)
            sel = {}
            if m_trace:
                s0 = m_trace[0]
                sel = {
                    "step0_selected_group_ids": s0.get("selected_group_ids"),
                    "step0_group_sizes": s0.get("selected_group_sizes"),
                    "step0_num_tokens": s0.get("num_tokens"),
                    "step0_num_groups": s0.get("num_groups"),
                    "step0_per_entity_score": s0.get("per_entity_score"),
                    "step0_selected_entities": s0.get("selected_entities"),
                    "mean_num_tokens": float(mm.get("mean_num_tokens", 0.0)),
                    "mean_num_groups": float(mm.get("mean_num_groups", 0.0)),
                }

            rows.append({
                "task": task, "seed": s,
                "non_deterministic": non_det,
                "success": {"vanilla": v_succ, ATTN_ARM: a_succ, MERGE_ARM: m_succ},
                "steps": {"vanilla": int(vs.get("control_steps", 0)),
                          ATTN_ARM: int(aa.get("control_steps", 0)),
                          MERGE_ARM: int(mm.get("control_steps", 0))},
                "attn_outcome": a_out, "merge_outcome": m_out,
                "joint_class": _joint_class(a_out, m_out) if not non_det else "non_deterministic",
                "attn_vs_vanilla": a_vs_v,
                "merge_vs_vanilla": m_vs_v,
                "merge_vs_attn": ma_vs_a,
                "cos_r_attn_r_merge_step0": cos_step0,
                "selector": sel,
                "vanilla_phase_flags": _phase_flags(vs.get("result", {})),
                "vanilla_failure_reason": vs.get("failure_reason"),
                "attn_failure_reason": aa.get("failure_reason"),
                "merge_failure_reason": mm.get("failure_reason"),
            })

    # ---- write JSON + CSV ----
    (out_dir / "flip_matrix.json").write_text(
        json.dumps({"rows": rows, "non_deterministic_seeds": failures}, indent=2, sort_keys=True) + "\n")

    csv_cols = [
        "task", "seed", "non_deterministic", "joint_class",
        "v_succ", "a_succ", "m_succ", "v_steps", "a_steps", "m_steps",
        "a_div_t", "a_div_dims", "a_flipped_tokens", "a_res_norm_div",
        "m_div_t", "m_div_dims", "m_flipped_tokens", "m_res_norm_div",
        "ma_div_t", "cos_step0",
        "sel_groups", "sel_sizes", "sel_num_tokens", "sel_per_entity",
        "v_first_success", "v_first_moved_correct_obj", "v_first_near_tgt_obj",
        "v_first_is_grasped", "v_first_lifted_object", "v_qpos",
    ]
    with (out_dir / "flip_matrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(csv_cols)
        for r in rows:
            pf = r["vanilla_phase_flags"]
            w.writerow([
                r["task"], r["seed"], r["non_deterministic"], r["joint_class"],
                int(r["success"]["vanilla"]), int(r["success"][ATTN_ARM]), int(r["success"][MERGE_ARM]),
                r["steps"]["vanilla"], r["steps"][ATTN_ARM], r["steps"][MERGE_ARM],
                r["attn_vs_vanilla"].get("first_div_timestep"),
                r["attn_vs_vanilla"].get("div_action_dims"),
                r["attn_vs_vanilla"].get("flipped_token_indices"),
                r["attn_vs_vanilla"].get("residual_norm_at_div"),
                r["merge_vs_vanilla"].get("first_div_timestep"),
                r["merge_vs_vanilla"].get("div_action_dims"),
                r["merge_vs_vanilla"].get("flipped_token_indices"),
                r["merge_vs_vanilla"].get("residual_norm_at_div"),
                r["merge_vs_attn"].get("first_div_timestep"),
                r["cos_r_attn_r_merge_step0"],
                r["selector"].get("step0_selected_group_ids"),
                r["selector"].get("step0_group_sizes"),
                r["selector"].get("step0_num_tokens"),
                r["selector"].get("step0_per_entity_score"),
                pf.get("first_success"), pf.get("first_moved_correct_obj"),
                pf.get("first_near_tgt_obj"), pf.get("first_is_grasped"),
                pf.get("first_lifted_object"), pf.get("qpos"),
            ])

    # ---- summary: 4-class counts per task ----
    print("=== 4-class joint outcome counts per task (excl. non-deterministic) ===")
    classes = ["attn_harm_merge_rescue", "attn_rescue_merge_harm",
               "both_rescue", "both_harm", "other"]
    for task in TASKS:
        tr = [r for r in rows if r["task"] == task and not r["non_deterministic"]]
        cnt = {c: sum(1 for r in tr if r["joint_class"] == c) for c in classes}
        print(f"\n{task}  (n={len(tr)})")
        for c in classes:
            print(f"  {c:26s}: {cnt[c]}")
    print(f"\nnon_deterministic_seeds: {failures}")
    print(f"wrote {out_dir / 'flip_matrix.json'} and {out_dir / 'flip_matrix.csv'}")


if __name__ == "__main__":
    main()

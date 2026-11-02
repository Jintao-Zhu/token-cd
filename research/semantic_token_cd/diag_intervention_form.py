"""Offline diagnosis: what kind of perturbation does the intervention apply to
the *applied* action, in discrete-argmax models versus continuous flow models?

Background
----------
`(1+lambda)*positive - lambda*negative` was validated on models whose actions are
discretised tokens decoded by argmax (OpenVLA, SpatialVLA, pi0-FAST).  On
flow-matching models (pi0, and by construction pi0.5) the same recipe perturbs
the continuous velocity field at every Euler step.  This script measures how the
two perturbations differ using only stored artifacts.  No rollouts, no GPU.

Both sides are reported in unit-free ratios of the displacement to (a) the norm
of the clean action and (b) the natural spread of the action across decisions,
so no cross-model action normalisation is assumed.

Conventions that were verified rather than assumed
--------------------------------------------------
* OpenVLA action bins.  `prompt_attn_shr_policy.py` slices the last 256 vocab
  entries (`start = vocab_size - 256`), and `decode_action_ids` maps
  `discretized = vocab_size - token_id` through `bin_centers = linspace(-1, 1, 256)`.
  Stored position `j` therefore corresponds to `a_norm = 1 - 2*j/255`, and one
  flipped bin moves a dimension by `2/255`.
  The reconstruction is validated against each summary's
  `mean_guided_change_ratio`, which it must reproduce exactly.
* pi0 guidance.  `guided = positive + 0.5*(positive - negative)` on the first six
  dimensions only; the gripper dimension is untouched.  The gripper must be
  excluded from *both* numerator and denominator, otherwise its large contrast
  but zero displacement biases every ratio.

Outputs: artifacts/intervention_form_diagnosis_v1/{discrete,flow}_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
from collections import defaultdict

import numpy as np

REPO = pathlib.Path("/home/leju-suzhou/zjt_ws/token-cd")
OUT = REPO / "artifacts" / "intervention_form_diagnosis_v1"

DISCRETE_ROOT = REPO / "artifacts" / "prompt_attn_l11_matched_full_9task_0_299_v1" / "run" / "episodes"
FLOW_ROOT = REPO / "artifacts" / "pi0_shr_5task_0_299_v1" / "episodes"

GUIDED_DIMS = slice(0, 6)   # dim 6 (gripper) is never guided
N_BINS = 256
FLOW_STEPS = 10
EXEC_STEPS = 2              # pi0 open-pi-zero config executes 2 actions per replan


def _norm(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm of a 2-D array."""
    return np.sqrt((x.astype(np.float64) ** 2).sum(axis=-1))


# --------------------------------------------------------------------------- #
def discrete_episode(npz_path: pathlib.Path, lam: float) -> dict | None:
    try:
        z = np.load(npz_path)
    except Exception:
        return None
    if not {"positive", "negative"} <= set(z.files):
        return None
    pos = z["positive"].astype(np.float64)
    neg = z["negative"].astype(np.float64)
    if pos.ndim != 3 or pos.shape[-1] != N_BINS or pos.shape[1] < 7:
        return None

    final = pos.copy()
    final[:, GUIDED_DIMS, :] = (1.0 + lam) * pos[:, GUIDED_DIMS, :] - lam * neg[:, GUIDED_DIMS, :]

    clean_pos = pos[:, GUIDED_DIMS, :].argmax(axis=-1)      # (T, 6)
    final_pos = final[:, GUIDED_DIMS, :].argmax(axis=-1)    # (T, 6)

    a_clean = 1.0 - 2.0 * clean_pos / (N_BINS - 1)
    a_guided = 1.0 - 2.0 * final_pos / (N_BINS - 1)
    delta = a_guided - a_clean
    flip = final_pos != clean_pos

    return {
        "n_steps": int(flip.shape[0]),
        "n_cells": int(flip.size),
        "n_flip_cells": int(flip.sum()),
        "steps_with_any_change": int((flip.sum(axis=1) > 0).sum()),
        "changed_dims_sum": int(flip.sum()),
        "delta_norm_sum": float(_norm(delta).sum()),
        "clean_norm_sum": float(_norm(a_clean).sum()),
        "delta_norm_mean": float(_norm(delta).mean()),
        "spread_per_dim": float(a_clean.std(axis=0).mean()),
    }


def flow_episode(npz_path: pathlib.Path) -> dict | None:
    try:
        z = np.load(npz_path)
    except Exception:
        return None
    need = {"positive_velocity", "guided_velocity", "initial_noise"}
    if not need <= set(z.files):
        return None
    pos = z["positive_velocity"].astype(np.float64)[..., GUIDED_DIMS]
    gui = z["guided_velocity"].astype(np.float64)[..., GUIDED_DIMS]
    noise = z["initial_noise"].astype(np.float64)[..., GUIDED_DIMS]
    if pos.ndim != 4 or pos.shape[0] % FLOW_STEPS:
        return None

    total = pos.shape[0]
    n_replan = total // FLOW_STEPS
    if noise.shape[0] != n_replan:
        return None
    dt = -1.0 / FLOW_STEPS
    per_step = pos.shape[2]                      # actions per chunk

    v_pos = pos.reshape(n_replan, FLOW_STEPS, -1)
    v_gui = gui.reshape(n_replan, FLOW_STEPS, -1)

    # first-order chunk displacement: the injected velocity summed over the flow
    # steps with the Euler weight.
    delta = dt * (v_gui - v_pos).sum(axis=1)                       # (n_replan, per_step*6)
    x_clean = noise.reshape(n_replan, -1) + dt * v_pos.sum(axis=1)  # (n_replan, per_step*6)

    keep = slice(0, min(EXEC_STEPS, per_step))
    d = delta.reshape(n_replan, per_step, -1)[:, keep, :].reshape(n_replan, -1)
    x = x_clean.reshape(n_replan, per_step, -1)[:, keep, :].reshape(n_replan, -1)

    return {
        "n_replans": int(n_replan),
        "delta_norm_sum": float(_norm(d).sum()),
        "clean_norm_sum": float(_norm(x).sum()),
        "delta_norm_mean": float(_norm(d).mean()),
        "spread_per_dim": float(x.std(axis=0).mean()),
        # L2 norm of the injected velocity summed over the ten flow steps of one
        # replan (not per step) -- this is the raw quantity whose Euler-weighted
        # version is the action displacement reported above.
        "inj_velocity_norm_per_replan": float(_norm((v_gui - v_pos).reshape(n_replan, -1)).mean()),
        "inj_velocity_norm_per_step": float(
            _norm((v_gui - v_pos).reshape(n_replan * FLOW_STEPS, -1)).mean()
        ),
    }


# --------------------------------------------------------------------------- #
def collect_discrete(limit: int | None) -> dict:
    per_task, pooled = {}, defaultdict(float)
    n_cells = n_flip = n_steps = steps_any = changed = 0
    validation = []
    if not DISCRETE_ROOT.is_dir():
        return {"error": f"missing {DISCRETE_ROOT}"}

    for task_dir in sorted(DISCRETE_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        t = defaultdict(float)
        n = 0
        for arm_dir in sorted(task_dir.iterdir()):
            if not arm_dir.is_dir():
                continue
            for npz in sorted(arm_dir.glob("*_arrays.npz")):
                if limit is not None and n >= limit:
                    break
                summary = npz.with_name(npz.name.replace("_arrays.npz", "_summary.json"))
                lam = 0.5
                reported = None
                if summary.exists():
                    try:
                        meta = json.loads(summary.read_text())
                        lam = float(meta.get("lambda", 0.5))
                        reported = meta.get("mean_guided_change_ratio")
                    except Exception:
                        pass
                rec = discrete_episode(npz, lam)
                if rec is None:
                    continue
                if reported is not None and len(validation) < 200:
                    validation.append({
                        "file": npz.name,
                        "recomputed": rec["n_flip_cells"] / rec["n_cells"],
                        "reported": float(reported),
                    })
                n += 1
                t["n_cells"] += rec["n_cells"]; t["n_flip_cells"] += rec["n_flip_cells"]
                t["n_steps"] += rec["n_steps"]; t["steps_with_any_change"] += rec["steps_with_any_change"]
                t["changed_dims_sum"] += rec["changed_dims_sum"]
                t["delta_norm_sum"] += rec["delta_norm_sum"]; t["clean_norm_sum"] += rec["clean_norm_sum"]
                t["spread"] += rec["spread_per_dim"]
            if limit is not None and n >= limit:
                break
        if not n:
            continue
        spread = t["spread"] / n
        per_task[task_dir.name] = {
            "episodes": n,
            "decision_no_change_rate": 1.0 - t["steps_with_any_change"] / t["n_steps"],
            "cell_flip_rate": t["n_flip_cells"] / t["n_cells"],
            "changed_dims_per_decision": t["changed_dims_sum"] / t["n_steps"],
            "rel_disp_vs_action_norm": t["delta_norm_sum"] / t["clean_norm_sum"],
            "rel_disp_vs_spread": (t["delta_norm_sum"] / t["n_steps"]) / (spread * np.sqrt(6)),
        }
        for k, v in t.items():
            pooled[k] += v
        pooled["episodes"] += n
        pooled["spread"] += spread
        n_cells += t["n_cells"]; n_flip += t["n_flip_cells"]; n_steps += t["n_steps"]
        steps_any += t["steps_with_any_change"]; changed += t["changed_dims_sum"]

    n_ep = int(pooled["episodes"])
    spread = pooled["spread"] / max(n_ep, 1)
    return {
        "source": str(DISCRETE_ROOT),
        "episodes": n_ep,
        "guidance": "(1+lambda)*positive - lambda*negative on dims 0..5, decoded by argmax",
        "pooled": {
            "decisions": n_steps,
            "decision_no_change_rate": 1.0 - steps_any / n_steps,
            "cell_flip_rate": n_flip / n_cells,
            "changed_dims_per_decision": changed / n_steps,
            "rel_disp_vs_action_norm": pooled["delta_norm_sum"] / pooled["clean_norm_sum"],
            "rel_disp_vs_spread": (pooled["delta_norm_sum"] / n_steps) / (spread * np.sqrt(6)),
        },
        "reconstruction_validation": validation,
        "per_task": per_task,
    }


def collect_flow(limit: int | None) -> dict:
    per_task = {}
    tot = defaultdict(float)
    n_ep = 0
    if not FLOW_ROOT.is_dir():
        return {"error": f"missing {FLOW_ROOT}"}

    for task_dir in sorted(FLOW_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        arm = task_dir / "pi0_shr_harmonic"
        if not arm.is_dir():
            continue
        t = defaultdict(float)
        recs = []
        for npz in sorted(arm.glob("*_arrays.npz")):
            if limit is not None and len(recs) >= limit:
                break
            rec = flow_episode(npz)
            if rec is None:
                continue
            recs.append(rec)
            t["delta_norm_sum"] += rec["delta_norm_sum"]; t["clean_norm_sum"] += rec["clean_norm_sum"]
            t["spread"] += rec["spread_per_dim"]
            t["inj"] += rec["inj_velocity_norm_per_step"]
            t["n_replans"] += rec["n_replans"]
        if not recs:
            continue
        spread = t["spread"] / len(recs)
        per_task[task_dir.name] = {
            "episodes": len(recs),
            "decisions": int(t["n_replans"]),
            "decision_no_change_rate": 0.0,
            "rel_disp_vs_action_norm": t["delta_norm_sum"] / t["clean_norm_sum"],
            "rel_disp_vs_spread": (t["delta_norm_sum"] / t["n_replans"]) / (spread * np.sqrt(2 * 6)),
            "injected_velocity_norm_per_step": t["inj"] / len(recs),
        }
        for k, v in t.items():
            tot[k] += v
        tot["spread"] += spread
        n_ep += len(recs)

    if not n_ep:
        return {"error": "no flow episodes parsed"}
    spread = tot["spread"] / n_ep
    return {
        "source": str(FLOW_ROOT),
        "episodes": n_ep,
        "guidance": "v_guided = v_positive + 0.5*(v_positive - v_negative) at every flow step, dims 0..5",
        "pooled": {
            "decisions": int(tot["n_replans"]),
            "decision_no_change_rate": 0.0,
            "rel_disp_vs_action_norm": tot["delta_norm_sum"] / tot["clean_norm_sum"],
            "rel_disp_vs_spread": (tot["delta_norm_sum"] / tot["n_replans"]) / (spread * np.sqrt(2 * 6)),
            "injected_velocity_norm_per_step": tot["inj"] / n_ep,
        },
        "per_task": per_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    disc = collect_discrete(args.limit)
    flow = collect_flow(args.limit)
    (OUT / "discrete_metrics.json").write_text(json.dumps(disc, indent=2))
    (OUT / "flow_metrics.json").write_text(json.dumps(flow, indent=2))

    v = disc.get("reconstruction_validation", [])
    if v:
        err = max(abs(x["recomputed"] - x["reported"]) for x in v)
        print(f"reconstruction check vs summary mean_guided_change_ratio: n={len(v)} max|err|={err:.3e}")

    d, f = disc.get("pooled", {}), flow.get("pooled", {})
    print("\n=== DISCRETE (OpenVLA, argmax action tokens) ===")
    print(f"  episodes {disc.get('episodes')}   decisions {d.get('decisions')}")
    print(f"  decisions with NO change       {d.get('decision_no_change_rate')}")
    print(f"  cell flip rate                 {d.get('cell_flip_rate')}")
    print(f"  changed dims per decision      {d.get('changed_dims_per_decision')} (of 6)")
    print(f"  displacement / action norm     {d.get('rel_disp_vs_action_norm')}")
    print(f"  displacement / natural spread  {d.get('rel_disp_vs_spread')}")
    print("\n=== CONTINUOUS (pi0, flow matching) ===")
    print(f"  episodes {flow.get('episodes')}   decisions {f.get('decisions')}")
    print(f"  decisions with NO change       {f.get('decision_no_change_rate')}")
    print(f"  displacement / action norm     {f.get('rel_disp_vs_action_norm')}")
    print(f"  displacement / natural spread  {f.get('rel_disp_vs_spread')}")
    print(f"  injected velocity norm / step  {f.get('injected_velocity_norm_per_step')}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

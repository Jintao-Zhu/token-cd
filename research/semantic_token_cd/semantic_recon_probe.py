"""SCR-CD Phase 0 integrity probe — 9 tasks x 10 frozen states = 90 states.

Fast (NO rollout, no guided forward, no CD): for each frozen snapshot we run the
EXACT Phase-1 KMeans K=8 semantic selector on the clean projector output ``h``,
then compute the context-reconstruction operator via ``reconstruct_groups(..., return_raw=True)``
and verify the three things the spec asks for:

  * G ∩ B = ∅ (the FPS-selected basis B never lies inside the semantic region G) — true
    by construction, re-checked here defensively.
  * no NaN/Inf anywhere (reconstruction + per-token diagnostics all finite).
  * the operator is NOT degenerate: v_hat is neither ~ v (identity; e_rel ~ 0) nor ~ 0
    (zero-map; q_ratio ~ 0). We deliberately do NOT prescribe "error large = good";
    we only fence off the two collapse modes.

For each (task, frozen state) we record the raw per-token float64 arrays
``e_rel_i = ||v_i - v_hat_i|| / ||v_i||``, ``q_i = ||v_hat_i|| / ||v_i||``,
``cos_i = <v_i, v_hat_i> / (||v_i|| ||v_hat_i||)`` and report mean/median/P10/P90
pooled over the 10 states per task (plus a cross-task aggregate).

The probe reuses ``build_policies`` from the Phase-1 rollout driver so the selector
(KMeans K=8, seed 0, n_init 10, entity_set top-1 cos) and the policy construction
are bit-identical to what Phase 1 will actually run. The frozen states are the
FIRST 10 Phase-1 seeds (300..309), captured with the same capture/restore logic,
so their canonical snapshots match the Phase-1 rollout's seeds 300..309 exactly.

Run (one process per task, fan out across the 6 free GPUs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/semantic_recon_probe.py \
      --artifact artifacts/semantic_recon_k8_m10_v1 --task google_robot_move_near --gpu 1

or all 9 tasks in one process:  --task all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_ROOT = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e")
PCD_SOURCE = PCD_ROOT / "source"

for _p in (str(REPO_ROOT / "task1/shim_site"), str(REPO_ROOT), str(PCD_SOURCE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.semantic_token_cd.distractor_rollout import (  # noqa: E402
    PCD_SOURCE,
    capture_snapshot,
    get_image_from_maniskill2_obs_dict,
    jsonable,
    restore_snapshot,
)
from research.semantic_token_cd.semantic_recon_policy import (  # noqa: E402
    M_BASIS,
    RHO_SCALE,
    _pstats,
    reconstruct_groups,
)
from research.semantic_token_cd.semantic_recon_rollout import (  # noqa: E402
    TASKS,
    build_policies,
)
from research.semantic_token_cd.spatial_grid_rollout import make_environment  # noqa: E402
from research.ar_token_counterfactual.intervention import projector_intervention  # noqa: E402

PROBE_SEEDS = list(range(300, 310))  # first 10 Phase-1 seeds -> identical snapshots


def _pooled_states(states: list[dict]) -> dict:
    """Concatenate per-state raw arrays and report mean/median/P10/P90 per key."""
    pooled: dict[str, dict] = {}
    for key in ("e_rel", "q_ratio", "cos"):
        pooled[key] = _pstats(np.concatenate([s[key] for s in states]))
    return pooled


def probe_task(task: str, gpu: int, artifact: Path, seeds: list[int]) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    env, environment_id = make_environment(task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, task, {}, False)
    # Reuse the EXACT Phase-1 policy construction; take only the semantic-recon arm.
    recon = build_policies(OpenVLAInference(**policy_config), task)["semantic_recon_k8_m10"]

    states: list[dict] = []
    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        instruction = env.unwrapped.get_language_instruction()
        recon.reset(instruction, seed=seed)

        inputs = recon.process_inputs(image, task_description=instruction)
        with projector_intervention(recon.vla) as trace:
            _clean_scores = recon._forward_scores(inputs, recon.unnorm_key, do_sample=False)
        h = trace.before[0].detach().float().cpu().numpy()  # [256, d]

        labels, semantic_groups, _per_entity_score = recon._semantic_clusters(h)
        h_tilde, diag, raw = reconstruct_groups(
            h, labels, sorted(int(g) for g in semantic_groups),
            M_BASIS, RHO_SCALE, return_raw=True,
        )

        g_idx = set(int(i) for g in semantic_groups for i in np.flatnonzero(labels == g))
        basis = set(int(b) for b in diag["recon_basis_token_ids"])
        disjoint = g_idx.isdisjoint(basis)
        finite = bool(
            diag["recon_finite"]
            and np.isfinite(raw["e_rel"]).all()
            and np.isfinite(raw["q_ratio"]).all()
            and np.isfinite(raw["cos"]).all()
        )

        states.append({
            "seed": seed,
            "state_sha256": state_sha,
            "rgb_sha256": rgb_sha,
            "finite": finite,
            "g_basis_disjoint": disjoint,
            "n_g_tokens": diag["recon_n_g_tokens"],
            "n_c_tokens": diag["recon_n_c_tokens"],
            "M": diag["recon_M"],
            "rho": diag["recon_rho"],
            "e_rel": raw["e_rel"],
            "q_ratio": raw["q_ratio"],
            "cos": raw["cos"],
        })
        print(json.dumps({
            "task": task, "seed": seed, "finite": finite, "disjoint": disjoint,
            "n_g": diag["recon_n_g_tokens"], "rho": round(diag["recon_rho"], 6),
        }, sort_keys=True), flush=True)

    pooled = _pooled_states(states)
    all_finite = all(s["finite"] for s in states)
    all_disjoint = all(s["g_basis_disjoint"] for s in states)
    # Fence off the two collapse modes only; do NOT gate on "error is large".
    median_e_rel = pooled["e_rel"]["median"]
    median_q = pooled["q_ratio"]["median"]
    identity_degenerate = median_e_rel < 1e-3          # v_hat ~ v (operator ~ identity)
    zero_degenerate = median_q < 1e-2                  # v_hat ~ 0 (operator ~ zero-map)
    pass_ = bool(all_finite and all_disjoint and not identity_degenerate and not zero_degenerate)

    report = {
        "task": task,
        "environment_id": environment_id,
        "seeds": seeds,
        "n_states": len(states),
        "finite_all": all_finite,
        "g_basis_disjoint_all": all_disjoint,
        "median_e_rel": median_e_rel,
        "median_q_ratio": median_q,
        "identity_degenerate": identity_degenerate,
        "zero_degenerate": zero_degenerate,
        "pass": pass_,
        "pooled": pooled,
        "states": [{k: v for k, v in s.items() if k not in ("e_rel", "q_ratio", "cos")} for s in states],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="300-309")
    args = parser.parse_args()

    def parse_seeds(spec: str) -> list[int]:
        seeds: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                seeds.extend(range(int(lo), int(hi) + 1))
            else:
                seeds.append(int(part))
        return seeds

    artifact = args.artifact.resolve()
    probe_dir = artifact / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    if args.task not in TASKS and args.task != "all":
        raise SystemExit(f"Unknown task: {args.task}")

    reports: dict[str, dict] = {}
    for task in tasks:
        report = probe_task(task, args.gpu, artifact, parse_seeds(args.seeds))
        reports[task] = report
        (probe_dir / f"{task}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.task == "all":
        agg = {
            "n_tasks": len(tasks),
            "seeds": parse_seeds(args.seeds),
            "all_pass": bool(all(r["pass"] for r in reports.values())),
            "any_identity_degenerate": any(r["identity_degenerate"] for r in reports.values()),
            "any_zero_degenerate": any(r["zero_degenerate"] for r in reports.values()),
            "per_task": {
                t: {k: reports[t][k] for k in (
                    "finite_all", "g_basis_disjoint_all", "median_e_rel",
                    "median_q_ratio", "identity_degenerate", "zero_degenerate", "pass")}
                for t in reports
            },
            "pooled": {t: reports[t]["pooled"] for t in reports},
        }
        (probe_dir / "AGGREGATE.json").write_text(
            json.dumps(agg, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"PROBE_DONE": True, **agg["per_task"]}, sort_keys=True), flush=True)
    else:
        r = reports[args.task]
        print(json.dumps({
            "PROBE_DONE": True, "task": args.task, "pass": r["pass"],
            "median_e_rel": r["median_e_rel"], "median_q_ratio": r["median_q_ratio"],
            "identity_degenerate": r["identity_degenerate"],
            "zero_degenerate": r["zero_degenerate"],
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
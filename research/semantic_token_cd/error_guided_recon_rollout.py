"""Error-Guided Recon CD — 4-arm, 3-task paired rollout.

Phase-2 of the SCR-CD program. Phase 1 showed semantic_recon_k8_m10 (FULL
replacement: v_i -> hat{v}_i) wins (+4.22pp, p=0.0005). The open question the
spec asks is *why*: is the gain from the semantic selector choosing WHAT to
touch, or could a different STRENGTH assignment on the SAME selected tokens do
better / explain the effect? The ApET reconstruction-error idea predicts the
tokens the context basis cannot explain (large e_i = ||v_i - hat{v}_i||/||v_i||)
are the region-unique signal, so they should be weakened HARD, while well-
explained tokens (small e_i) should be left mostly intact:

    Arm A  error_guided_recon_k8_m10   v_i^- = (1 - s_i) v_i + s_i hat{v}_i,
                                       s_i = clip(e_i, 0, 1)
    Arm B  matched_strength_recon_k8_m10  v_i^- = (1 - s_bar) v_i + s_bar hat{v}_i,
                                       s_bar = mean_i clip(e_i, 0, 1)

Arm B is the decisive control: it applies the SAME total weakening strength as
Arm A but UNIFORMLY (no per-token selectivity), so
    error_guided > matched_strength  => the *selectivity* (by error) helps;
    error_guided ~ matched_strength  => only the overall lighter touch matters.

References (paired on the same canonical snapshot):
    vanilla               — clean OpenVLA (no CD).
    semantic_recon_k8_m10 — Phase-1 FULL replacement (s_i = 1), re-run HERE so
                            all four arms share the exact same snapshot pairing.

Frozen config identical to Phase 1: KMeans K=8 semantic selector (G = source ∪
target), M=10 FPS context basis, ridge rho = 1e-3 tr(B_c^T B_c)/M, CD
z* = z+ + 0.5(z+ - z-) on action dims 0..5, dim 6 (gripper) clean, greedy, shared
guided prefix. Reconstruction computed once per control observation.

Seeds 300-399 (100/task). Vanilla re-run HERE (not reused) so all four arms
restore the exact same canonical snapshot, hash-verified fail-closed.

Run (per task; seed-shard across processes/GPUs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/error_guided_recon_rollout.py \
      --artifact artifacts/error_guided_recon_v1 \
      --task google_robot_move_near --gpu 1 --seeds 300-319
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.semantic_recon_policy import (
    M_BASIS,
    RHO_SCALE,
    SemanticReconCDInference,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)

PROTOCOL = "SCR_CD_ERROR_GUIDED_RECON_V1"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_close_drawer",
    "google_robot_move_near",
)
TASK_INDEX = {t: i for i, t in enumerate(TASKS)}
ARMS = (
    "vanilla",
    "semantic_recon_k8_m10",
    "error_guided_recon_k8_m10",
    "matched_strength_recon_k8_m10",
)
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0
SEED_START = 300
N_SEEDS = 100

# strength_mode per recon arm (the ONLY difference between the three recon arms).
STRENGTH_MODE = {
    "semantic_recon_k8_m10": "full",
    "error_guided_recon_k8_m10": "error_guided",
    "matched_strength_recon_k8_m10": "matched_strength",
}
RECON_ARMS = tuple(STRENGTH_MODE)  # every non-vanilla arm here is a recon arm


def parse_seeds(spec: str) -> list[int]:
    if not spec.strip():
        return list(range(SEED_START, SEED_START + N_SEEDS))
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


def _init_common(policy, lambd: float) -> None:
    policy.alpha = lambd
    policy.lambd = lambd
    policy.selection_mode = "semantic"
    policy.kmeans_K = KMEANS_K
    policy.kmeans_seed = KMEANS_SEED
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0


def build_policies(base, task: str):
    policies: dict[str, object] = {}

    vanilla = copy.copy(base)
    vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace = []
    vanilla._episode_logits = []
    policies["vanilla"] = vanilla

    for arm in RECON_ARMS:
        pol = copy.copy(base)
        pol.__class__ = SemanticReconCDInference
        _init_common(pol, LAMBDA)
        pol.recon_selection_mode = "semantic"
        pol.selection_mode = "semantic"
        pol.strength_mode = STRENGTH_MODE[arm]
        pol.M = M_BASIS
        pol.rho_scale = RHO_SCALE
        pol._task_id = TASK_INDEX[task]
        policies[arm] = pol

    return policies


def audit_recon_trace(trace: list[dict], strength_mode: str) -> dict:
    if not trace:
        raise RuntimeError(f"{strength_mode} arm produced no trace")
    feature_equal = all(s.get("feature_equal", False) for s in trace)
    guided = all(s.get("guided_prefix", False) for s in trace)
    finite = all(np.isfinite(s.get("residual_norm", 0.0)) for s in trace)
    n_tokens_ok = all(s.get("n_tokens_negative", 0) == 256 for s in trace)
    m_ok = all(s.get("recon_M", -1) == M_BASIS for s in trace)
    disjoint_ok = all(s.get("recon_g_basis_disjoint", False) for s in trace)
    recon_finite = all(s.get("recon_finite", False) for s in trace)
    kmeans_ok = all(s.get("kmeans_K") == KMEANS_K for s in trace)
    sel_mode_ok = all(s.get("recon_selection_mode") == "semantic" for s in trace)
    strength_ok = all(s.get("recon_strength_mode") == strength_mode for s in trace)
    # G must be exactly the semantic source/target groups every step (as sets).
    groups_semantic = all(
        set(s.get("selected_group_ids", [])) == set(s.get("semantic_group_ids", []))
        for s in trace
    )
    # The reconstruction must be a *real* deletion, not v_hat ~ v.
    real_recon = any(
        s.get("recon_e_rel", {}).get("median", 0.0) > 1e-3
        and s.get("recon_q_ratio", {}).get("median", 0.0) < 0.999
        for s in trace
    )
    non_degenerate = all(not s.get("degenerate", True) for s in trace)

    mean_s = np.array([s.get("recon_mean_s", 0.0) for s in trace], dtype=np.float64)
    D = np.array([s.get("recon_perturb_norm_D", 0.0) for s in trace], dtype=np.float64)
    # Spread of the per-token strengths (p90 - p10, max over steps). Diagnostic only:
    # near-zero => s_i effectively uniform => error_guided degenerates to matched.
    selective_spread = float(max(
        (s.get("recon_s", {}).get("p90", 0.0) - s.get("recon_s", {}).get("p10", 0.0)
         for s in trace),
        default=0.0,
    ))
    if strength_mode == "full":
        # full replacement: every token's strength is exactly 1.
        strength_sane = bool(np.allclose(mean_s, 1.0, atol=1e-6))
    elif strength_mode == "error_guided":
        # selective: strengths sit strictly inside (0,1) — neither full nor zero.
        # (whether they are actually NON-uniform is reported, not gated, above.)
        strength_sane = bool(np.all((mean_s > 0.0) & (mean_s < 1.0)))
    elif strength_mode == "matched_strength":
        # uniform: s_i is constant across tokens, so per-step spread is ~0.
        uniform = all(
            abs(s.get("recon_s", {}).get("p90", 0.0) - s.get("recon_s", {}).get("p10", 0.0)) < 1e-6
            for s in trace
        )
        strength_sane = bool(np.all((mean_s > 0.0) & (mean_s < 1.0)) and uniform)
    else:
        raise ValueError(f"Unknown strength_mode: {strength_mode}")

    technical_pass = bool(
        feature_equal and guided and finite and n_tokens_ok and m_ok
        and disjoint_ok and recon_finite and kmeans_ok and sel_mode_ok
        and strength_ok and groups_semantic and real_recon and non_degenerate
        and strength_sane
    )
    audit = {
        "all_negative_branch_sees_clean_v": feature_equal,
        "all_guided_prefix": guided,
        "all_residual_norms_finite": finite,
        "all_256_tokens": n_tokens_ok,
        "all_M_basis": m_ok,
        "all_G_basis_disjoint": disjoint_ok,
        "all_recon_finite": recon_finite,
        "all_kmeans_K": kmeans_ok,
        "all_selection_mode": sel_mode_ok,
        "all_strength_mode": strength_ok,
        "group_semantics_ok": groups_semantic,
        "at_least_one_real_reconstruction": real_recon,
        "all_non_degenerate": non_degenerate,
        "strength_assignment_sane": strength_sane,
        "selective_spread": selective_spread,
        "technical_pass": technical_pass,
        "n_cd_steps": len(trace),
    }
    if not technical_pass:
        raise RuntimeError(f"{strength_mode} technical audit failed: {audit}")
    return audit


def config_lock(task_root: Path, task: str) -> None:
    lock = {
        "experiment": PROTOCOL,
        "phase": "2 Error-Guided Recon (strength assignment on the semantic region)",
        "task": task,
        "seeds": list(range(SEED_START, SEED_START + N_SEEDS)),
        "arms": list(ARMS),
        "lambda": LAMBDA,
        "guided_prefix": "both branches share a_{<q}^* = clean greedy tokens; negative branch teacher-forced",
        "kmeans": {"K": KMEANS_K, "seed": KMEANS_SEED, "n_init": 10,
                   "selector": "entity_set KMeans top-1 cos match (source/target), deduped"},
        "reconstruction": {
            "basis": "M tokens FPS on cosine distance from C = V \\ G (L2-normalized for FPS, RAW for reconstruction)",
            "M": M_BASIS,
            "rho_scale": RHO_SCALE,
            "rho": "rho = 1e-3 * tr(B_c^T B_c) / M (fixed, not swept)",
            "operator": "v_i -> mu_B + B_c alpha_i; alpha=(B_c^T B_c + rho I)^{-1} B_c^T (v_i - mu_B)",
            "token_count": "256 -> 256 (no pruning); positions unchanged; non-selected bit-identical",
            "scope": "source AND target groups excluded from basis; computed once per control observation",
        },
        "strength_assignment": {
            "e_i": "||v_i - v_hat_i|| / (||v_i|| + eps)",
            "full": "s_i = 1",
            "error_guided": "s_i = clip(e_i, 0, 1); v_i^- = (1-s_i) v_i + s_i v_hat_i",
            "matched_strength": "s_bar = mean_i clip(e_i,0,1); v_i^- = (1-s_bar) v_i + s_bar v_hat_i",
            "perturbation_norm": "D = ||V~_G - V_G||_F / (||V_G||_F + eps)",
        },
        "cd_tokens": "action dims 0..5, gripper dim 6 keeps clean",
        "pairing": "capture-once-reuse; hash-verified fail-closed across all 4 arms",
        "vanilla": "re-run here (4-arm snapshot pairing), not reused",
    }
    path = task_root / "CONFIG_LOCK.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing != lock:
            raise RuntimeError(f"CONFIG_LOCK differs from requested run: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


AUDITORS = {
    arm: (lambda t, sm=mode: audit_recon_trace(t, sm))
    for arm, mode in STRENGTH_MODE.items()
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="", help="comma/range; empty = 300..399")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    seeds = parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    task_root = artifact / "episodes" / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    config_lock(task_root, args.task)

    env, environment_id = make_environment(args.task)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, args.task, {}, False)
    policies = build_policies(OpenVLAInference(**policy_config), args.task)

    pairs = []
    for seed in seeds:
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        summaries: dict[str, dict] = {}
        initial_hashes: dict[str, tuple] = {}
        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
            arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
            if summary_path.exists() and arrays_path.exists():
                summaries[arm] = json.loads(summary_path.read_text())
                print(json.dumps({"skip": args.task, "seed": seed, "arm": arm}), flush=True)
                continue
            obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
            initial_hashes[arm] = (state_sha, rgb_sha)
            instruction = env.unwrapped.get_language_instruction()
            policy = policies[arm]
            policy.reset(instruction, seed=seed)
            policy._episode_trace = []
            policy._episode_logits = []
            result, steps, reason, actions, jitter = run_episode(
                env, policy, instruction, obs
            )
            write_arrays(arrays_path, policy._episode_logits, actions)
            trace = jsonable(policy._episode_trace)

            audit = None
            if arm != "vanilla":
                audit = AUDITORS[arm](trace)

            summary = {
                "protocol_id": PROTOCOL,
                "environment_id": environment_id,
                "task": args.task,
                "seed": seed,
                "episode_id": seed,
                "arm": arm,
                "instruction": instruction,
                "success": bool(result["success"]),
                "failure_reason": reason,
                "control_steps": steps,
                "action_jitter_index": jitter,
                "lambda": 0.0 if arm == "vanilla" else LAMBDA,
                "guided_prefix": arm != "vanilla",
                "kmeans_K": KMEANS_K,
                "kmeans_seed": KMEANS_SEED,
                "canonical_snapshot_sha256": canonical,
                "initial_state_sha256": state_sha,
                "initial_rgb_sha256": rgb_sha,
                "arrays_file": arrays_path.name,
                "selector_trace": trace,
                "result": jsonable(result),
            }
            if arm in RECON_ARMS:
                summary["reconstruction"] = {
                    "M": M_BASIS,
                    "rho_scale": RHO_SCALE,
                    "strength_mode": STRENGTH_MODE[arm],
                    "task_id": TASK_INDEX[args.task],
                }
                summary["mean_e_rel"] = float(np.mean([
                    s.get("recon_e_rel", {}).get("mean", 0.0) for s in trace
                ])) if trace else 0.0
                summary["mean_q_ratio"] = float(np.mean([
                    s.get("recon_q_ratio", {}).get("mean", 0.0) for s in trace
                ])) if trace else 0.0
                summary["mean_cos_v_vhat"] = float(np.mean([
                    s.get("recon_cos", {}).get("mean", 0.0) for s in trace
                ])) if trace else 0.0
                summary["mean_mean_s"] = float(np.mean([
                    s.get("recon_mean_s", 0.0) for s in trace
                ])) if trace else 0.0
                summary["mean_perturb_norm_D"] = float(np.mean([
                    s.get("recon_perturb_norm_D", 0.0) for s in trace
                ])) if trace else 0.0
                summary["mean_residual_norm"] = float(np.mean([
                    s.get("residual_norm", 0.0) for s in trace
                ]))
                summary["mean_num_tokens"] = float(np.mean([
                    s.get("num_tokens", 0) for s in trace
                ]))
            if audit is not None:
                summary.update(audit)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            summaries[arm] = summary
            print(json.dumps({
                "task": args.task, "seed": seed, "arm": arm,
                "success": summary["success"], "steps": steps,
                "technical_pass": audit["technical_pass"] if audit else True,
            }, sort_keys=True), flush=True)

        for arm in ARMS:
            if arm not in initial_hashes:
                initial_hashes[arm] = (
                    summaries[arm]["initial_state_sha256"],
                    summaries[arm]["initial_rgb_sha256"],
                )
        if len({v[0] for v in initial_hashes.values()}) != 1 or len({v[1] for v in initial_hashes.values()}) != 1:
            raise RuntimeError(f"Cross-arm initial-state mismatch for {args.task} seed {seed}")
        if len({s["canonical_snapshot_sha256"] for s in summaries.values()}) != 1:
            raise RuntimeError(f"Cross-arm snapshot mismatch for {args.task} seed {seed}")
        pairs.append({
            "seed": seed,
            "canonical_snapshot_sha256": canonical,
            "initial_state_sha256": initial_hashes[ARMS[0]][0],
            "initial_rgb_sha256": initial_hashes[ARMS[0]][1],
            "exact_four_arm_pairing": True,
        })

    manifest_path = task_root / f"pairing_manifest_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps({
        "task": args.task, "seeds": seeds, "arms": list(ARMS), "pairs": pairs,
        "all_four_arm_exact_pairing": True,
        "vanilla_source": "this_run",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()

"""SCR-CD: Semantic Context-Reconstruction Contrastive Decoding — 5-arm, 9-task paired rollout.

Phase 1 of the Semantic Token-CD program. The NEW operator here is the
context-reconstruction negative branch: for each *selected* semantic token v_i
(i in G = source ∪ target KMeans groups), keep only hat{v}_i = the part of v_i
that the surrounding (non-semantic) vision can linearly explain from an M=10
FPS-selected context basis B ⊂ C = V \\ G, and delete the region-unique
residual e_i = v_i - hat{v}_i. This is a *minimal information deletion* — the
opposite of Merge (prototype collapse) and of Attention-block (sever access).

Five paired arms on the SAME canonical snapshot per (task, seed):

    vanilla                 — clean OpenVLA (no CD).
    semantic_attn_k8_l8_15  — block Action Query -> selected visual keys, L8-15.
    semantic_merge_k8_eta100— collapse each selected group to its own prototype.
    semantic_recon_k8_m10   — context-reconstruction negative (G = semantic groups).
    random_recon_k8_m10     — context-reconstruction negative, but G = q *random*
                              non-semantic KMeans groups (q = #semantic groups),
                              total size matched within 20%. The MOST IMPORTANT
                              control (Semantic≈Random was seen before).

Reconstruction is computed ONCE per control observation; the 7 action tokens
share it via the guided single-pass forward. Token count stays 256, positions
unchanged, non-selected tokens bit-identical. CD identical to Phase 1B:
z* = z+ + 0.5(z+ - z-) on action dims 0..5, dim 6 clean, guided greedy prefix.

Seeds 300-399 (not used by any prior artifact). Vanilla is re-run HERE (not
reused) so all five arms restore the exact same canonical snapshot, hash-verified
fail-closed (initial_state / initial_rgb / canonical snapshot SHA256 all match).

Run (per task; seed-shard across processes/GPUs):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<gpu> \
  PYTHONPATH=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source \
    <venv>/bin/python research/semantic_token_cd/semantic_recon_rollout.py \
      --artifact artifacts/semantic_recon_k8_m10_v1 \
      --task google_robot_move_near --gpu 1 --seeds 300-319
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
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
from research.semantic_token_cd.semantic_merge_policy import (
    GuidedSemanticAttentionCDInference,
    SemanticMergeCDInference,
)
from research.semantic_token_cd.semantic_recon_policy import (
    M_BASIS,
    RHO_SCALE,
    SemanticReconCDInference,
)
from research.semantic_token_cd.st_shr_policy import STSHRCDInference
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)

PROTOCOL = "SCR_CD_SEMANTIC_RECON_K8_M10_V1"
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
TASK_INDEX = {t: i for i, t in enumerate(TASKS)}
ARMS = (
    "vanilla",
    "semantic_recon_k8_m10",
    "shr_harmonic",
)
LAMBDA = 0.5
KMEANS_K = 8
KMEANS_SEED = 0
LAYER_START, LAYER_END = 8, 16
MASK_VALUE = -1e4
ETA = 1.0
M_BASIS_TXT = M_BASIS
RHO_SCALE_TXT = RHO_SCALE
SEED_START = 300
N_SEEDS = 100
N_ATTENTION_LAYERS = LAYER_END - LAYER_START


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

    attn = copy.copy(base)
    attn.__class__ = STSHRCDInference
    _init_common(attn, LAMBDA)
    attn.beta = 0.0
    policies["shr_harmonic"] = attn

    merge = copy.copy(base)
    merge.__class__ = SemanticMergeCDInference
    _init_common(merge, LAMBDA)
    merge.eta = ETA
    policies["semantic_merge_k8_eta100"] = merge

    recon = copy.copy(base)
    recon.__class__ = SemanticReconCDInference
    _init_common(recon, LAMBDA)
    recon.recon_selection_mode = "semantic"
    recon.selection_mode = "semantic"
    recon.M = M_BASIS
    recon.rho_scale = RHO_SCALE
    recon._task_id = TASK_INDEX[task]
    policies["semantic_recon_k8_m10"] = recon

    rand = copy.copy(base)
    rand.__class__ = SemanticReconCDInference
    _init_common(rand, LAMBDA)
    rand.recon_selection_mode = "random_recon"
    rand.selection_mode = "random_recon"
    rand.M = M_BASIS
    rand.rho_scale = RHO_SCALE
    rand._task_id = TASK_INDEX[task]
    policies["random_recon_k8_m10"] = rand

    return policies


def audit_attn_trace(trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError("semantic_attn_k8_l8_15 arm produced no trace")
    feature_equal = all(s.get("feature_equal", False) for s in trace)
    guided = all(s.get("guided_prefix", False) for s in trace)
    finite = all(np.isfinite(s.get("residual_norm", 0.0)) for s in trace)
    hook_pass = all(s.get("attention_mask", {}).get("hook_calls") == N_ATTENTION_LAYERS for s in trace)
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_attention_hook_audits_pass": hook_pass,
        "all_guided_prefix": guided,
        "all_residual_norms_finite": finite,
        "technical_pass": feature_equal and hook_pass and guided and finite,
        "n_cd_steps": len(trace),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"semantic_attn_k8_l8_15 technical audit failed: {audit}")
    return audit


def audit_merge_trace(trace: list[dict]) -> dict:
    if not trace:
        raise RuntimeError("semantic_merge_k8_eta100 arm produced no trace")
    feature_equal = all(s.get("feature_equal", False) for s in trace)
    guided = all(s.get("guided_prefix", False) for s in trace)
    finite = all(np.isfinite(s.get("residual_norm", 0.0)) for s in trace)
    n_tokens_ok = all(s.get("n_tokens_negative", 0) == 256 for s in trace)
    eta_ok = all(abs(s.get("eta", 0.0) - ETA) < 1e-9 for s in trace)
    audit = {
        "all_negative_branch_sees_clean_v": feature_equal,
        "all_guided_prefix": guided,
        "all_residual_norms_finite": finite,
        "all_256_tokens": n_tokens_ok,
        "all_eta_100": eta_ok,
        "technical_pass": feature_equal and guided and finite and n_tokens_ok and eta_ok,
        "n_cd_steps": len(trace),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"semantic_merge_k8_eta100 technical audit failed: {audit}")
    return audit


def audit_recon_trace(trace: list[dict], mode: str) -> dict:
    if not trace:
        raise RuntimeError(f"{mode} arm produced no trace")
    feature_equal = all(s.get("feature_equal", False) for s in trace)
    guided = all(s.get("guided_prefix", False) for s in trace)
    finite = all(np.isfinite(s.get("residual_norm", 0.0)) for s in trace)
    n_tokens_ok = all(s.get("n_tokens_negative", 0) == 256 for s in trace)
    m_ok = all(s.get("recon_M", -1) == M_BASIS for s in trace)
    disjoint_ok = all(s.get("recon_g_basis_disjoint", False) for s in trace)
    recon_finite = all(s.get("recon_finite", False) for s in trace)
    sel_mode_ok = all(s.get("recon_selection_mode") == mode for s in trace)
    kmeans_ok = all(s.get("kmeans_K") == KMEANS_K for s in trace)
    # The reconstruction must be a *real* deletion, not v_hat ~ v: at least one
    # step has median e_rel strictly above zero AND median q_ratio strictly below 1.
    real_recon = any(
        s.get("recon_e_rel", {}).get("median", 0.0) > 1e-3
        and s.get("recon_q_ratio", {}).get("median", 0.0) < 0.999
        for s in trace
    )
    non_degenerate = all(not s.get("degenerate", True) for s in trace)

    if mode == "semantic":
        # G must be exactly the semantic source/target groups every step. The
        # recon policy sorts ``selected_groups`` while ``semantic_groups`` keeps
        # entity-iteration order, so compare as sets, not as ordered lists.
        groups_semantic = all(
            set(s.get("selected_group_ids", [])) == set(s.get("semantic_group_ids", []))
            for s in trace
        )
    else:  # random_recon
        # G must be non-semantic groups (disjoint from the semantic groups) and
        # differ from the semantic selection on at least one step.
        groups_semantic = all(
            set(s.get("selected_group_ids", [])) & set(s.get("semantic_group_ids", [])) == set()
            for s in trace
        )
        groups_semantic = groups_semantic and any(
            s.get("selected_group_ids") != s.get("semantic_group_ids") for s in trace
        )

    audit = {
        "all_negative_branch_sees_clean_v": feature_equal,
        "all_guided_prefix": guided,
        "all_residual_norms_finite": finite,
        "all_256_tokens": n_tokens_ok,
        "all_M_basis": m_ok,
        "all_G_basis_disjoint": disjoint_ok,
        "all_recon_finite": recon_finite,
        "all_selection_mode": sel_mode_ok,
        "all_kmeans_K": kmeans_ok,
        "at_least_one_real_reconstruction": real_recon,
        "all_non_degenerate": non_degenerate,
        "group_semantics_ok": groups_semantic,
        "technical_pass": (
            feature_equal and guided and finite and n_tokens_ok and m_ok
            and disjoint_ok and recon_finite and sel_mode_ok and kmeans_ok
            and real_recon and non_degenerate and groups_semantic
        ),
        "n_cd_steps": len(trace),
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"{mode} technical audit failed: {audit}")
    return audit


def config_lock(task_root: Path, task: str) -> None:
    lock = {
        "experiment": PROTOCOL,
        "phase": "1 SCR-CD (Semantic Context-Reconstruction CD)",
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
        "random_recon_k8_m10": {
            "groups": "q random NON-semantic groups (q = #semantic groups), size-matched within 20%",
            "repro": "rng seed = SeedSequence([task_id, episode_seed, timestep, 0x5CEED]); resample <=200x",
        },
        "cd_tokens": "action dims 0..5, gripper dim 6 keeps clean",
        "pairing": "capture-once-reuse; hash-verified fail-closed across all 5 arms",
        "vanilla": "re-run here (5-arm snapshot pairing), not reused",
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
    "shr_harmonic": lambda t: {"technical_pass": bool(t)},
    "semantic_merge_k8_eta100": audit_merge_trace,
    "semantic_recon_k8_m10": lambda t: audit_recon_trace(t, "semantic"),
    "random_recon_k8_m10": lambda t: audit_recon_trace(t, "random_recon"),
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
        snapshot_dir = artifact / "snapshots" / args.task
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"seed_{seed:03d}.pkl"
        if snapshot_path.exists():
            with snapshot_path.open("rb") as fh:
                snapshot = pickle.load(fh)
        else:
            snapshot = capture_snapshot(env, seed)
            tmp_snapshot = snapshot_path.with_suffix(f".{os.getpid()}.tmp")
            with tmp_snapshot.open("wb") as fh:
                pickle.dump(snapshot, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_snapshot, snapshot_path)
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
            if arm == "semantic_attn_k8_l8_15":
                summary["attention_layers"] = [LAYER_START, LAYER_END]
                summary["attention_mask_value_requested"] = float(MASK_VALUE)
            elif arm == "semantic_merge_k8_eta100":
                summary["eta"] = ETA
                summary["merge"] = "semantic_selected_group_prototype"
            elif arm in ("semantic_recon_k8_m10", "random_recon_k8_m10"):
                summary["reconstruction"] = {
                    "M": M_BASIS,
                    "rho_scale": RHO_SCALE,
                    "task_id": TASK_INDEX[args.task],
                }
                # E_t = (1/|G|) sum ||v_i - v_hat_i|| / (||v_i|| + eps), averaged over steps.
                summary["mean_e_rel"] = float(np.mean([
                    s.get("recon_e_rel", {}).get("mean", 0.0) for s in trace
                ])) if trace else 0.0
                summary["mean_q_ratio"] = float(np.mean([
                    s.get("recon_q_ratio", {}).get("mean", 0.0) for s in trace
                ])) if trace else 0.0
                summary["mean_cos_v_vhat"] = float(np.mean([
                    s.get("recon_cos", {}).get("mean", 0.0) for s in trace
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
            "exact_five_arm_pairing": True,
        })

    manifest_path = task_root / f"pairing_manifest_{min(seeds):03d}_{max(seeds):03d}.json"
    manifest_path.write_text(json.dumps({
        "task": args.task, "seeds": seeds, "arms": list(ARMS), "pairs": pairs,
        "all_five_arm_exact_pairing": True,
        "vanilla_source": "this_run",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"task": args.task, "DONE": True, "n_seeds": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()

"""VLA-Pruner reproduction -- FORMAL 1200-episode closed-loop worker.

Protocol: VLA_PRUNER_OPENVLA_REPRODUCTION_V1 (formal stage).
1200 = 100 seeds x 4 google_robot tasks x 3 arms
        (vanilla / vla_pruner_prune25 / vla_pruner_prune50),
strictly paired: every (task, seed) arm restores the identical canonical
snapshot with the current closed-loop harness.  The vanilla arm is re-run
(current-harness vanilla); old canonical vanilla episode files are never used.

Every prune-arm env step also runs one vanilla reference forward on the exact
same observation so first-divergence / token flips / raw-action L1/L2 measure
the local effect of pruning (executed trajectory is always the arm's own).

Resume / robustness:
  * episode-level resume: an episode is skipped iff both summary+arrays exist;
  * per-episode attempts with fresh environment rebuild after a technical
    failure (Vulkan / simulator), recorded to technical_failures_<worker>.jsonl;
  * audit failures are fatal (implementation bug, not a technical retry);
  * --check exits 0 only when every episode of the worker chunk is complete.

Timers are separated per executed env step: policy_ms (the arm's own generate,
== model inference for that step), ref_ms (vanilla reference forward, prune arms
only), env_ms (simulator env.step).

Run (per worker chunk, one tmux session per worker):
  python research/semantic_token_cd/vla_pruner_formal_rollout.py \
    --manifest RUN_MANIFEST.json --worker w1 --gpu 2 --artifact-dir <dir> [--round N]
  python research/semantic_token_cd/vla_pruner_formal_rollout.py \
    --manifest RUN_MANIFEST.json --worker w1 --gpu 2 --artifact-dir <dir> --check
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_ROOT = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e")
PCD_SOURCE = PCD_ROOT / "source"
CANONICAL_SNAPSHOTS = REPO_ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots"
DEFAULT_ARTIFACT = REPO_ROOT / "artifacts/vla_pruner_openvla_reproduction/formal_1200"

PROTOCOL = "VLA_PRUNER_OPENVLA_REPRODUCTION_V1"
FORMAL_STAGE = "formal_1200"
ACTION_DIM = 7
ARMS = ("vanilla", "vla_pruner_prune25", "vla_pruner_prune50")
ARM_TARGET_R = {"vanilla": None, "vla_pruner_prune25": 0.25, "vla_pruner_prune50": 0.50}
ARM_IMAGE_KEEP = {"vanilla": None, "vla_pruner_prune25": 192, "vla_pruner_prune50": 128}

for _p in (str(REPO_ROOT), str(PCD_SOURCE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_chunk(manifest_path: Path, worker: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if worker not in manifest["workers"]:
        raise KeyError(f"worker {worker} not in manifest")
    return manifest["workers"][worker]


def make_environment(task: str, gpu: int):
    from research.semantic_token_cd.xswap_rollout import make_environment as _me

    return _me(task, gpu)


def build_policies(base, task: str, arms) -> dict:
    from research.semantic_token_cd.vla_pruner_policy import build_vla_pruner_policy

    names = list(arms)
    if any(a != "vanilla" for a in arms) and "vanilla" not in names:
        names.append("vanilla")  # reference forward policy
    return {name: build_vla_pruner_policy(base, name, task) for name in names}


def load_base_policy(task: str):
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    config = get_policy_config("openvla", checkpoint, task, {}, False)
    return OpenVLAInference(**config)


def restore_and_sha(env, seed, snapshot):
    from research.semantic_token_cd.rollout_pilot import restore_snapshot, snapshot_sha

    canonical = snapshot_sha(snapshot)
    obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
    return obs, canonical, state_sha, rgb_sha


def run_episode(env, policy, ref_policy, instruction, obs, need_ref: bool):
    """Closed-loop episode; mirrors the audited calibration run semantics."""
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.rollout_pilot import flatten_action
    from utils import convert_numpy_or_torch_to_python, summarize

    image = get_image_from_maniskill2_obs_dict(env, obs)
    infos = []
    actions = []
    ref_tokens = []
    flips = []
    raw_l1 = []
    raw_l2 = []
    policy_ms = []
    ref_ms = []
    env_ms = []
    inferred = []  # policy inference ms attributed per executed env step
    inferred_ref = []
    predicted = False
    truncated = False
    control = 0
    while not (predicted or truncated) and control < 240:
        proprio = obs["agent"]["eef_pos"]
        t0 = time.monotonic()
        if need_ref:
            _r0, _ra0, _ = ref_policy.step(image, None, instruction, proprio=proprio)
            t_ref = (time.monotonic() - t0) * 1000.0
            ref_trace = ref_policy._episode_trace[-1]
            r_tok = list(ref_trace["token_ids"])
            r_raw = np.asarray(ref_trace["raw_action"], dtype=np.float64)
        else:
            t_ref, r_tok, r_raw = 0.0, None, None
        _raw, acts, _meta = policy.step(image, None, instruction, proprio=proprio)
        t_policy = (time.monotonic() - t0) * 1000.0 - t_ref
        trace = policy._episode_trace[-1]
        p_tok = list(trace["token_ids"])
        p_raw = np.asarray(trace["raw_action"], dtype=np.float64)
        if need_ref:
            if len(p_tok) != len(r_tok):
                raise RuntimeError("prune/ref token length mismatch")
            ref_tokens.append(r_tok)
            flips.append([int(a != b) for a, b in zip(p_tok, r_tok)])
            raw_l1.append(float(np.abs(p_raw - r_raw).sum()))
            raw_l2.append(float(np.sqrt(((p_raw - r_raw) ** 2).sum())))
        if not isinstance(acts, list):
            acts = [acts]
        n_acts = len(acts)
        for act in acts:
            executed = flatten_action(act)
            if executed.shape != (ACTION_DIM,) or not np.isfinite(executed).all():
                raise FloatingPointError(f"invalid executed action at step {control}: {executed}")
            actions.append(executed.copy())
            t_env = time.monotonic()
            obs, _reward, _success, truncated, info = env.step(executed)
            env_ms.append((time.monotonic() - t_env) * 1000.0)
            inferred.append(t_policy / n_acts)
            inferred_ref.append(t_ref / n_acts)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            control += 1
            infos.append(convert_numpy_or_torch_to_python(info))
            predicted = bool(act["terminate_episode"][0] > 0)
            if predicted and not env.unwrapped.is_final_subtask():
                predicted = False
                env.advance_to_next_subtask()
        policy_ms.append(t_policy)
        ref_ms.append(t_ref)
    result = summarize(infos)
    if result.get("success"):
        reason = None
    elif truncated:
        reason = "environment_time_limit"
    else:
        reason = "policy_terminated_without_success"
    return {
        "result": result,
        "actions": np.asarray(actions, dtype=np.float32),
        "infos": infos,
        "reason": reason,
        "ref_tokens": (np.asarray(ref_tokens, dtype=np.int64)
                       if ref_tokens else np.zeros((0, ACTION_DIM), dtype=np.int64)),
        "flips": (np.asarray(flips, dtype=np.int64)
                  if flips else np.zeros((0, ACTION_DIM), dtype=np.int64)),
        "raw_l1": np.asarray(raw_l1, dtype=np.float64),
        "raw_l2": np.asarray(raw_l2, dtype=np.float64),
        "policy_ms": np.asarray(policy_ms, dtype=np.float64),
        "ref_ms": np.asarray(ref_ms, dtype=np.float64),
        "env_ms": np.asarray(env_ms, dtype=np.float64),
        "inferred_ms": np.asarray(inferred, dtype=np.float64),
        "inferred_ref_ms": np.asarray(inferred_ref, dtype=np.float64),
    }


def audit(trace: list[dict], arm: str, need_ref: bool, warmup_flips: np.ndarray) -> dict:
    checks = {"trace_nonempty": bool(trace)}
    if arm == "vanilla":
        checks["never_prunes"] = all("pruned_count" not in x for x in trace)
        checks["all_finite_raw"] = all(np.isfinite(x["raw_action"]).all() for x in trace)
        checks["technical_pass"] = bool(trace) and all(checks.values())
        return checks
    target_r = ARM_TARGET_R[arm]
    warmup_ok = []
    cfg_locked = []
    for x in trace:
        hist = int(x.get("history_len", -1))
        r_eff = float(x.get("fastv_r_effective", -1.0))
        prune_count = int(x.get("pruned_count", -1))
        kept_img = x.get("kept_image_count", None)
        layer = x.get("pruning_layer", None)
        if hist < 3:
            warmup_ok.append(r_eff == 0.0 and prune_count == 0 and kept_img == 256 and layer == 3)
        else:
            cfg_locked.append(
                abs(r_eff - float(target_r)) < 1e-9
                and prune_count > 0
                and kept_img == ARM_IMAGE_KEEP[arm]
                and layer == 3
            )
    checks["first_trace_history_zero"] = int(trace[0].get("history_len", -1)) == 0
    checks["warmup_steps_never_prune"] = bool(warmup_ok) and all(warmup_ok)
    checks["pruning_layer_locked_3"] = all(int(x.get("pruning_layer", -1)) == 3 for x in trace)
    checks["prune_rows_cfg_locked"] = (not cfg_locked) or all(cfg_locked)
    checks["kept_image_count_violations"] = not any(
        x.get("kept_image_count") not in (256, ARM_IMAGE_KEEP[arm]) for x in trace
    )
    checks["original_seq_constant"] = len({int(x["original_seq_length"]) for x in trace}) <= 1
    checks["all_finite_raw"] = all(np.isfinite(x["raw_action"]).all() for x in trace)
    checks["warmup_exact_vs_ref"] = True
    if need_ref:
        warmup_rows = int(sum(int(x.get("history_len", -1)) < 3 for x in trace))
        checks["warmup_exact_vs_ref"] = (
            warmup_rows == int(warmup_flips.shape[0])
            and bool(np.count_nonzero(warmup_flips) == 0)
        )
    checks["technical_pass"] = all(bool(v) for k, v in checks.items() if k != "technical_pass")
    return checks


def episode_stats(trace, flips, raw_l1, raw_l2, need_ref: bool) -> dict:
    n = len(trace)
    out = {"policy_steps": n}
    if not n:
        return out
    hist = np.asarray([int(x.get("history_len", -1)) for x in trace], dtype=np.int64)
    prune_counts = np.asarray([int(x.get("pruned_count", 0) or 0) for x in trace], dtype=np.int64)
    kept_img = np.asarray([int(x.get("kept_image_count", 256)) for x in trace], dtype=np.int64)
    overlap = np.asarray(
        [x.get("guide_topk_overlap") if x.get("guide_topk_overlap") is not None else -1
         for x in trace], dtype=np.int64
    )
    active = hist >= 3
    out["prune_activation_steps"] = int((active & (prune_counts > 0)).sum())
    out["total_pruned_tokens"] = int(prune_counts.sum())
    out["mean_kept_image_count_active"] = float(kept_img[active].mean()) if active.any() else None
    out["actual_pruning_ratio_active"] = (
        float(1.0 - kept_img[active].mean() / 256.0) if active.any() else None
    )
    bucket = np.zeros(n, dtype=np.int64)
    third = max(1, n // 3)
    for i in range(n):
        bucket[i] = min(2, i // third) if third else 0
    for label, b in (("early", 0), ("middle", 1), ("late", 2)):
        sel = (bucket == b) & active
        out[f"phase_{label}_prune_steps"] = int((sel & (prune_counts > 0)).sum())
        out[f"phase_{label}_pruned_tokens"] = int(prune_counts[sel].sum())
        if need_ref:
            fsel = flips[(bucket == b)]
            l1sel = raw_l1[(bucket == b)]
            out[f"phase_{label}_flips"] = int(fsel.sum()) if len(fsel) else 0
            out[f"phase_{label}_flip_steps"] = int((fsel.sum(axis=1) > 0).sum()) if len(fsel) else 0
            out[f"phase_{label}_mean_raw_l1"] = float(l1sel.mean()) if len(l1sel) else None
    out["guide_topk_overlap_active"] = (
        float(overlap[(overlap >= 0) & active].mean()) if ((overlap >= 0) & active).any() else None
    )
    out["overlap_used_redundancy_steps"] = int(
        sum(1 for x in trace if x.get("used_redundancy"))
    )
    out["history_used_steps"] = int(active.sum())
    if need_ref:
        out["total_token_flips"] = int(flips.sum())
        out["flip_steps"] = int((flips.sum(axis=1) > 0).sum())
        out["mean_raw_l1"] = float(raw_l1.mean()) if len(raw_l1) else None
        out["mean_raw_l2"] = float(raw_l2.mean()) if len(raw_l2) else None
        nz = np.flatnonzero(flips.sum(axis=1) > 0)
        out["first_divergence_step"] = int(nz[0]) if len(nz) else None
    return out


def save_episode(artifact: Path, task: str, seed: int, arm: str, summary: dict,
                 arrays: dict) -> None:
    arm_root = artifact / "episodes" / task / arm
    summary_path = arm_root / f"episode_{seed:03d}_summary.json"
    arrays_path = arm_root / f"episode_{seed:03d}_arrays.npz"
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    atomic_json(summary_path, summary)


def chunk_done(artifact: Path, chunk: dict) -> tuple[bool, int, int]:
    done = 0
    total = 0
    for task, seeds in chunk["tasks"].items():
        for seed in seeds:
            for arm in ARMS:
                total += 1
                sp = artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                ap = artifact / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"
                if sp.exists() and ap.exists():
                    done += 1
    return done == total, done, total


def record_failure(artifact: Path, worker: str, task: str, seed: int, arm: str,
                   attempt: int, error: str) -> None:
    path = artifact / "logs" / f"technical_failures_{worker}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({
            "task": task, "seed": seed, "arm": arm, "attempt": attempt,
            "error": error[:2000], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")


def run_chunk(artifact: Path, chunk: dict, worker: str, gpu: int,
              max_attempts: int, round_no: int) -> tuple[int, int]:
    """Run pending episodes of this worker chunk. Returns (done, failed_after_retries)."""
    total_pending = 0
    failures = 0
    for task in chunk["tasks"]:
        env = None
        base = None
        try:
            seeds = [int(s) for s in chunk["tasks"][task]]
            env, _environment_id = make_environment(task, gpu)
            base = load_base_policy(task)
            policies = build_policies(base, task, ARMS)
            for seed in seeds:
                snapshot_path = CANONICAL_SNAPSHOTS / task / f"seed_{seed:03d}.pkl"
                if not snapshot_path.exists():
                    raise FileNotFoundError(f"missing canonical snapshot: {snapshot_path}")
                for arm in ARMS:
                    sp = artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                    ap = artifact / "episodes" / task / arm / f"episode_{seed:03d}_arrays.npz"
                    if sp.exists() and ap.exists():
                        continue
                    total_pending += 1
                    need_ref = arm != "vanilla"
                    success_saved = False
                    last_err = None
                    for attempt in range(1, max_attempts + 1):
                        try:
                            if env is None:
                                env, _environment_id = make_environment(task, gpu)
                            with snapshot_path.open("rb") as handle:
                                snapshot = pickle.load(handle)
                            obs, canonical, state_sha, rgb_sha = restore_and_sha(env, seed, snapshot)
                            instruction = env.unwrapped.get_language_instruction()
                            policy = policies[arm]
                            policy.reset(instruction, seed=seed)
                            policy._episode_trace = []
                            policy._episode_step = 0
                            ref_policy = policies["vanilla"] if need_ref else None
                            if need_ref:
                                ref_policy.reset(instruction, seed=seed)
                                ref_policy._episode_trace = []
                                ref_policy._episode_step = 0
                            started = time.monotonic()
                            ep = run_episode(env, policy, ref_policy, instruction, obs,
                                             need_ref=need_ref)
                            runtime = time.monotonic() - started
                            trace_typed = list(policy._episode_trace)
                            trace = jsonable(trace_typed)
                            flips = ep["flips"] if need_ref else np.zeros((0, ACTION_DIM), dtype=np.int64)
                            n_warmup = int(sum(int(x.get("history_len", -1)) < 3 for x in trace_typed))
                            checks = audit(trace_typed, arm, need_ref, flips[:n_warmup])
                            if not checks["technical_pass"]:
                                raise RuntimeError(f"audit failed {arm} seed={seed}: {checks}")
                            stats = episode_stats(trace_typed, flips, ep["raw_l1"], ep["raw_l2"],
                                                  need_ref)
                            summary = {
                                "protocol_id": PROTOCOL,
                                "stage": FORMAL_STAGE,
                                "task": task,
                                "seed": seed,
                                "episode_id": seed,
                                "arm": arm,
                                "fastv_r": ARM_TARGET_R[arm],
                                "fastv_k": 3,
                                "temporal_w": 3,
                                "temporal_gamma": 0.8,
                                "selection": "prefill",
                                "history_layers": list(
                                    getattr(policy, "fastv_cfg", {}).get("history_layers", (15,))
                                ) if getattr(policy, "fastv_cfg", None) is not None else None,
                                "combine_filter": True,
                                "redundancy_filter": True,
                                "instruction": instruction,
                                "success": bool(ep["result"].get("success", False)),
                                "result": jsonable(ep["result"]),
                                "failure_reason": ep["reason"],
                                "control_steps": len(ep["infos"]),
                                "runtime_seconds": runtime,
                                "gpu_id": gpu,
                                "worker_id": worker,
                                "round_no": round_no,
                                "canonical_snapshot_sha256": canonical,
                                "initial_state_sha256": state_sha,
                                "initial_rgb_sha256": rgb_sha,
                                "prune_stats": stats,
                                "trace": trace,
                                **checks,
                            }
                            arrays = {
                                "executed_actions": np.asarray(ep["actions"], dtype=np.float32),
                                "ref_token_ids": ep["ref_tokens"],
                                "token_flips": flips,
                                "raw_l1": ep["raw_l1"],
                                "raw_l2": ep["raw_l2"],
                                "policy_ms": ep["policy_ms"],
                                "ref_ms": ep["ref_ms"],
                                "env_ms": ep["env_ms"],
                                "inferred_ms": ep["inferred_ms"],
                                "inferred_ref_ms": ep["inferred_ref_ms"],
                                "kept_image_count": np.asarray(
                                    [int(x.get("kept_image_count", 256)) for x in trace_typed],
                                    dtype=np.int64),
                                "pruned_count": np.asarray(
                                    [int(x.get("pruned_count", 0) or 0) for x in trace_typed],
                                    dtype=np.int64),
                                "history_len": np.asarray(
                                    [int(x.get("history_len", -1)) for x in trace_typed],
                                    dtype=np.int64),
                                "guide_topk_overlap": np.asarray(
                                    [x.get("guide_topk_overlap") if x.get("guide_topk_overlap") is not None else -1
                                     for x in trace_typed], dtype=np.int64),
                            }
                            save_episode(artifact, task, seed, arm, summary, arrays)
                            print(json.dumps({
                                "task": task, "seed": seed, "arm": arm,
                                "success": summary["success"], "steps": len(ep["infos"]),
                                "runtime_seconds": round(runtime, 2),
                                "attempt": attempt,
                                "prune_activation_steps": stats.get("prune_activation_steps"),
                                "total_token_flips": stats.get("total_token_flips"),
                                "first_divergence_step": stats.get("first_divergence_step"),
                                "technical_pass": True,
                            }, default=str), flush=True)
                            success_saved = True
                            break
                        except Exception as exc:  # technical failure -> fresh env + retry
                            if "audit failed" in str(exc):
                                raise
                            last_err = repr(exc)
                            record_failure(artifact, worker, task, seed, arm, attempt, last_err)
                            print(json.dumps({"TECH_FAIL": task, "seed": seed, "arm": arm,
                                              "attempt": attempt, "error": last_err}), flush=True)
                            try:
                                if env is not None:
                                    env.close()
                            except Exception:
                                pass
                            env = None
                            time.sleep(15 * attempt)
                    if not success_saved:
                        failures += 1
                        print(json.dumps({"FAILED_AFTER_RETRIES": task, "seed": seed,
                                          "arm": arm, "last_error": last_err}), flush=True)
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            del base
            gc.collect()
            torch.cuda.empty_cache()
    done_now, done_total, total = chunk_done(artifact, chunk)
    return done_total, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    artifact = args.artifact_dir.resolve()
    chunk = load_chunk(args.manifest.resolve(), args.worker)
    if int(chunk.get("gpu", args.gpu)) != int(args.gpu):
        raise RuntimeError(f"manifest gpu {chunk.get('gpu')} != --gpu {args.gpu}")
    if args.check:
        is_done, done, total = chunk_done(artifact, chunk)
        print(json.dumps({"worker": args.worker, "done": done, "total": total,
                          "complete": is_done}), flush=True)
        raise SystemExit(0 if is_done else 1)
    done, failed = run_chunk(artifact, chunk, args.worker, args.gpu,
                             args.max_attempts, args.round)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired Vanilla / historical dynamic L11-Matched rollouts on RT depth 0."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
CANON = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
RT0_REFERENCE = ROOT / "artifacts/l11_rt_depth0_pick_move_full_100_129_v1/episodes"
DRAWER_RT0_REFERENCE = ROOT / "artifacts/l11_rank_causal_effect_study_v1_20260926/closedloop_simpler"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
)
PROTOCOL = "SIMPLER_RT_DEPTH0_DRAWERS_L11_MATCHED_PAIRED_0_299_V1"
RENDERER = "SAPIEN RT depth0, spp=32, denoiser=true"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_seeds(spec: str) -> list[int]:
    result: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = (int(x) for x in item.split("-", 1))
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(item))
    seeds = sorted(set(result))
    if not seeds or any(s < 0 or s > 299 for s in seeds):
        raise ValueError("This experiment is locked to canonical seeds 0-299")
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    for path in (ROOT / "task1/shim_site", ROOT, PCD):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch
    torch.set_num_threads(1)
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import jsonable, restore_snapshot, snapshot_sha
    from research.semantic_token_cd.distractor_policy import (
        ACTION_VOCAB_SIZE, AuditedVanillaInference, _action_logits,
    )
    from research.semantic_token_cd.l11_causal_group_preflight import build_historical_policy
    from research.semantic_token_cd.prompt_attn_shr_rollout import write_arrays
    from research.semantic_token_cd.replay_simpler_l11_rank_effects import make_rt_depth0_environment
    from research.semantic_token_cd.spatial_grid_rollout import run_episode

    output = args.output.resolve()
    env = base = vanilla = matched = None
    try:
        env = make_rt_depth0_environment(args.task)
        config = get_policy_config("openvla", str(PCD / "pretrained/openvla-7b"), args.task, {}, False)
        base = OpenVLAInference(**config)
        class ProtocolVanillaInference(AuditedVanillaInference):
            """Audited vanilla with the repository baseline's early-EOS fallback.

            OpenVLAContrastInference.predict_action replaces a prematurely
            terminated score sequence with zero logits of the configured
            fixed action length. AuditedVanillaInference normally treats this
            rare case as fatal; reproduce the established baseline fallback
            here so it cannot selectively drop WidowX episodes.
            """

            def step(self, image, contrast_image=None, task_description=None, *args, **kwargs):
                inputs = self.process_inputs(image, task_description=task_description)
                clean_scores = self._forward_scores(inputs, self.unnorm_key, do_sample=False)
                expected = int(self.get_action_dim(self.unnorm_key))
                if expected != 7:
                    raise RuntimeError(f"Expected 7 action dimensions, got {expected}")
                early_eos_fallback = clean_scores.shape[0] < expected
                if early_eos_fallback:
                    clean_scores = torch.zeros(
                        (expected, int(self.vla.vocab_size)),
                        dtype=clean_scores.dtype,
                        device=clean_scores.device,
                    )
                elif clean_scores.shape[0] > expected:
                    clean_scores = clean_scores[:expected]
                token_ids = clean_scores.argmax(dim=-1)
                raw = self._decode_actions(token_ids, self.unnorm_key)[None]
                raw_action, actions = self.postprocess_actions(raw)
                self._episode_logits.append({"positive": _action_logits(self, clean_scores)})
                meta = {
                    "lambda": 0.0,
                    "residual_norm": 0.0,
                    "early_eos_fallback": bool(early_eos_fallback),
                }
                self._episode_trace.append(meta)
                return raw_action, actions, meta

        vanilla = copy.copy(base)
        vanilla.__class__ = ProtocolVanillaInference
        vanilla._episode_trace, vanilla._episode_logits = [], []
        matched = build_historical_policy(base, args.task)
        for seed in seeds:
            snapshot_path = CANON / "snapshots" / args.task / f"seed_{seed:03d}.pkl"
            reference_path = CANON / "episodes" / args.task / "vanilla" / f"episode_{seed:03d}_summary.json"
            snapshot = pickle.loads(snapshot_path.read_bytes())
            reference = json.loads(reference_path.read_text())
            canonical_hash = snapshot_sha(snapshot)
            if canonical_hash != reference.get("canonical_snapshot_sha256"):
                raise RuntimeError(f"Canonical snapshot hash mismatch: {args.task}:{seed}")

            pair_rgb = None
            for arm, policy in (("vanilla", vanilla), ("l11_matched", matched)):
                arm_dir = output / "episodes" / args.task / arm
                summary_path = arm_dir / f"episode_{seed:03d}_summary.json"
                arrays_path = arm_dir / f"episode_{seed:03d}_arrays.npz"
                if summary_path.exists() and arrays_path.exists():
                    old = json.loads(summary_path.read_text())
                    if old.get("protocol_id") != PROTOCOL or old.get("renderer") != RENDERER:
                        raise RuntimeError(f"Incompatible pre-existing output: {summary_path}")
                    if pair_rgb is not None and old.get("initial_rgb_sha256") != pair_rgb:
                        raise RuntimeError(f"Existing pair RGB mismatch: {args.task}:{seed}")
                    pair_rgb = old.get("initial_rgb_sha256")
                    continue

                obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
                if state_hash != reference.get("initial_state_sha256"):
                    raise RuntimeError(f"Restored state hash mismatch: {args.task}:{seed}")
                # Reuse the independently audited RT-depth0 image identity for
                # Pick Coke Can and Move Near wherever the 100-129 records exist.
                reference_rt0 = RT0_REFERENCE / args.task / arm / f"episode_{seed:03d}_summary.json"
                if reference_rt0.exists():
                    rt0 = json.loads(reference_rt0.read_text())
                    if (rt0.get("canonical_snapshot_sha256") != canonical_hash or
                            rt0.get("initial_state_sha256") != state_hash or
                            rt0.get("initial_rgb_sha256") != rgb_hash or
                            rt0.get("renderer") != RENDERER):
                        raise RuntimeError(f"Known RT-depth0 identity mismatch: {args.task}:{seed}:{arm}")
                if args.task in ("google_robot_open_drawer", "google_robot_close_drawer"):
                    drawer_ref = (DRAWER_RT0_REFERENCE / args.task / arm /
                                  f"seed_{seed:03d}.json")
                    if drawer_ref.exists():
                        known = json.loads(drawer_ref.read_text())
                        if (known.get("canonical_snapshot_sha256") != canonical_hash or
                                known.get("initial_state_sha256") != state_hash or
                                known.get("initial_rgb_sha256_rt0") != rgb_hash or
                                known.get("renderer") != RENDERER):
                            raise RuntimeError(f"Known drawer RT-depth0 identity mismatch: {args.task}:{seed}:{arm}")
                if pair_rgb is not None and rgb_hash != pair_rgb:
                    raise RuntimeError(f"Vanilla/L11 initial RGB mismatch: {args.task}:{seed}")
                pair_rgb = rgb_hash
                instruction = env.unwrapped.get_language_instruction()
                if instruction != reference.get("instruction"):
                    raise RuntimeError(f"Instruction mismatch: {args.task}:{seed}")

                arm_dir.mkdir(parents=True, exist_ok=True)
                policy.reset(instruction, seed=seed)
                policy._episode_trace, policy._episode_logits = [], []
                started = time.monotonic()
                result, steps, reason, actions, jitter = run_episode(env, policy, instruction, obs)
                elapsed = time.monotonic() - started
                trace = jsonable(policy._episode_trace)
                if arm == "l11_matched":
                    if not trace or not all(row.get("coverage_exact") is True for row in trace):
                        raise RuntimeError(f"Historical Matched coverage audit failed: {args.task}:{seed}")
                    if not all(abs(float(row.get("lambda", -1)) - 0.5) < 1e-12 for row in trace):
                        raise RuntimeError(f"Historical lambda audit failed: {args.task}:{seed}")
                write_arrays(arrays_path, policy._episode_logits, actions)
                payload = {
                    "protocol_id": PROTOCOL,
                    "benchmark": "SIMPLER",
                    "task": args.task,
                    "seed": seed,
                    "arm": arm,
                    "selector": "historical PromptAttentionSHRInference matched-budget path" if arm == "l11_matched" else "Vanilla",
                    "success": bool(result["success"]),
                    "steps": int(steps),
                    "failure_reason": reason,
                    "runtime_seconds": elapsed,
                    "action_jitter_index": jitter,
                    "canonical_snapshot_sha256": canonical_hash,
                    "initial_state_sha256": state_hash,
                    "initial_rgb_sha256": rgb_hash,
                    "renderer": RENDERER,
                    "renderer_gpu": args.gpu,
                    "inference_gpu": args.gpu,
                    "lambda": 0.5 if arm == "l11_matched" else 0.0,
                    "mean_m_t": float(np.mean([row["m_t"] for row in trace])) if arm == "l11_matched" and trace else 0.0,
                    "sampled_states": len(trace),
                    "selector_trace": trace,
                    "arrays_file": arrays_path.name,
                    "worker_id": args.worker_id,
                }
                atomic_json(summary_path, payload)
                print(json.dumps({"worker": args.worker_id, "gpu": args.gpu, "task": args.task,
                                  "seed": seed, "arm": arm, "success": payload["success"],
                                  "steps": steps, "seconds": round(elapsed, 1),
                                  "mean_m_t": round(payload["mean_m_t"], 2),
                                  "rgb": rgb_hash}, sort_keys=True), flush=True)
    except Exception as exc:
        fail_path = output / "logs" / f"failure_{args.worker_id}_{int(time.time())}.json"
        atomic_json(fail_path, {"task": args.task, "seeds": args.seeds, "gpu": args.gpu,
                                "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        del matched, vanilla, base, env
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

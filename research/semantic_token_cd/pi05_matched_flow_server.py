"""Serve pi0.5-LIBERO with L6 matched harmonic flow guidance."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from research.semantic_token_cd.pi05_matched_flow_cd import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_LAYER_INDEX,
    Pi05MatchedFlowCDPolicy,
)


class Pi05VanillaMatchedPolicy:
    """Route paired requests through one model with matched action noise."""

    def __init__(self, policy: Pi05MatchedFlowCDPolicy) -> None:
        self.policy = policy

    def infer(self, observation: dict) -> dict:
        request = dict(observation)
        arm = str(request.pop("pi05_arm", "matched"))
        noise_seed = int(request.pop("pi05_noise_seed", 0))
        guidance_scale = request.pop("pi05_guidance_scale", None)
        layer_index = request.pop("pi05_layer_index", None)
        active_steps = request.pop("pi05_active_steps", None)
        num_steps = request.pop("pi05_num_steps", None)
        trace_candidates = bool(request.pop("pi05_trace_candidates", False))
        if isinstance(active_steps, str):
            active_steps = [
                int(part) for part in active_steps.split(",") if part.strip()
            ]
        if arm == "capture":
            scores = self.policy.capture_all_layers(request)
            return {
                "attention_scores": np.asarray(scores, dtype=np.float32),
                "pi05_paired_trace": {"arm": arm, "n_layers": int(scores.shape[0])},
            }
        if arm in ("pcd", "uniform"):
            n_candidates = int(request.pop("pi05_pcd_candidates", 8))
            alpha = float(request.pop("pi05_pcd_alpha", 0.5))
            if arm == "uniform":
                # alpha=0 makes softmax(alpha * score) uniform over the same
                # candidate pool, so this arm is the random-selection control.
                alpha = 0.0
            bandwidth_scale = float(request.pop("pi05_pcd_bandwidth_scale", 0.5))
            key = self.policy.jax.random.key(noise_seed)
            shape = (self.policy.model.action_horizon, self.policy.model.action_dim)
            vanilla_noise = self.policy.jax.random.normal(key, shape)
            extra_keys = self.policy.jax.random.split(
                self.policy.jax.random.fold_in(key, 1), n_candidates - 1
            )
            extra_noise = self.policy.jax.vmap(
                lambda candidate_key: self.policy.jax.random.normal(candidate_key, shape)
            )(extra_keys)
            candidate_noise = np.concatenate(
                [np.asarray(vanilla_noise, dtype=np.float32)[None, ...],
                 np.asarray(extra_noise, dtype=np.float32)],
                axis=0,
            )
            result = self.policy.infer_pcd(
                request,
                seed=noise_seed,
                noise=candidate_noise,
                n_candidates=n_candidates,
                alpha=alpha,
                bandwidth_scale=bandwidth_scale,
                num_steps=num_steps,
                trace_candidates=trace_candidates,
            )
            result["pi05_paired_trace"] = {
                "arm": arm,
                "noise_seed": noise_seed,
                **result.get("pi05_pcd_trace", {}),
            }
            return result
        noise = np.asarray(
            self.policy.jax.random.normal(
                self.policy.jax.random.key(noise_seed),
                (self.policy.model.action_horizon, self.policy.model.action_dim),
            ),
            dtype=np.float32,
        )
        if arm == "vanilla":
            original_kwargs = dict(self.policy.policy._sample_kwargs)
            if num_steps is not None:
                self.policy.policy._sample_kwargs = {
                    **original_kwargs,
                    "num_steps": int(num_steps),
                }
            try:
                result = self.policy.policy.infer(request, noise=noise)
            finally:
                self.policy.policy._sample_kwargs = original_kwargs
            result["pi05_paired_trace"] = {
                "arm": arm,
                "noise_seed": noise_seed,
                "guidance_scale": 0.0,
                "num_flow_steps": int(num_steps) if num_steps is not None else None,
            }
            if trace_candidates:
                result["pi05_vtcg_candidates"] = {"noise": noise}
            return result
        if arm != "matched":
            raise ValueError(f"unknown pi0.5 paired arm {arm!r}")
        result = self.policy.infer(
            request,
            seed=noise_seed,
            noise=noise,
            guidance_scale=guidance_scale,
            layer_index=layer_index,
            active_steps=active_steps,
            num_steps=num_steps,
        )
        trace = self.policy.last_trace
        result["pi05_paired_trace"] = {
            "arm": arm,
            "noise_seed": noise_seed,
            "guidance_scale": trace["guidance_scale"],
            "selected_token_count": trace["selected_token_count"],
            "selector_layer_index_0based": trace["selector_layer_index_0based"],
            "active_flow_steps": trace["active_flow_steps"],
            "num_flow_steps": trace["num_flow_steps"],
            # Lever-diagnostic fields (additive; used by pi05_lever_diagnostic.py).
            "selected_token_ids": list(trace["selected_token_ids"]),
            "matched_m_t": int(trace["matched_m_t"]),
            "matched_cluster_ids": list(trace["matched_cluster_ids"]),
            "entities": list(trace["entities"]),
            "selected_count_equals_matched": bool(trace["selected_count_equals_matched"]),
            "attention_score_min": float(trace["attention_score_min"]),
            "attention_score_max": float(trace["attention_score_max"]),
        }
        return result


def _parse_steps(value: str, num_steps: int) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(num_steps))
    steps = tuple(sorted(set(int(part) for part in value.split(",") if part.strip())))
    if not steps:
        raise argparse.ArgumentTypeError("active steps cannot be empty")
    if any(step < 0 or step >= num_steps for step in steps):
        raise argparse.ArgumentTypeError(f"active steps must be in [0, {num_steps - 1}]")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--active-steps", default="all")
    parser.add_argument("--warmup-only", action="store_true")
    args = parser.parse_args()
    active_steps = _parse_steps(args.active_steps, args.num_steps)

    policy = Pi05MatchedFlowCDPolicy(
        checkpoint=args.checkpoint,
        gpu=args.gpu,
        layer_index=args.layer_index,
        guidance_scale=args.guidance_scale,
        active_steps=active_steps,
        num_steps=args.num_steps,
    )
    sample = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "pick up the black bowl and place it on the plate",
    }
    result = policy.infer(sample, seed=0)
    actions = np.asarray(result["actions"])
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise RuntimeError(f"pi0.5 guided warmup returned invalid actions {actions.shape}")
    n_candidates = 8
    pcd_key = policy.jax.random.key(0)
    pcd_shape = (policy.model.action_horizon, policy.model.action_dim)
    base_noise = policy.jax.random.normal(pcd_key, pcd_shape)
    extra_keys = policy.jax.random.split(policy.jax.random.fold_in(pcd_key, 1), n_candidates - 1)
    extra_noise = policy.jax.vmap(
        lambda key: policy.jax.random.normal(key, pcd_shape)
    )(extra_keys)
    pcd_result = policy.infer_pcd(
        sample,
        seed=0,
        noise=np.concatenate(
            [np.asarray(base_noise, dtype=np.float32)[None, ...],
             np.asarray(extra_noise, dtype=np.float32)], axis=0
        ),
        n_candidates=n_candidates,
        alpha=0.5,
        bandwidth_scale=0.5,
        num_steps=args.num_steps,
    )
    pcd_actions = np.asarray(pcd_result["actions"])
    if pcd_actions.shape != (10, 7) or not np.isfinite(pcd_actions).all():
        raise RuntimeError(f"pi0.5 PCD warmup returned invalid actions {pcd_actions.shape}")
    if len(pcd_result["pi05_pcd_trace"]["weights"]) != n_candidates:
        raise RuntimeError("pi0.5 PCD warmup returned malformed candidate weights")
    logging.info(
        "pi0.5 PCD warmup passed: shape=%s candidates=%d selected=%d score_terms=%d",
        pcd_actions.shape, n_candidates, pcd_result["pi05_pcd_trace"]["chosen_index"],
        pcd_result["pi05_pcd_trace"]["score_terms"],
    )
    logging.info(
        "pi0.5 guided warmup passed: shape=%s selected=%d lambda=%.3f active_steps=%s",
        actions.shape,
        policy.last_trace["selected_token_count"],
        args.guidance_scale,
        active_steps,
    )
    if args.warmup_only:
        return

    from openpi.serving import websocket_policy_server

    metadata = {
        "model": "pi0.5",
        "benchmark": "LIBERO",
        "method": "L10 matched-flow guidance and PCD harmonic contrastive resampling",
        "checkpoint": str(args.checkpoint),
        "guidance_scale": args.guidance_scale,
        "active_flow_steps": list(active_steps),
        "action_horizon": 10,
        "action_dim": 7,
    }
    websocket_policy_server.WebsocketPolicyServer(
        policy=Pi05VanillaMatchedPolicy(policy),
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

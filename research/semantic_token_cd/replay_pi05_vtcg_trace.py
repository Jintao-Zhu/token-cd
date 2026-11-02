from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.pi05_matched_flow_cd import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LAYER_INDEX,
    Pi05MatchedFlowCDPolicy,
)


TRACE_ARRAYS = (
    "noise",
    "actions_clean",
    "actions_cf",
    "log_pc",
    "log_pn",
    "normalized_score",
    "weights",
    "chosen_index",
)


def _text(data, key: str) -> str:
    return str(np.asarray(data[key]).item())


def _hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if not actual.size:
        return 0.0
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(args.report)
    trace_paths = sorted(args.trace_dir.rglob("*.npz"))
    if not trace_paths:
        raise SystemExit(f"no trace files in {args.trace_dir}")

    policy = Pi05MatchedFlowCDPolicy(
        checkpoint=args.checkpoint,
        gpu=args.gpu,
        layer_index=DEFAULT_LAYER_INDEX,
        num_steps=10,
    )
    rows = []
    for path in trace_paths:
        with np.load(path, allow_pickle=False) as data:
            image = np.asarray(data["image"])
            wrist_image = np.asarray(data["wrist_image"])
            state = np.asarray(data["state"])
            prompt = _text(data, "prompt")
            request = {
                "observation/image": image,
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "prompt": prompt,
            }
            input_hashes = {
                "image": _hash(image),
                "wrist_image": _hash(wrist_image),
                "state": _hash(state),
                "prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            for key, value in input_hashes.items():
                if input_hashes[key] != _text(data, f"{key}_sha256"):
                    raise ValueError(f"input hash mismatch in {path.name}: {key}")

            seed = int(np.asarray(data["noise_seed"]).item())
            arm = _text(data, "arm")
            trace = json.loads(_text(data, "trace_json"))
            if arm == "pcd":
                noise = np.asarray(data["vtcg_noise"], dtype=np.float32)
                replay = policy.infer_pcd(
                    request,
                    seed=seed,
                    noise=noise,
                    n_candidates=int(trace["n_candidates"]),
                    alpha=float(trace["alpha"]),
                    bandwidth_scale=float(trace["bandwidth_scale"]),
                    num_steps=int(trace["num_flow_steps"]),
                    trace_candidates=True,
                )
                actual_arrays = replay["pi05_vtcg_candidates"]
                candidate_errors = {
                    key: _max_abs(actual_arrays[key], data[f"vtcg_{key}"])
                    for key in TRACE_ARRAYS
                }
                index_matches = int(actual_arrays["chosen_index"]) == int(
                    np.asarray(data["vtcg_chosen_index"]).item()
                )
                candidates_match = all(
                    np.allclose(
                        actual_arrays[key],
                        data[f"vtcg_{key}"],
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    for key in TRACE_ARRAYS
                )
            elif arm == "vanilla":
                noise = np.asarray(data["vtcg_noise"], dtype=np.float32)
                original_kwargs = dict(policy.policy._sample_kwargs)
                policy.policy._sample_kwargs = {
                    **original_kwargs,
                    "num_steps": int(trace.get("num_flow_steps") or 10),
                }
                try:
                    replay = policy.policy.infer(request, noise=noise)
                finally:
                    policy.policy._sample_kwargs = original_kwargs
                candidate_errors = {}
                index_matches = True
                candidates_match = True
            else:
                raise ValueError(f"unsupported arm {arm!r} in {path.name}")

            action_error = _max_abs(replay["actions"], data["action_chunk"])
            action_matches = np.allclose(
                replay["actions"], data["action_chunk"], atol=args.atol, rtol=args.rtol
            )
            row = {
                "trace_file": path.name,
                "arm": arm,
                "task_index": int(np.asarray(data["task_index"]).item()),
                "replan_index": int(np.asarray(data["replan_index"]).item()),
                "input_hashes_match": True,
                "candidate_max_abs_error": candidate_errors,
                "chosen_index_matches": index_matches,
                "action_max_abs_error": action_error,
                "passed": bool(candidates_match and index_matches and action_matches),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump({"atol": args.atol, "rtol": args.rtol, "traces": rows}, handle, indent=2)
        handle.write("\n")
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

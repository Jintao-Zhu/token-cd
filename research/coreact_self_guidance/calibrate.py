from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_self_guidance.sampler import reference_trajectory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if (artifact / "phase0_calibration.json").exists():
        raise FileExistsError("refusing to overwrite phase0 calibration")
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    states = [json.loads(line) for line in (artifact / "phase0_state_manifest.jsonl").read_text().splitlines()]
    task2_9_norms: dict[str, list[float]] = {f"{delta:.1f}": [] for delta in np.arange(0.1, 1.0, 0.1)}
    state_hashes = []
    for spec in states:
        env, env_preprocessor, _ = make_task_env("libero_spatial", spec["task_id"], config)
        try:
            env.envs[0].init_state_id = spec["init_state_id"]
            observation, _ = env.reset(seed=spec["reset_seed"])
            batch = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
            state_hashes.append({"state_id": spec["state_id"], "input_sha256": hashlib.sha256("".join(tensor.detach().cpu().contiguous().numpy().tobytes().hex() for tensor in [*batch["images"], batch["state"], batch["lang_tokens"]]).encode()).hexdigest()})
            generator = torch.Generator(device=batch["state"].device).manual_seed(spec["action_noise_seed"] * 1000)
            noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, device=batch["state"].device, dtype=batch["state"].dtype)
            prefix, pad_masks, att_masks = policy.model.embed_prefix(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], state=batch["state"])
            from research.coreact_closed_loop.guidance import _prefix_cache
            cache = _prefix_cache(policy.model, prefix, pad_masks, att_masks)
            _, velocities, _, _ = reference_trajectory(policy.model, pad_masks, cache, noise, num_steps=10)
            velocity = [value[..., :7] for value in velocities]
            for delta_text in task2_9_norms:
                lag = round(float(delta_text) * 10)
                norms = [float(torch.linalg.vector_norm(velocity[i] - velocity[i - lag])) for i in range(lag, 10)]
                task2_9_norms[delta_text].extend(norms)
        finally:
            env.close()
    e1 = workspace / "artifacts/coreact_ensemble_vs_contrast_control_v1_20260809_152414"
    reference_values = []
    for path in e1.glob("episodes/*.json"):
        record = json.loads(path.read_text())
        if record["arm"] == "B_toward_top8":
            for trace in record["replan_traces"]:
                reference_values.extend(trace["clean_minus_masked_l2_norm_pre_clip"])
    if not reference_values:
        raise RuntimeError("E1 reference intervention logs are missing")
    reference_median = float(np.median(reference_values))
    candidates = []
    for delta_text, values in task2_9_norms.items():
        median = float(np.median(values))
        candidates.append({"delta": float(delta_text), "median_l2_norm": median, "ratio_to_e1_masked": median / (reference_median + 1e-12), "n": len(values)})
    initial_selected = [row for row in candidates[:5] if 0.7 <= row["ratio_to_e1_masked"] <= 1.4]
    expanded = not bool(initial_selected)
    selected = initial_selected if initial_selected else [row for row in candidates if 0.7 <= row["ratio_to_e1_masked"] <= 1.4]
    if not selected:
        result = {"pass": False, "reason": "no delta in locked grid matched [0.7,1.4]", "reference_median_l2_norm": reference_median, "candidates": candidates, "expanded_grid": expanded, "state_hashes": state_hashes}
        (artifact / "phase0_calibration.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise RuntimeError("phase0 delta calibration failed")
    chosen = min(selected, key=lambda row: (abs(row["ratio_to_e1_masked"] - 1.0), row["delta"]))
    result = {"pass": True, "reference_source": str(e1), "reference_median_l2_norm": reference_median, "candidates": candidates, "selected": chosen, "expanded_grid": expanded, "state_hashes": state_hashes, "outcome_blind": True}
    (artifact / "phase0_calibration.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    protocol = yaml.safe_load((artifact / "protocol.phase0.yaml").read_text())
    protocol["locked_delta"] = chosen["delta"]
    protocol["phase0_result"] = {"selected_delta": chosen["delta"], "reference_median_l2_norm": reference_median, "selected_ratio": chosen["ratio_to_e1_masked"], "outcome_blind": True}
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")

    e1_manifest = [json.loads(line) for line in (e1 / "episode_manifest.jsonl").read_text().splitlines()]
    bases = {}
    for row in e1_manifest:
        if row["arm"] == "A_vanilla":
            bases[(row["task_id"], row["init_state_id"])] = row
    manifest = []
    for task_id in (4, 7):
        for init_state_id in range(50):
            base = bases[(task_id, init_state_id)]
            for arm in ("A_vanilla", "N0_pure_negative", "W05_shrink", "W15_extrapolate", "W20_extrapolate", "REF_toward_top8"):
                manifest.append({**base, "episode_id": f"task{task_id:02d}__init{init_state_id:02d}__{arm}", "arm": arm, "delta": chosen["delta"], "w": {"A_vanilla": 1.0, "N0_pure_negative": 0.0, "W05_shrink": 0.5, "W15_extrapolate": 1.5, "W20_extrapolate": 2.0, "REF_toward_top8": 0.5}[arm]})
    if len(manifest) != 600:
        raise RuntimeError("600 episode manifest failure")
    with (artifact / "episode_manifest.jsonl").open("x", encoding="utf-8") as stream:
        for row in manifest:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"phase0": result, "episodes": len(manifest), "locked_delta": chosen["delta"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()

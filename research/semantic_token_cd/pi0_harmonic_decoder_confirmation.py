"""Held-out paired decoder confirmation for Pi0 harmonic token guidance.

Three new decoder/control arms use seeds 50--99. The Bridge layer is selected
by a separate offline scan; Fractal Coke retains its previously selected L10.
Existing Vanilla and historical whole-chunk results are used only as paired
references after exact initial state/RGB hash checks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
ARTIFACT = REPO / "artifacts/pi0_harmonic_decoder_confirmation_v1"
TASKS = ("google_robot_pick_coke_can", "widowx_carrot_on_plate")
SEEDS = tuple(range(50, 100))
ARMS = {
    "positive24_nocontrast": "positive_coordinate",
    "harmonic_whole_chunk": "whole_chunk",
    "harmonic_official_independent": "official_coordinate_independent",
}
SOURCE = {
    "google_robot_pick_coke_can": {
        "root": REPO / "artifacts/pi0_fractal_l10_matched_pcd_google800_v1/episodes/google_robot_pick_coke_can",
        "vanilla": "vanilla",
        "matched": "l10_matched_pcd",
    },
    "widowx_carrot_on_plate": {
        "root": REPO / "artifacts/pi0_l10_token_pcd_remaining5_rtfull_0_99_v1/episodes/widowx_carrot_on_plate",
        "vanilla": "vanilla",
        "matched": "l10_token_pcd",
    },
}
PROTOCOL = "PI0_HARMONIC_DECODER_CONFIRMATION_V1"
FRACTAL_LAYER = 10
BRIDGE_LAYER_SCAN = REPO / "artifacts/pi0_bridge_carrot_layer_selection_v1/LAYER_SCAN_RESULTS.json"


def source_row(task: str, arm: str, seed: int) -> dict:
    root = SOURCE[task]["root"]
    p = root / arm / f"episode_{seed:03d}_summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"required paired reference is missing: {p}")
    return json.loads(p.read_text())


def selector_signature(row: dict) -> dict:
    trace = row.get("attention_trace", [])
    if not trace:
        raise RuntimeError("source matched episode has no attention trace")
    item = trace[0]
    return {
        "attention_sha256": item.get("attention_sha256"),
        "m_t": item.get("m_t"),
        "selected_token_ids": item.get("selected_token_ids"),
        "matched_groups": item.get("matched_groups"),
    }


def write_lock(artifact: Path, bridge_layer: int) -> None:
    from research.semantic_token_cd.pi0_l10_pcd_remaining5_rollout import (
        FRACTAL_CKPT, BRIDGE_CKPT, CANONICAL_SNAPSHOTS,
    )

    payload = {
        "protocol_id": PROTOCOL,
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "new_arms": ARMS,
        "reused_reference_arms": {t: {"Vanilla": SOURCE[t]["vanilla"], "historical_whole_chunk": SOURCE[t]["matched"]} for t in TASKS},
        "reference_roots": {t: str(SOURCE[t]["root"]) for t in TASKS},
        "canonical_snapshots": str(CANONICAL_SNAPSHOTS),
        "checkpoints": {TASKS[0]: str(FRACTAL_CKPT), TASKS[1]: str(BRIDGE_CKPT)},
        "renderer": {"shader": "rt", "depth": 8, "spp": 32, "denoiser": True, "offscreen_only": True},
        "layer_index_0based_by_task": {TASKS[0]: FRACTAL_LAYER, TASKS[1]: bridge_layer},
        "bridge_layer_selection": str(BRIDGE_LAYER_SCAN),
        "matched_budget": "canonical 8-way KMeans/source-target entity union; L10 top-m tokens",
        "negative": "16x16 four-neighbour harmonic reconstruction at beta=0",
        "candidate_count": 24,
        "alpha": 0.2,
        "candidate_seeds": "seed*1009+replan; common noise except official_independent, which uses independent positive/negative noise",
        "contrasts": {
            "positive24_nocontrast": "coordinate-wise KDE candidate selection with positive samples as both density references, alpha=0",
            "harmonic_whole_chunk": "historical whole-chunk KDE likelihood-ratio decoder, selected positive candidate clipped per deployed method",
            "harmonic_official_independent": "coordinate-wise KDE-PM decoder with independent positive/negative noise and post-decode clipping",
        },
    }
    artifact.mkdir(parents=True, exist_ok=True)
    path = artifact / "CONFIG_LOCK.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"CONFIG_LOCK differs from requested protocol: {path}")
    else:
        path.write_text(encoded)


def parse_seed_spec(spec: str | None) -> tuple[int, ...]:
    if spec is None:
        return SEEDS
    values: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid descending seed range: {item}")
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    seeds = tuple(sorted(values))
    if not seeds or any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"seed spec must select values in 50..99, got {spec!r}")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--seeds", help="comma-separated seeds/ranges, e.g. 17-24,30,32-36")
    args = parser.parse_args()
    artifact_arg = args.artifact if args.artifact.is_absolute() else (REPO / args.artifact)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.gpu}")

    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "checkpoints/pi0"))
    sys.path[:0] = [str(ROOT / "open-pi-zero"), str(REPO / "task1/shim_site"), str(REPO)]
    os.chdir(ROOT / "open-pi-zero")

    import torch
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from research.semantic_token_cd.pi0_fractal_simpler_l2_pcd_rollout import (
        FractalPi0PCD, restore_snapshot, snapshot_sha,
        make_env as make_fractal_env,
    )
    from research.semantic_token_cd.pi0_l10_pcd_remaining5_rollout import (
        CANONICAL_SNAPSHOTS, TASK_CFG, atomic_json, load_model,
        load_snapshot as load_bridge_snapshot, make_env as make_bridge_env,
        run_episode, sha256_file,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    artifact = artifact_arg.resolve()
    if not BRIDGE_LAYER_SCAN.is_file():
        raise FileNotFoundError(f"Bridge layer scan is not complete: {BRIDGE_LAYER_SCAN}")
    bridge_layer = int(json.loads(BRIDGE_LAYER_SCAN.read_text())["selected_single_layer_index_0based"])
    layer = FRACTAL_LAYER if args.task == "google_robot_pick_coke_can" else bridge_layer
    write_lock(artifact, bridge_layer)
    cfg, model, adapter, checkpoint = load_model(args.task, torch.device("cuda:0"))
    checkpoint_sha = sha256_file(checkpoint)
    # Coke references were generated with the Google-run protocol's explicit
    # less-distractor environment and its run-local canonical snapshots. The
    # WidowX references use the shared canonical snapshot collection and the
    # Bridge task environment.
    if args.task == "google_robot_pick_coke_can":
        env, env_id = make_fractal_env(args.task)
        snapshot_root = REPO / "artifacts/pi0_fractal_l10_matched_pcd_google800_v1/snapshots"
    else:
        env, env_id = make_bridge_env(args.task)
        snapshot_root = CANONICAL_SNAPSHOTS
    task_out = artifact / "episodes" / args.task
    seeds = (50,) if args.preflight_only else parse_seed_spec(args.seeds)
    print(json.dumps({"event": "model_ready", "protocol": PROTOCOL, "task": args.task,
                      "gpu": args.gpu, "worker": args.worker_id, "preflight": args.preflight_only,
                      "checkpoint": str(checkpoint), "config": TASK_CFG[args.task][0],
                      "layer": layer, "bridge_layer": bridge_layer}), flush=True)

    try:
        for seed in seeds:
            snapshot_path = snapshot_root / args.task / f"seed_{seed:03d}.pkl"
            with snapshot_path.open("rb") as stream:
                import pickle
                snapshot = pickle.load(stream)
            canonical_hash = snapshot_sha(snapshot)
            ref_v = source_row(args.task, SOURCE[args.task]["vanilla"], seed)
            ref_m = source_row(args.task, SOURCE[args.task]["matched"], seed)
            reference_hashes = {
                (ref_v.get("initial_state_sha256"), ref_v.get("initial_rgb_sha256")),
                (ref_m.get("initial_state_sha256"), ref_m.get("initial_rgb_sha256")),
            }
            if len(reference_hashes) != 1:
                raise RuntimeError(f"source Vanilla/Matched inputs are not paired: task={args.task} seed={seed}")
            expected_state, expected_rgb = next(iter(reference_hashes))
            hashes = []
            paired_new_selector = None

            for arm, decoder in ARMS.items():
                summary_path = task_out / arm / f"episode_{seed:03d}_summary.json"
                if summary_path.exists():
                    prior = json.loads(summary_path.read_text())
                    if (prior.get("initial_state_sha256"), prior.get("initial_rgb_sha256")) == (expected_state, expected_rgb):
                        continue
                    summary_path.unlink()

                obs, state_hash, rgb_hash = restore_snapshot(
                    env, seed, snapshot, get_image_from_maniskill2_obs_dict
                )
                if (state_hash, rgb_hash) != (expected_state, expected_rgb):
                    raise RuntimeError(
                        f"reference input mismatch task={args.task} seed={seed}: "
                        f"new={(state_hash,rgb_hash)} ref={(expected_state,expected_rgb)}"
                    )
                hashes.append((state_hash, rgb_hash))
                if args.preflight_only:
                    print(json.dumps({"event": "preflight_pair_ok", "task": args.task,
                                      "seed": seed, "arm": arm, "state_sha256": state_hash,
                                      "rgb_sha256": rgb_hash, "layer": layer}), flush=True)
                    continue
                adapter.reset()
                instruction = env.unwrapped.get_language_instruction()
                pcd = FractalPi0PCD(model, adapter, candidates=24, alpha=0.2, layer=layer, decoder=decoder)
                result = run_episode(env, cfg, model, adapter, pcd, obs, instruction,
                                     args.task, seed, arm, torch.device("cuda:0"))
                trace = result["trace"]
                if not trace:
                    raise RuntimeError(f"missing intervention trace: {arm} seed={seed}")
                actual_selector = selector_signature({"attention_trace": trace})
                if layer == FRACTAL_LAYER and args.task == "google_robot_pick_coke_can":
                    expected_selector = selector_signature(ref_m)
                    if actual_selector != expected_selector:
                        raise RuntimeError(f"first-replan Fractal L10 selector mismatch task={args.task} seed={seed} arm={arm}: new={actual_selector} source={expected_selector}")
                if paired_new_selector is None:
                    paired_new_selector = actual_selector
                elif actual_selector != paired_new_selector:
                    raise RuntimeError(f"new-arm selector mismatch task={args.task} seed={seed} arm={arm}: new={actual_selector} paired={paired_new_selector}")
                row = {
                    "protocol_id": PROTOCOL,
                    "task": args.task,
                    "environment_id": env_id,
                    "seed": seed,
                    "arm": arm,
                    "decoder": decoder,
                    "success": bool(result["success"]),
                    "control_steps": result["steps"],
                    "replans": result["replans"],
                    "runtime_seconds": result["runtime_seconds"],
                    "gpu": args.gpu,
                    "worker_id": args.worker_id,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "config": TASK_CFG[args.task][0],
                    "renderer": {"shader": "rt", "depth": 8, "spp": 32, "denoiser": True},
                    "canonical_snapshot_sha256": canonical_hash,
                    "initial_state_sha256": state_hash,
                    "initial_rgb_sha256": rgb_hash,
                    "reference_vanilla_artifact": str(SOURCE[args.task]["root"] / SOURCE[args.task]["vanilla"] / f"episode_{seed:03d}_summary.json"),
                    "reference_whole_chunk_artifact": str(SOURCE[args.task]["root"] / SOURCE[args.task]["matched"] / f"episode_{seed:03d}_summary.json"),
                    "prompt_attention_layer_0based": layer,
                    "bridge_layer_selection_artifact": str(BRIDGE_LAYER_SCAN),
                    "attention_trace": trace,
                    "last_info": result["last_info"],
                }
                atomic_json(summary_path, row)
                print(json.dumps({"task": args.task, "seed": seed, "arm": arm,
                                  "success": row["success"], "replans": row["replans"],
                                  "seconds": round(row["runtime_seconds"], 1),
                                  "worker": args.worker_id}), flush=True)

            if hashes and len(set(hashes)) != 1:
                raise RuntimeError(f"new-arm initial inputs are not paired: task={args.task} seed={seed}")

    finally:
        env.close()
    print(json.dumps({"event": "worker_complete", "worker": args.worker_id,
                      "task": args.task, "gpu": args.gpu, "preflight": args.preflight_only}), flush=True)


if __name__ == "__main__":
    main()

"""SpatialVLA rollout runner for the 4,700-episode three-study batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.semantic_token_cd import spatialvla_l10_matched_rtfull7_rollout as core


TASKS = core.TASKS
DRAWERS = ("google_robot_open_drawer", "google_robot_close_drawer")
REMAINING7 = tuple(t for t in TASKS if t not in DRAWERS)
ALL_SEEDS_300 = tuple(range(300))
EXTENSION_SEEDS = tuple(range(100)) + tuple(range(200, 300))
SNAPSHOTS = Path(__file__).resolve().parents[2] / "artifacts/vanilla_recon_shr_canonical_0_299_v2"

STUDIES = {
    "drawers_rt_depth0": {
        "protocol": "SPATIALVLA_DRAWERS_RT_DEPTH0_VANILLA_L10_SEEDS100_199_V1",
        "tasks": DRAWERS, "seeds": tuple(range(100, 200)),
        "arms": ("vanilla", "l10_matched"),
        "depths": {t: 0 for t in DRAWERS},
        "reference": "simpler_rt_depth0_drawers_0_299_v1",
    },
    "remaining7_rtfull_vanilla": {
        "protocol": "SPATIALVLA_REMAINING7_RTFULL_VANILLA_SEEDS100_199_V1",
        "tasks": REMAINING7, "seeds": tuple(range(100, 200)),
        "arms": ("vanilla",),
        "depths": {t: 8 for t in REMAINING7},
        "reference": "simpler_rt_full_7task_4200_v1",
    },
    "remaining7_rtfull_expand300": {
        "protocol": "SPATIALVLA_L10_VANILLA_REMAINING7_RTFULL_SEEDS0_299_V1",
        "tasks": REMAINING7, "seeds": ALL_SEEDS_300,
        "arms": ("vanilla", "l10_matched"),
        "depths": {t: 8 for t in REMAINING7},
        "reference": "simpler_rt_full_7task_4200_v1",
    },
    "expand9_to300": {
        "protocol": "SPATIALVLA_L10_VANILLA_NINE_TASKS_300_SEEDS_V1",
        "tasks": TASKS, "seeds": EXTENSION_SEEDS,
        "arms": ("vanilla", "l10_matched"),
        "depths": {t: (8 if t in DRAWERS else 0) for t in TASKS},
        "reference": None,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", choices=tuple(STUDIES), required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, default=SNAPSHOTS)
    ap.add_argument("--reference-artifact", type=Path)
    ap.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--model-setup", choices=("google_robot", "widowx_bridge"), required=True)
    args = ap.parse_args()

    study = STUDIES[args.study]
    task_list = tuple(t for t in args.tasks.split(",") if t)
    seeds = core.parse_seeds(args.seeds)
    if not task_list or len(set(task_list)) != len(task_list):
        raise ValueError(f"empty or duplicate tasks: {task_list}")
    if any(t not in study["tasks"] for t in task_list):
        raise ValueError(f"worker tasks are outside {args.study}: {task_list}")
    if any(s not in study["seeds"] for s in seeds):
        raise ValueError(f"worker seeds are outside {args.study}: {seeds}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"empty or duplicate seeds: {seeds}")
    if any(core.POLICY_SETUP[t] != args.model_setup for t in task_list):
        raise ValueError(f"tasks {task_list} do not match policy setup {args.model_setup}")

    reference = args.reference_artifact.resolve() if args.reference_artifact else None
    if study["reference"]:
        expected = Path(__file__).resolve().parents[2] / "artifacts" / study["reference"]
        if reference is None:
            reference = expected
        if reference != expected.resolve():
            raise ValueError(f"{args.study} reference must be {expected}")
    elif reference is not None:
        raise ValueError(f"{args.study} must not use an external paired reference")

    core.ARMS = study["arms"]
    core.DEPTH = study["depths"]
    core.PROTOCOL = study["protocol"]

    # Reuse the audited inference/rollout implementation while supplying a
    # frozen study manifest. main() expects these argparse attributes.
    import sys
    sys.argv = [sys.argv[0], "--artifact", str(args.artifact),
                "--snapshot-artifact", str(args.snapshot_artifact),
                "--gpu", str(args.gpu), "--worker-id", args.worker_id,
                "--tasks", ",".join(task_list), "--seeds", args.seeds,
                "--model-setup", args.model_setup]
    if reference is not None:
        sys.argv.extend(("--reference-artifact", str(reference)))
    core.main = _make_study_main(core.main, study, args, reference, task_list, seeds)
    core.main()


def _make_study_main(original_main, study, args, reference, tasks, seeds):
    """Adapt legacy runner's fixed manifest checks/config lock for this study."""
    # The core entry point is invoked through a temporary CLI-compatible shim:
    # patch its parser and run its body with the study's immutable metadata.
    # Keep all policy math and environment code in the already audited runner.
    import inspect
    import textwrap
    source = inspect.getsource(original_main)
    source = source.replace('ap.add_argument("--reference-artifact", type=Path, required=True,',
                            'ap.add_argument("--reference-artifact", type=Path, default=None,')
    source = source.replace('ap.add_argument("--preflight", action="store_true", help="one paired seed and decoder parity audit")',
                            'ap.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)')
    source = source.replace('if not TASK_CD or any(t not in RTFULL7_TASKS for t in TASK_CD):',
                            'if not TASK_CD or any(t not in TASKS for t in TASK_CD):')
    source = source.replace('seeds = [100] if args.preflight else parse_seeds(args.seeds)\n    if any(seed < 100 or seed > 199 for seed in seeds):\n        raise ValueError(f"this protocol is locked to seeds 100-199, got {seeds}")',
                            f'seeds = parse_seeds(args.seeds)\n    if any(seed not in {list(study["seeds"])} for seed in seeds):\n        raise ValueError(f"seeds outside locked study manifest: {{seeds}}")')
    start = source.index('    config = {')
    end = source.index('    lock = artifact / "CONFIG_LOCK.json"', start)
    config_block = '''    config = {
        "protocol_id": PROTOCOL,
        "study_id": "''' + study["protocol"] + '''",
        "tasks": ''' + repr(list(study["tasks"])) + ''',
        "seeds_by_task": {t: ''' + repr(list(study["seeds"])) + ''' for t in ''' + repr(study["tasks"]) + '''},
        "snapshot_artifact": str(args.snapshot_artifact.resolve()),
        "paired_reference_artifact": ''' + (repr(str(reference)) if reference else "None") + ''',
        "arms": ''' + repr(list(study["arms"])) + ''',
        "models_by_policy_setup": {k: str(v) for k, v in MODEL_BY_SETUP.items()},
        "policy_setup_by_task": POLICY_SETUP,
        "renderer": {t: {"backend": "SAPIEN RT", "depth": DEPTH[t], "spp": 32, "denoiser": True} for t in ''' + repr(study["tasks"]) + '''},
        "pairing": "canonical snapshots; exact same-seed Vanilla reference for studies 1/2; paired arm initial hashes for extension",
        "selector": "SpatialVLA post-softmax layer-10 instruction-to-image attention; canonical dynamic Matched m",
        "negative": "16x16 4-neighbor harmonic reconstruction beta=0 on post-projector/post-Ego3D features",
        "contrastive": "greedy full-vocabulary scores (1+0.5)*positive - 0.5*negative",
        "action_chunk_size": 4, "lambda": 0.5,
        "study_seed_manifest": "''' + args.study + '''",
    }
'''
    source = source[:start] + config_block + source[end:]
    source = source.replace('    reference_artifact = args.reference_artifact.resolve()\n',
                            '    reference_artifact = args.reference_artifact.resolve() if args.reference_artifact else None\n')
    source = source.replace('                reference_path = (reference_artifact / "episodes" / task / "vanilla"\n                                  / f"episode_{seed:03d}_summary.json")',
                            '                reference_path = ((reference_artifact / "episodes" / task / "vanilla"\n                                   / f"episode_{seed:03d}_summary.json") if reference_artifact is not None else None)')
    source = source.replace('                if not reference_path.is_file():\n                    raise FileNotFoundError(f"paired RT-full Vanilla reference missing: {reference_path}")\n                reference = json.loads(reference_path.read_text())\n                renderer_ref = reference.get("renderer", "")\n                if isinstance(renderer_ref, dict):\n                    is_depth8 = (renderer_ref.get("depth") == 8\n                                 or renderer_ref.get("rt_max_path_depth") == 8)\n                else:\n                    is_depth8 = "depth8" in str(renderer_ref).lower()\n                if not is_depth8:\n                    raise RuntimeError(f"reference is not RT-full depth 8: {reference_path}")',
                            '                reference = None\n                if reference_artifact is not None:\n                    if not reference_path.is_file():\n                        raise FileNotFoundError(f"paired Vanilla reference missing: {reference_path}")\n                    reference = json.loads(reference_path.read_text())\n                    renderer_ref = reference.get("renderer", "")\n                    expected_depth = DEPTH[task]\n                    if isinstance(renderer_ref, dict):\n                        ref_depth = renderer_ref.get("depth", renderer_ref.get("rt_max_path_depth"))\n                    else:\n                        text_ref = str(renderer_ref).lower()\n                        ref_depth = 8 if "depth8" in text_ref or "full" in text_ref else (0 if "depth0" in text_ref else None)\n                    if ref_depth != expected_depth:\n                        raise RuntimeError(f"reference renderer depth {ref_depth} != expected {expected_depth}: {reference_path}")')
    source = source.replace('                if args.preflight and task == TASK_CD[0]:\n',
                            '                if False:\n')
    source = source.replace('                    if reference.get("canonical_snapshot_sha256") != canonical_sha:',
                            '                    if reference is not None and reference.get("canonical_snapshot_sha256") != canonical_sha:')
    source = source.replace('                    if reference.get("initial_state_sha256") != state_hash:',
                            '                    if reference is not None and reference.get("initial_state_sha256") != state_hash:')
    source = source.replace('                    if reference.get("initial_rgb_sha256") != rgb_hash:',
                            '                    if reference is not None and reference.get("initial_rgb_sha256") != rgb_hash:')
    source = source.replace('                        env, obs, arm, instruction, seed, single_step=args.preflight\n',
                            '                        env, obs, arm, instruction, seed, single_step=False\n')
    namespace = dict(original_main.__globals__)
    exec(compile(source, inspect.getsourcefile(original_main), "exec"), namespace)
    return namespace["main"]


if __name__ == "__main__":
    main()

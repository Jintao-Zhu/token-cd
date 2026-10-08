#!/usr/bin/env python3
"""Run one pinned SpatialVLA Vanilla episode in SIMPLER without changing shared envs."""

from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace


WORKSPACE = Path(__file__).resolve().parents[2]
MODEL_DIR = WORKSPACE / "checkpoints/spatialvla-4b-224-sft-fractal"
SPATIALVLA_ENV_REPO = WORKSPACE / "SpatialVLA-SimplerEnv"
MANISKILL_REPO = SPATIALVLA_ENV_REPO / "ManiSkill2_real2sim"
TRANSFORMERS_OVERLAY = WORKSPACE / "spatialvla-runtime-overlay-20260928"
BASE_PYTHON = WORKSPACE / "openvla-ar-h100/bin/python"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True,
                        help="Use one of the previously authorized SIMPLER GPUs (1-3).")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--render-depth", type=int, choices=(0, 8), default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run-imports", action="store_true",
                        help="Check Python/model/environment imports without initializing a GPU or renderer.")
    return parser.parse_args()


def _bootstrap_paths() -> None:
    for path in (TRANSFORMERS_OVERLAY, SPATIALVLA_ENV_REPO, MANISKILL_REPO):
        if not path.exists():
            raise FileNotFoundError(f"Required SpatialVLA path is missing: {path}")
        sys.path.insert(0, str(path))


def _tensorflow_memory_stub() -> None:
    # main_inference imports TensorFlow only to cap its allocator; SpatialVLA itself
    # does not need TensorFlow for inference. Avoid adding TensorFlow to the shared env.
    tensorflow = ModuleType("tensorflow")
    tensorflow.config = SimpleNamespace(list_physical_devices=lambda *_args, **_kwargs: [])
    sys.modules["tensorflow"] = tensorflow


def _patch_rt_depth(depth: int) -> None:
    from mani_skill2_real2sim.envs.sapien_env import BaseEnv

    original_init = BaseEnv.__init__

    def init_with_pinned_rt_depth(self, *args, **kwargs):
        if kwargs.get("shader_dir") == "rt":
            render_config = dict(kwargs.get("render_config") or {})
            render_config["rt_max_path_depth"] = depth
            kwargs["render_config"] = render_config
        return original_init(self, *args, **kwargs)

    BaseEnv.__init__ = init_with_pinned_rt_depth


def _patch_gymnasium_wrapper_forwarding() -> None:
    """Restore the legacy custom-attribute forwarding used by this SIMPLER fork."""
    import gymnasium
    from gymnasium.core import Wrapper

    if hasattr(Wrapper, "__getattr__"):
        return

    def forward_custom_attribute(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            inner_env = object.__getattribute__(self, "env")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(inner_env, name)

    Wrapper.__getattr__ = forward_custom_attribute


def main() -> None:
    args = _arguments()
    _bootstrap_paths()
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["PYTHONUNBUFFERED"] = "1"

    import numpy as np
    import torch
    # Eagerly initialize Dynamo and the multimodal auto classes before SIMPLER's
    # optional policy imports; this avoids a lazy torchvision/Dynamo import cycle
    # in the shared PyTorch 2.7 + torchvision 0.22 runtime.
    import torch._dynamo  # noqa: F401
    import transformers
    from transformers import AutoModel, AutoProcessor  # noqa: F401

    if transformers.__version__ != "4.47.0":
        raise RuntimeError(f"SpatialVLA requires its isolated Transformers 4.47.0 overlay, got {transformers.__version__}")

    if args.dry_run_imports:
        import simpler_env
        import mani_skill2_real2sim
        from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference

        print(json.dumps({
            "dry_run_imports": "PASS",
            "python": sys.executable,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "simpler_env": simpler_env.__file__,
            "mani_skill2_real2sim": mani_skill2_real2sim.__file__,
            "spatialvla_policy": SpatialVLAInference.__module__,
        }, indent=2))
        return

    _tensorflow_memory_stub()

    if not MODEL_DIR.joinpath("model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"SpatialVLA weights have not been downloaded: {MODEL_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; no renderer or episode was started.")
    try:
        torch.cuda.init()
        torch.empty(1, device="cuda")
    except Exception as exc:
        raise RuntimeError(
            f"Physical GPU {args.gpu} failed a CUDA allocation preflight; no SIMPLER episode was started."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _patch_rt_depth(args.render_depth)
    _patch_gymnasium_wrapper_forwarding()

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = WORKSPACE / "token-cd/artifacts/spatialvla_simpler_vanilla_smoke" / f"seed{args.seed}_{stamp}"
    else:
        output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    overlay = MANISKILL_REPO / "data/real_inpainting/google_coke_can_real_eval_1.png"
    if not overlay.is_file():
        raise FileNotFoundError(f"SIMPLER image overlay is missing: {overlay}")
    manifest = {
        "status": "RUNNING",
        "arm": "vanilla",
        "benchmark": "SIMPLER",
        "task": "google_robot_pick_coke_can",
        "seed": args.seed,
        "physical_gpu": args.gpu,
        "model": str(MODEL_DIR),
        "renderer": "RT",
        "rt_max_path_depth": args.render_depth,
        "max_episode_steps": args.max_episode_steps,
        "prompt": "environment language instruction",
        "note": "One fixed-layout vanilla smoke episode; not a success-rate estimate.",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    argv = [
        str(SPATIALVLA_ENV_REPO / "simpler_env/main_inference.py"),
        "--policy-model", "spatialvla",
        "--policy-setup", "google_robot",
        "--ckpt-path", str(MODEL_DIR),
        "--env-name", "GraspSingleOpenedCokeCanInScene-v0",
        "--scene-name", "google_pick_coke_can_1_v4",
        "--robot", "google_robot_static",
        "--control-freq", "3",
        "--sim-freq", "513",
        "--max-episode-steps", str(args.max_episode_steps),
        "--enable-raytracing",
        "--rgb-overlay-path", str(overlay),
        "--robot-init-x-range", "0.35", "0.35", "1",
        "--robot-init-y-range", "0.20", "0.20", "1",
        "--robot-init-rot-quat-center", "0", "0", "0", "1",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--obj-init-x-range", "-0.22", "-0.22", "1",
        "--obj-init-y-range", "0.20", "0.20", "1",
        "--additional-env-build-kwargs", "upright=True", "urdf_version=None",
        "--action-ensemble-temp", "-0.8",
        "--logging-dir", str(output),
    ]
    sys.argv = argv
    print(json.dumps({**manifest, "status": "STARTING", "argv": argv}, indent=2), flush=True)
    try:
        runpy.run_path(str(SPATIALVLA_ENV_REPO / "simpler_env/main_inference.py"), run_name="__main__")
    except BaseException as exc:
        manifest["status"] = "ERROR"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest["status"] = "COMPLETED"
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SPATIALVLA_SIMPLER_SMOKE_COMPLETED output={output}", flush=True)


if __name__ == "__main__":
    main()

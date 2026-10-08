"""Paired OpenVLA SIMPLER visual-shift robustness experiment.

The policy arms share the same canonical simulator snapshot and transformed
RGB for each (task, seed, condition).  Visual OOD edits are renderer/image only;
they do not alter the canonical dynamic simulator state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.semantic_token_cd.distractor_rollout import (  # noqa: E402
    PCD_SOURCE, array_sha256, get_image_from_maniskill2_obs_dict, restore_snapshot,
)
from research.semantic_token_cd.prompt_attn_l11_budget_causal_rollout import (  # noqa: E402
    RT_DEPTH, load_task_reference, make_environment,
)
from research.semantic_token_cd.prompt_attn_shr_rollout import (  # noqa: E402
    _init_common,
)
from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference  # noqa: E402
from research.semantic_token_cd.distractor_policy import AuditedVanillaInference  # noqa: E402
from research.semantic_token_cd.rollout_pilot import wrapped_observation  # noqa: E402
from research.semantic_token_cd.spatial_grid_rollout import flatten_action  # noqa: E402
from research.semantic_token_cd.ood_visual_shift_utils import (  # noqa: E402
    apply_arena_texture_pixels, create_arena_texture,
)

# PCD vendors SAM2 as a nested repository rather than installing it into the
# SIMPLER environment.  Expose that package explicitly for Grounded-SAM PCD.
SAM2_VENDOR_ROOT = PCD_SOURCE / "third_party" / "grounded_sam_2"
if str(SAM2_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_VENDOR_ROOT))

TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
CONDITIONS = ("clean", "distractor", "texture", "illumination", "viewpoint", "composite")
ARMS = ("vanilla", "pcd", "ours")
PROTOCOL = "OPENVLA_OOD_ROBUSTNESS_200_299_V3_ASSET_DISTRACTORS"
LAMBDA = 0.5  # frozen from the global four-task OpenVLA lambda analysis
CAMERA = "overhead_camera"


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _add_render_only_distractors(env, obs, seed: int) -> tuple[list, dict]:
    """Place two visible, collision-free can/bottle props on visible arena/table pixels."""
    import sapien.core as sapien

    scene = env.unwrapped._scene
    camera = env.unwrapped._cameras[CAMERA]
    camera_obs = obs["image"][CAMERA]
    seg = np.asarray(camera_obs["Segmentation"])
    arena = next((a for a in scene.get_all_actors() if getattr(a, "name", "") == "arena"), None)
    if arena is None:
        raise RuntimeError("SAPIEN arena actor needed to place visual distractors")
    arena_id = int(arena.get_id())
    mask = seg[..., 1] == arena_id
    # The Gym observation wrapper drops Position, so read the underlying SAPIEN
    # camera texture directly while retaining the already wrapped segmentation.
    raw = camera.get_images(take_picture=True)
    position = np.asarray(raw["Position"], dtype=np.float64)
    cam2world = np.asarray(camera.get_params()["cam2world_gl"], dtype=np.float64)
    xyz1 = position.reshape(-1, 4) @ cam2world.T
    xyz = (xyz1[:, :3] / np.where(np.abs(xyz1[:, 3:4]) < 1e-9, 1.0, xyz1[:, 3:4])).reshape(*mask.shape, 3)
    finite = np.isfinite(xyz).all(axis=-1) & mask
    h, w = mask.shape
    # Pick visible arena points near the highest broad surface (normally the table),
    # then require a clear local image neighborhood so the new objects are legible.
    zvals = xyz[..., 2][finite]
    if zvals.size < 100:
        raise RuntimeError("too few visible arena pixels to place the distractors")
    import cv2
    # For these fixed Google Robot cameras, the visible worktable occupies the
    # central-lower image. Restrict candidates to that tabletop ROI so a high
    # percentile over the whole arena cannot accidentally select a wall pixel.
    roi = np.zeros_like(mask, dtype=bool)
    # Leave a generous left margin: the camera's GL-to-RGB raster conversion
    # shifts some projected static props by ~40 px near the image edge.
    roi[int(0.20*h):int(0.80*h), int(0.28*w):int(0.88*w)] = True
    candidate_mask = finite & roi
    # Estimate the tabletop plane from the dominant height mode of arena pixels
    # in the central view (the surrounding floor/walls form other height bands).
    z_roi = xyz[..., 2][candidate_mask]
    hist, edges = np.histogram(z_roi, bins=np.arange(-0.5, 2.01, 0.01))
    if not len(hist) or int(hist.max()) < 100:
        raise RuntimeError("could not estimate the tabletop height mode")
    z_mode = float((edges[int(hist.argmax())] + edges[int(hist.argmax()) + 1]) / 2)
    candidate_mask &= np.abs(xyz[..., 2] - z_mode) <= 0.035
    dist = cv2.distanceTransform(candidate_mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.where(candidate_mask & (dist >= 18))
    if len(xs) < 2:
        raise RuntimeError("no two clear arena pixels inside the tabletop image region")
    candidates = [(int(x), int(y), xyz[y, x].copy()) for x, y in zip(xs, ys)
                  if np.isfinite(xyz[y, x]).all()]
    # Search most-separated pairs among clear tabletop pixels; deterministic
    # ordering follows row-major source coordinates.
    selected = [candidates[0]]
    for _ in range(1):
        prev = np.asarray([p[:2] for p in selected], dtype=np.float64)
        score = [min(float(np.linalg.norm(np.asarray(c[:2]) - p)) for p in prev)
                 for c in candidates]
        selected.append(candidates[int(np.argmax(score))])
    chosen = selected
    model_root = ROOT.parent / "SimplerEnv" / "ManiSkill2_real2sim" / "data" / "custom" / "models"
    specs = [
        ("ood_distractor_can", "can", model_root / "baked_opened_7up_can" / "textured_cand1.glb"),
        ("ood_distractor_bottle", "bottle", model_root / "baked_blue_plastic_bottle" / "textured.glb"),
    ]
    actors = []
    positions = []
    from transforms3d.euler import euler2quat
    upright = sapien.Pose([0, 0, 0], euler2quat(np.pi / 2, 0.0, 0.0))
    for (name, kind, mesh_path), (_, _, surface) in zip(specs, chosen):
        if not mesh_path.is_file():
            raise FileNotFoundError(f"semantic OOD asset missing: {mesh_path}")
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(str(mesh_path), pose=upright)
        # The released can mesh is 12.2 cm tall; the bottle mesh is 20.1 cm.
        # After the Y-to-Z rotation their lowest local Z coordinates are the
        # negative half-heights listed here, so each rests on the sampled surface.
        half_height = 0.063 if kind == "can" else 0.087
        actor = builder.build_static(name=name)
        xyz_pos = np.asarray(surface, dtype=np.float64).copy()
        xyz_pos[2] += half_height
        actor.set_pose(sapien.Pose(xyz_pos.tolist()))
        actors.append(actor)
        positions.append({"name": name, "kind": kind, "world_position": xyz_pos.tolist(),
                          "camera_pixel": [chosen[len(actors)-1][0], chosen[len(actors)-1][1]],
                          "asset": str(mesh_path), "collision_shapes": 0})
    return actors, {"objects": positions, "arena_actor_id": arena_id,
                    "object_count": 2, "physics_changed": False,
                    "placement_seed": int(seed), "estimated_table_height": z_mode}


def _remove_distractors(env):
    scene = env.unwrapped._scene
    for actor in list(scene.get_all_actors()):
        if getattr(actor, "name", "") in {"ood_distractor_can", "ood_distractor_bottle"}:
            scene.remove_actor(actor)


def _set_viewpoint(env, original_pose, seed):
    import sapien.core as sapien
    from transforms3d.euler import euler2quat

    # Fixed magnitude; signs are deterministic per seed. Translation is in world x/y.
    signs = np.random.default_rng(np.random.SeedSequence([int(seed), 0xCA6E2A])).choice([-1.0, 1.0], 4)
    yaw = np.deg2rad(5.0 * signs[2])
    pitch = np.deg2rad(3.0 * signs[3])
    dq = euler2quat(0.0, pitch, yaw, axes="sxyz")
    delta = sapien.Pose([0.05 * signs[0], 0.05 * signs[1], 0.0], dq)
    shifted = delta * original_pose
    env.unwrapped._cameras[CAMERA].camera.set_local_pose(shifted)
    return {"translation_m": [float(0.05 * signs[0]), float(0.05 * signs[1]), 0.0],
            "yaw_deg": float(5.0 * signs[2]), "pitch_deg": float(3.0 * signs[3])}


def _transform_image(image, obs, condition, seed, texture):
    out = np.asarray(image).copy()
    camera = obs["image"][CAMERA]
    seg = np.asarray(camera["Segmentation"])
    # The transform receives a private metadata slot set by the rollout; it is not
    # passed into the environment or the policy.
    arena_id = obs.get("_ood_arena_id")
    apply_texture = condition in {"texture", "composite"}
    apply_light = condition in {"illumination", "composite"}
    apply_distractor = condition in {"distractor", "composite"}
    if apply_texture:
        out = apply_arena_texture_pixels(out, seg, texture, int(arena_id))
    if apply_light:
        factor = 0.65 if int(seed) % 2 == 0 else 1.35
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if apply_distractor:
        # Distractors are actual render-only SAPIEN visual actors; no post-render patch.
        pass
    return out


def _with_rgb(obs, image):
    result = dict(obs)
    images = dict(obs["image"])
    camera = dict(images[CAMERA])
    camera["rgb"] = image
    images[CAMERA] = camera
    result["image"] = images
    return result


def _run_episode(env, policy, arm, instruction, obs, condition, seed, texture,
                 pcd_generator=None, max_steps=None):
    from utils import convert_numpy_or_torch_to_python, stat_final, stat_first, summarize
    terminated = truncated = False
    timestep = 0
    infos = []
    image = get_image_from_maniskill2_obs_dict(env, obs)
    image = _transform_image(image, obs, condition, seed, texture)
    while not (terminated or truncated):
        if arm == "pcd":
            contrast = pcd_generator.generate(_with_rgb(obs, image), instruction)
            _raw, actions, _aux = policy.step(image, contrast, instruction,
                                               proprio=obs["agent"]["eef_pos"])
        else:
            _raw, actions, _aux = policy.step(image, None, instruction,
                                               proprio=obs["agent"]["eef_pos"])
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            executed = flatten_action(action)
            if executed.shape != (7,) or not np.isfinite(executed).all():
                raise FloatingPointError(f"invalid action {executed}")
            obs, _reward, _success, truncated, info = env.step(executed)
            obs = dict(obs)
            arena_actor = next((a for a in env.unwrapped._scene.get_all_actors()
                                if getattr(a, "name", "") == "arena"), None)
            obs["_ood_arena_id"] = int(arena_actor.get_id()) if arena_actor else -1
            timestep += 1
            infos.append(convert_numpy_or_torch_to_python(info))
            image = get_image_from_maniskill2_obs_dict(env, obs)
            image = _transform_image(image, obs, condition, seed, texture)
            terminated = bool(action["terminate_episode"][0] > 0)
            if max_steps is not None and timestep >= max_steps:
                truncated = True
                break
            if terminated and not env.unwrapped.is_final_subtask():
                terminated = False
                env.advance_to_next_subtask()
            instruction = env.unwrapped.get_language_instruction()
    result = summarize(infos)
    result.update(stat_first(infos)); result.update(stat_final(infos))
    failure = None if result["success"] else ("environment_time_limit" if truncated else "policy_terminated_without_success")
    return result, timestep, failure


def _config(artifact: Path, snapshot_artifact: Path):
    return {
        "protocol_id": PROTOCOL, "model": "OpenVLA-7B Fractal SIMPLER checkpoint",
        "checkpoint": str(PCD_SOURCE / "pretrained/openvla-7b"),
        "tasks": list(TASKS), "seeds": list(range(200, 300)),
        "conditions": list(CONDITIONS), "arms": list(ARMS), "lambda_ours": LAMBDA,
        "pcd": {"implementation": "local official-reproduction ContrastImageGenerator",
                "mask": "Grounded-SAM2 (GroundingDINO + SAM2)", "inpaint": "LaMa", "alpha": 0.8},
        "ours": {"selector": "L11 prompt-to-visual attention", "budget": "Matched",
                 "counterfactual": "Harmonic", "intervention": "projector", "guided_dims": [0,1,2,3,4,5],
                 "gripper": "clean", "decode": "greedy", "negative_prefix": "clean teacher-forced"},
        "rendering": {t: {"backend": "SAPIEN RT", "depth": int(RT_DEPTH[t]), "spp": 32,
                           "denoiser": True} for t in TASKS},
        "conditions_definition": {
            "clean": "canonical renderer/image untouched",
            "distractor": "2 collision-free render-only SIMPLER assets (7Up can + blue plastic bottle) on visible table/arena; deterministic placements",
            "texture": "fixed held-out procedural wood texture on arena-segmented pixels only",
            "illumination": "post-render RGB gain 0.65 for even seeds, 1.35 for odd seeds, clipped [0,255]",
            "viewpoint": "camera extrinsic shift: translation +/-5cm in world xy, yaw +/-5deg, pitch +/-3deg; deterministic signs",
            "composite": "distractor + texture + illumination + viewpoint"},
        "pairing": "same canonical snapshot and exact condition RGB hash across all 3 arms",
        "snapshot_artifact": str(snapshot_artifact.resolve()),
        "rendering_gpu": "same physical GPU as inference worker for each paired seed",
        "clean_reuse_policy": "rerun all three arms; history is state/instruction audit only because RT RGB differs across GPUs",
    }


def _patch_legacy_lama_imports():
    """Bridge LaMa's old Albumentations symbols without changing installed packages."""
    import importlib.util
    import types
    if importlib.util.find_spec("imgaug") is None:
        # Vendored LaMa imports imgaug for training-only transforms. The OOD
        # PCD arm only loads the frozen inpainting checkpoint, so keep these
        # symbols lazy and fail loudly if an inference path ever tries to use them.
        imgaug = types.ModuleType("imgaug")
        augmenters = types.ModuleType("imgaug.augmenters")
        def _training_only(*args, **kwargs):
            raise RuntimeError("imgaug training augmentation is unavailable in PCD inference")
        augmenters.Affine = _training_only
        augmenters.PerspectiveTransform = _training_only
        imgaug.augmenters = augmenters
        sys.modules["imgaug"] = imgaug
        sys.modules["imgaug.augmenters"] = augmenters
    import kornia
    if not hasattr(kornia, "SamplePadding"):
        from kornia.constants import SamplePadding
        kornia.SamplePadding = SamplePadding
    import albumentations as A
    if not hasattr(A, "DualIAATransform"):
        from albumentations.core.transforms_interface import DualTransform
        from albumentations.core.utils import to_tuple

        class DualIAATransform(DualTransform):
            # LaMa only needs these legacy augmenters to be importable for model
            # loading; inference never constructs its training dataset pipeline.
            def __init__(self, always_apply=False, p=0.5):
                self.always_apply = bool(always_apply)
                super().__init__(p=p)

        A.DualIAATransform = DualIAATransform
        A.to_tuple = to_tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--snapshot-artifact", type=Path, default=ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2")
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--renderer-device", default="cuda:0",
                    help="local CUDA device index used by SAPIEN; inference stays on visible device 0")
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--smoke-steps", type=int, default=0, help="limit rollout steps for preflight only")
    ap.add_argument("--save-audit-images", action="store_true",
                    help="save one identical paired input image per task/seed/condition for visual preflight")
    args = ap.parse_args()
    # Rendering and inference stay on the same physical worker GPU.  This
    # preserves byte-exact within-condition RGB pairing across all three arms.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if int(visible[0]) != args.gpu:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES first device {visible[0]} != inference GPU {args.gpu}")
    render_index = int(args.renderer_device.split(":")[-1])
    if render_index >= len(visible):
        raise RuntimeError(f"renderer device {args.renderer_device} is not in CUDA_VISIBLE_DEVICES={visible}")
    if int(visible[render_index]) != args.gpu:
        raise RuntimeError("renderer must use the same physical GPU as its paired inference worker")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    seeds = []
    for spec in args.seeds.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if "-" in spec:
            lo, hi = (int(x) for x in spec.split("-", 1))
            if hi < lo:
                raise ValueError(f"descending seed range: {spec}")
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(spec))
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed in worker assignment")
    conditions = [c for c in args.conditions.split(",") if c]
    if not seeds or not conditions or any(c not in CONDITIONS for c in conditions):
        raise ValueError("invalid seeds or conditions")
    root, snapshots = args.artifact.resolve(), args.snapshot_artifact.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "CONFIG_LOCK.json"
    config = _config(root, snapshots)
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise RuntimeError("CONFIG_LOCK differs from the frozen OOD protocol")
    _write(lock, config)
    texture_path = create_arena_texture(root / "assets" / "arena_ood_texture.png")
    texture = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(texture_path).convert("RGB"))

    import torch
    from properties import get_policy_config, get_contrast_image_generator_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from contrast_policies.openvla_contrast import OpenVLAContrastInference
    _patch_legacy_lama_imports()
    from contrast_utils import get_contrast_image_generator
    from research.semantic_token_cd.prompt_attn_shr_policy import PromptAttentionSHRInference
    from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json

    env, environment_id = make_environment(args.task, renderer_device=args.renderer_device)
    model_cfg = get_policy_config("openvla", str(PCD_SOURCE / "pretrained/openvla-7b"), args.task, {}, False)
    base = OpenVLAInference(**model_cfg)
    vanilla = copy.copy(base); vanilla.__class__ = AuditedVanillaInference
    vanilla._episode_trace, vanilla._episode_logits = [], []
    pcd = copy.copy(base); pcd.__class__ = OpenVLAContrastInference; pcd.alpha = 0.8
    ours = copy.copy(base); ours.__class__ = PromptAttentionSHRInference
    _init_common(ours, LAMBDA)
    ours.beta = 0.0; ours.selector_mode = "prompt_attention"; ours.attention_layers = (11,)
    ours.selection_count = None; ours.selection_top_p = None; ours.selection_budget_schedule = None
    ours.selection_budget_scale = 1.0; ours.selection_budget_source = "matched"
    ours.selection_budget_label = "l11_matched"; ours.counterfactual_constructor = "harmonic"
    ours.guidance_execution = "contrast"; ours.save_prompt_attention = True
    policies = {"vanilla": vanilla, "pcd": pcd, "ours": ours}
    # Follow the attached protocol: Grounded-SAM2 with LaMa, not the repository's
    # simulator-GT default. This keeps the PCD mask perception path independent.
    pcd_cfg = dict(get_contrast_image_generator_config({}))
    pcd_cfg.update({"env": env, "by": "grounded_sam", "inpaint_mode": "lama"})
    lama_path = str(PCD_SOURCE / "pretrained/big-lama/models/best.ckpt")
    original_torch_load = torch.load
    def _load_lama_compat(f, *a, **kw):
        if os.path.realpath(os.fspath(f)) == os.path.realpath(lama_path): kw.setdefault("weights_only", False)
        return original_torch_load(f, *a, **kw)
    torch.load = _load_lama_compat
    try: pcd_gen = get_contrast_image_generator(pcd_cfg)
    finally: torch.load = original_torch_load

    out_root = root / "episodes" / args.task
    snapshot_dir = snapshots / "snapshots" / args.task
    try:
        for seed in seeds:
            snapshot_file = snapshot_dir / f"seed_{seed:03d}.pkl"
            if not snapshot_file.is_file(): raise FileNotFoundError(snapshot_file)
            snapshot = pickle.loads(snapshot_file.read_bytes())
            ref = load_task_reference(snapshots, args.task, seed)
            canonical = hashlib.sha256(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
            # The canonical snapshot hash function is the repo's stable hash; use it
            # through the loader to compare with reference metadata.
            from research.semantic_token_cd.distractor_rollout import snapshot_sha
            canonical = snapshot_sha(snapshot)
            if ref.get("canonical_snapshot_sha256") != canonical:
                raise RuntimeError(f"canonical snapshot mismatch {args.task}:{seed}")
            expected_state = ref["initial_state_sha256"]
            for condition in conditions:
                pair_states, pair_rgbs = [], []
                perturb_audit = None
                for arm, policy in policies.items():
                    summary_path = out_root / condition / arm / f"episode_{seed:03d}_summary.json"
                    if summary_path.exists():
                        old = json.loads(summary_path.read_text())
                        if old.get("protocol_id") != PROTOCOL: raise RuntimeError(f"foreign output {summary_path}")
                        pair_states.append(old["initial_state_sha256"]); pair_rgbs.append(old["initial_rgb_sha256"])
                        continue
                    _remove_distractors(env)
                    obs, state_sha, _ = restore_snapshot(env, seed, snapshot)
                    if state_sha != expected_state: raise RuntimeError(f"snapshot state mismatch {args.task}:{seed}")
                    # Restore the canonical camera pose before each condition/arm.
                    camera = env.unwrapped._cameras[CAMERA].camera
                    if not hasattr(env.unwrapped, "_ood_camera_pose0"):
                        env.unwrapped._ood_camera_pose0 = env.unwrapped._cameras[CAMERA].camera_cfg.pose
                    camera_pose0 = env.unwrapped._ood_camera_pose0
                    camera.set_local_pose(camera_pose0)
                    clean_obs = wrapped_observation(env)
                    clean_image = get_image_from_maniskill2_obs_dict(env, clean_obs)
                    clean_rgb_sha = _sha(clean_image)
                    before = np.asarray(env.unwrapped.get_state()).copy()
                    audit = {"condition": condition, "distractor": None, "viewpoint": None}
                    if condition in {"viewpoint", "composite"}:
                        audit["viewpoint"] = _set_viewpoint(env, camera_pose0, seed)
                    if condition in {"distractor", "composite"}:
                        base_obs = wrapped_observation(env)
                        actors, distractor_audit = _add_render_only_distractors(env, base_obs, seed)
                        audit["distractor"] = distractor_audit
                        # New static actors need a fresh render plus a blocking
                        # readback before the RGB observation is consumed.
                        env.unwrapped.update_render()
                        env.unwrapped.take_picture()
                        env.unwrapped.get_images()
                        obs = wrapped_observation(env)
                        seg0 = np.asarray(obs["image"][CAMERA]["Segmentation"])[..., 1]
                        actor_visibility = []
                        for actor in actors:
                            yy, xx = np.where(seg0 == int(actor.get_id()))
                            actor_visibility.append({"name": actor.name, "visible_pixels": int(len(xx)),
                                "bbox_xyxy": ([int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
                                               if len(xx) else None)})
                        # This is a visibility check, not a minimum-size filter.
                        # A few RT rasterizations showed a fully in-frame bottle
                        # with 57 visible pixels; rejecting it aborted an otherwise
                        # valid paired seed. Require a clearly non-empty mask while
                        # preserving the deterministic scene placement and RGB.
                        if any(item["visible_pixels"] < 25 for item in actor_visibility):
                            raise RuntimeError(f"OOD distractor is not visibly rendered: {actor_visibility}")
                        audit["distractor"]["actor_visibility"] = actor_visibility
                    else:
                        obs = wrapped_observation(env)
                    post = np.asarray(env.unwrapped.get_state())
                    if not np.array_equal(before, post):
                        raise RuntimeError(f"visual OOD condition changed dynamic state {args.task}:{seed}:{condition}")
                    arena_actor = next((a for a in env.unwrapped._scene.get_all_actors() if getattr(a, "name", "") == "arena"), None)
                    obs = dict(obs); obs["_ood_arena_id"] = int(arena_actor.get_id()) if arena_actor else -1
                    instruction = env.unwrapped.get_language_instruction()
                    if instruction != ref["instruction"]: raise RuntimeError(f"instruction mismatch {args.task}:{seed}")
                    image0 = get_image_from_maniskill2_obs_dict(env, obs)
                    shifted0 = _transform_image(image0, obs, condition, seed, texture)
                    rgb_sha = _sha(shifted0)
                    # RT output is deterministic on a fixed GPU, but is not
                    # byte-identical across GPU models/indices.  The historical
                    # baseline was rendered on GPU 3; the OOD study uses GPUs
                    # 1--5.  We therefore validate the canonical simulator state
                    # against history, rerun *all three* arms in this protocol,
                    # and require exact within-seed/condition RGB pairing below.
                    # Historical RGB equality is retained as an audit field, not
                    # treated as a reason to reject a valid same-GPU paired run.
                    if condition == "clean" and rgb_sha != clean_rgb_sha:
                        raise RuntimeError(f"clean condition changed RGB {args.task}:{seed}")
                    if condition != "clean" and rgb_sha == clean_rgb_sha:
                        raise RuntimeError(f"condition made no RGB change {args.task}:{seed}:{condition}")
                    if args.save_audit_images:
                        from PIL import Image
                        audit_image_path = root / "audit_images" / args.task / f"seed_{seed:03d}_{condition}.png"
                        if not audit_image_path.exists():
                            audit_image_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(shifted0).save(audit_image_path)
                    if arm == "ours":
                        policy.reset(instruction, seed=seed)
                    else:
                        policy.reset(instruction)
                    if hasattr(policy, "_episode_trace"): policy._episode_trace, policy._episode_logits = [], []
                    if arm == "pcd": pcd_gen.reset()
                    started = time.monotonic()
                    result, steps, failure = _run_episode(
                        env, policy, arm, instruction, obs, condition, seed, texture,
                        pcd_generator=pcd_gen if arm == "pcd" else None,
                        max_steps=args.smoke_steps or None)
                    trace = list(getattr(policy, "_episode_trace", []))
                    if arm == "ours" and not trace: raise RuntimeError("ours selector trace empty")
                    if arm == "ours" and not all(t.get("feature_equal") and t.get("non_target_bit_identical")
                                                   and t.get("reconstruction_finite") for t in trace):
                        raise RuntimeError("ours projector/harmonic audit failed")
                    summary = {
                        "protocol_id": PROTOCOL, "task": args.task, "seed": seed,
                        "condition": condition, "arm": arm, "success": bool(result["success"]),
                        "result": result, "steps": steps, "failure_reason": failure,
                        "runtime_seconds": time.monotonic() - started, "gpu_id": args.gpu,
                        "worker_id": args.worker_id, "environment_id": environment_id,
                        "renderer": {"backend": "SAPIEN RT", "depth": int(RT_DEPTH[args.task]), "spp": 32,
                                     "denoiser": True, "physical_gpu": args.gpu},
                        "lambda": LAMBDA if arm == "ours" else (0.8 if arm == "pcd" else 0.0),
                        "canonical_snapshot_sha256": canonical,
                        "initial_state_sha256": state_sha, "initial_rgb_sha256": rgb_sha,
                        "base_initial_rgb_sha256": clean_rgb_sha, "instruction": instruction,
                        "historical_reference_rgb_sha256": ref.get("initial_rgb_sha256"),
                        "historical_reference_rgb_match": rgb_sha == ref.get("initial_rgb_sha256"),
                        "shift_audit": audit, "selector_trace": trace,
                        "attention_layers": [11] if arm == "ours" else [],
                    }
                    atomic_json(summary_path, summary)
                    pair_states.append(state_sha); pair_rgbs.append(rgb_sha)
                    perturb_audit = audit
                    print(json.dumps({"task": args.task, "seed": seed, "condition": condition,
                                      "arm": arm, "success": summary["success"], "steps": steps,
                                      "seconds": round(summary["runtime_seconds"], 1)}, sort_keys=True), flush=True)
                    if arm == "pcd": pcd_gen.reset()
                if len(set(pair_states)) != 1 or len(set(pair_rgbs)) != 1:
                    raise RuntimeError(f"paired state/RGB hash mismatch {args.task}:{seed}:{condition}")
    finally:
        try: pcd_gen.reset()
        except Exception: pass
        env.close()


if __name__ == "__main__":
    main()

"""Controlled, physics-preserving visual shifts for the SIMPLER OOD card."""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np


def reset_sim_visual_shift(env) -> None:
    """Return renderer-only OOD edits to this environment's captured baseline."""
    unwrapped = env.unwrapped
    scene = unwrapped._scene
    if not hasattr(unwrapped, "_ood_baseline_ambient"):
        unwrapped._ood_baseline_ambient = np.asarray(scene.ambient_light, dtype=np.float32).copy()
    scene.set_ambient_light(np.asarray(unwrapped._ood_baseline_ambient).tolist())
    old = next((a for a in scene.get_all_actors()
                if getattr(a, "name", "") == "ood_render_only_distractor"), None)
    if old is not None:
        scene.remove_actor(old)


def apply_sim_visual_shift(env, shift: str, texture_path: str | Path | None = None,
                           observation: dict | None = None) -> dict:
    """Modify only renderer state; never add collision geometry or touch task state."""
    scene = env.unwrapped._scene
    audit = {"shift": shift, "physics_changed": False}
    if shift == "lighting":
        before = np.asarray(scene.ambient_light, dtype=np.float32).copy()
        scene.set_ambient_light([0.55, 0.55, 0.55])
        audit.update({"ambient_before": before.tolist(), "ambient_after": [0.55, 0.55, 0.55]})
    elif shift == "texture":
        if texture_path is None or not Path(texture_path).is_file():
            raise FileNotFoundError(f"arena texture file is missing: {texture_path}")
        arena = next((a for a in scene.get_all_actors() if getattr(a, "name", "") == "arena"), None)
        if arena is None:
            raise RuntimeError("SAPIEN static arena actor was not found")
        # The bundled arena mesh has no usable UVs in this renderer: assigning a
        # texture to its material produced bit-identical RGB. The rollout applies
        # the fixed texture only to pixels whose renderer segmentation ID is the
        # arena actor; no target/robot pixels or simulator state are changed.
        audit.update({"texture_path": str(Path(texture_path).resolve()),
                      "application": "segmentation-masked RGB texture on arena pixels",
                      "arena_actor_id": int(arena.get_id())})
    elif shift == "distractor":
        if observation is None:
            raise ValueError("distractor placement requires the baseline observation/segmentation")
        arena = next((a for a in scene.get_all_actors() if getattr(a, "name", "") == "arena"), None)
        if arena is None:
            raise RuntimeError("SAPIEN static arena actor was not found")
        seg = np.asarray(observation["image"]["overhead_camera"]["Segmentation"])
        arena_mask = seg[..., 1] == int(arena.get_id())
        candidates = [(550.0, 82.0), (90.0, 82.0), (550.0, 430.0), (90.0, 430.0)]
        ranked = []
        for cu, cv in candidates:
            ui, vi = int(cu), int(cv)
            patch = arena_mask[max(0, vi-14):min(arena_mask.shape[0], vi+14),
                               max(0, ui-14):min(arena_mask.shape[1], ui+14)]
            ranked.append((float(patch.mean()), cu, cv))
        arena_fraction, u, v = max(ranked)
        if arena_fraction < 0.999:
            raise RuntimeError(f"no collision-free arena image patch for distractor; best arena fraction={arena_fraction:.3f}")
        audit.update({"application": "segmentation-masked render-only RGB distractor; no simulator actor or physics",
                      "distractor_camera_pixel_target": [u, v],
                      "distractor_baseline_arena_fraction_in_28px_window": arena_fraction,
                      "distractor_patch_size_px": [24, 24], "distractor_has_collision": False,
                      "arena_actor_id": int(arena.get_id())})
    else:
        raise ValueError(f"unknown visual shift: {shift}")
    return audit


def create_arena_texture(path: str | Path, size: int = 256) -> Path:
    """Write one fixed held-out, deterministic low-frequency wood-like texture."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return path
    x = np.linspace(0, 8 * np.pi, size, dtype=np.float32)[None, :]
    y = np.linspace(0, 2 * np.pi, size, dtype=np.float32)[:, None]
    rng = np.random.default_rng(20261002)
    noise = rng.normal(0, 2.0, (size, size)).astype(np.float32)
    grain = 0.5 + 0.16 * np.sin(x + 0.35 * np.sin(y)) + 0.05 * np.sin(3 * x + y)
    grain = np.clip(grain + noise / 255.0, 0.0, 1.0)
    rgb = np.stack([
        105 + 78 * grain,
        72 + 67 * grain,
        44 + 47 * grain,
    ], axis=-1).clip(0, 255).astype(np.uint8)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.png")
    Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)
    return path


def apply_arena_texture_pixels(image: np.ndarray, segmentation: np.ndarray,
                               texture: np.ndarray, arena_actor_id: int) -> np.ndarray:
    """Apply a held-out texture to arena-only pixels, preserving all other pixels."""
    image = np.asarray(image)
    segmentation = np.asarray(segmentation)
    if image.ndim != 3 or image.shape[-1] != 3 or segmentation.ndim != 3 or segmentation.shape[-1] < 2:
        raise ValueError(f"unexpected image/segmentation shapes: {image.shape}, {segmentation.shape}")
    if image.shape[:2] != segmentation.shape[:2]:
        raise ValueError("RGB and segmentation dimensions differ")
    mask = segmentation[..., 1] == int(arena_actor_id)
    if not mask.any():
        raise RuntimeError("arena segmentation mask is empty")
    pattern = np.asarray(texture)
    if pattern.ndim != 3 or pattern.shape[-1] < 3:
        raise ValueError(f"invalid arena texture image: {pattern.shape}")
    yy = np.arange(image.shape[0])[:, None] % pattern.shape[0]
    xx = np.arange(image.shape[1])[None, :] % pattern.shape[1]
    tiled = pattern[yy, xx, :3]
    out = image.copy()
    # Moderate blend keeps geometry and segmentation cues legible while creating
    # a clear, deterministic material-style shift in the arena context.
    mixed = 0.45 * image.astype(np.float32) + 0.55 * tiled.astype(np.float32)
    out[mask] = np.clip(mixed[mask], 0, 255).astype(image.dtype)
    if not np.array_equal(out[~mask], image[~mask]):
        raise RuntimeError("arena texture changed non-arena pixels")
    return out


def apply_static_distractor_pixels(image: np.ndarray, segmentation: np.ndarray,
                                   arena_actor_id: int, center_xy: tuple[float, float],
                                   size: int = 24) -> np.ndarray:
    """Draw one colorful, fixed render-only distractor wholly on arena pixels."""
    image = np.asarray(image)
    segmentation = np.asarray(segmentation)
    if image.ndim != 3 or image.shape[-1] != 3 or segmentation.ndim != 3 or segmentation.shape[-1] < 2:
        raise ValueError(f"unexpected image/segmentation shapes: {image.shape}, {segmentation.shape}")
    cx, cy = (int(round(center_xy[0])), int(round(center_xy[1])))
    half = int(size) // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        raise RuntimeError("distractor patch extends outside the camera image")
    arena = segmentation[y0:y1, x0:x1, 1] == int(arena_actor_id)
    if not arena.any():
        # Keep the screen-space anchor fixed, but never paint over a foreground
        # object that later moves through it. The visual distractor can be
        # partially or temporarily occluded by the scene, as a background
        # texture patch naturally would be.
        return image.copy()
    yy, xx = np.mgrid[0:size, 0:size]
    cells = ((xx // 6 + yy // 6) % 2).astype(bool)
    patch = np.where(cells[..., None], np.asarray([245, 55, 40], dtype=np.uint8),
                     np.asarray([25, 220, 235], dtype=np.uint8))
    out = image.copy()
    roi = out[y0:y1, x0:x1]
    roi[arena] = patch[arena].astype(image.dtype)
    return out

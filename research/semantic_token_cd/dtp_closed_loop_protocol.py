"""DTP OpenVLA calibration v1 closed-loop shared constants and manifests.

The calibration stage uses the precomputed physical-scene audit
(CALIBRATION_SCENE_AUDIT.json).  Drawer tasks keep the small set of scenes that
are physically novel relative to the formal 0-99 seeds; pick_coke_can and
move_near take ten novel scenes each.  The formal stage reuses the canonical
0-99 snapshots already covered by the fixed vanilla baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path("artifacts/dtp_openvla_calibration_v1")
CANONICAL = Path("artifacts/vanilla_recon_shr_canonical_0_299_v2")
CANONICAL_SNAPSHOTS = CANONICAL / "snapshots"
CANONICAL_EPISODES = CANONICAL / "episodes"

PROTOCOL = "DTP_OPENVLA_CALIBRATION_V1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
# Control runs the identical DTP autoregressive adapter with masking disabled;
# it isolates the effect of visual-key pruning from the adapter code path.
# mode="dynamic": per-dimension detection on the refined prefix (current v1).
# mode="fixed": paper Appendix A.2 SpatialVLA path - detection on the clean
# first action token, one mask D reused for the whole step (decode_dtp_fixed).
ARMS = ("control", "l11_k64_t05", "l7_k64_t05", "v2_fix_k64_t05", "v2_fix_k109_t05")
ARM_CONFIG = {
    "control": {"layer": 11, "k": 64, "tau": 0.5, "enabled": False, "mode": "dynamic"},
    "l11_k64_t05": {"layer": 11, "k": 64, "tau": 0.5, "enabled": True, "mode": "dynamic"},
    "l7_k64_t05": {"layer": 7, "k": 64, "tau": 0.5, "enabled": True, "mode": "dynamic"},
    "v2_fix_k64_t05": {"layer": 11, "k": 64, "tau": 0.5, "enabled": True, "mode": "fixed"},
    "v2_fix_k109_t05": {"layer": 11, "k": 109, "tau": 0.5, "enabled": True, "mode": "fixed"},
}
SPATIAL = "corners then Gaussian sigma=.65 patch units, reflect padding, truncate=4"

CALIBRATION_NOVEL_SCENES = {
    "google_robot_open_drawer": 1,
    "google_robot_close_drawer": 1,
    "google_robot_pick_coke_can": 10,
    "google_robot_move_near": 10,
}


def load_scene_audit() -> dict:
    path = ARTIFACT / "CALIBRATION_SCENE_AUDIT.json"
    if not path.exists():
        raise FileNotFoundError(f"missing scene audit: {path}")
    return json.loads(path.read_text())


def calibration_seeds(task: str) -> list[int]:
    """Ascending eligible seeds for the task; drawer tasks use the single novel scene."""
    audit = load_scene_audit()
    short = task.removeprefix("google_robot_")
    eligible = audit["tasks"][short]["eligible_seeds"]
    take = CALIBRATION_NOVEL_SCENES[task]
    if len(eligible) < take:
        raise RuntimeError(
            f"{task}: only {len(eligible)} novel seeds, need {take}; "
            "cannot run an overlap-free calibration"
        )
    return sorted(int(seed) for seed in eligible[:take])


def write_calibration_manifest() -> dict:
    """Lock the exact calibration seed list derived from the scene audit."""
    import json
    manifest = {task: calibration_seeds(task) for task in TASKS}
    path = ARTIFACT / "closed_loop/CALIBRATION_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL,
        "rule": "eligible seeds from CALIBRATION_SCENE_AUDIT.json, ascending; "
                "drawer tasks use the single physical scene novel versus formal 0-99",
        "tasks": manifest,
        "audit_file": "CALIBRATION_SCENE_AUDIT.json",
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return manifest

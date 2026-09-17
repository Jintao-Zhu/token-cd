"""XSWAP-V1: does Prompt-Attention selection value depend on instruction/task match?

Four canonical google-robot tasks; five arms per (task, dedup scene):
  vanilla / correct / paraphrase / swapped / random.
All intervention arms share the L11 prompt-attention + matched coverage
downstream (lambda=0.5, beta=0, clean prefix, gripper clean).  Only the
*selector instruction* changes between correct/paraphrase/swapped; the actual
task instruction used for the positive/negative decode never changes.
"""
from __future__ import annotations

import hashlib
import os
import json
import pickle
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
PROTOCOL = "PROMPT_ATTN_INSTR_SWAP_V1"
CANONICAL = REPO_ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
ARTIFACT = REPO_ROOT / "artifacts/prompt_attn_instr_swap_v1"

TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SHORT = {t: t.removeprefix("google_robot_") for t in TASKS}
ARMS = ("vanilla", "correct", "paraphrase", "swapped", "random")
LAMBDA = 0.5
ATTENTION_LAYERS = (11,)
N_VISUAL = 256
MAX_SCENES_PER_TASK = 100
# 2000 nominal = 4 tasks x 100 scenes x 5 arms; if a task has fewer unique
# physical scenes we record the actual count instead of padding with repeats.
STATE_STEPS = 3          # offline same-state decode steps per vanilla episode
EMIT_STEPS = (0.12, 0.5, 0.88)  # pre-fixed fraction positions along the episode

DRAWER_LEVELS = ("top", "middle", "bottom")
DRAWER_RE = re.compile(r"^(open|close) (top|middle|bottom) drawer$")
MOVE_NEAR_RE = re.compile(r"^move (.+?) near (.+)$")
OBJECT_PHRASE = {
    "opened_coke_can": "coke can", "coke_can": "coke can",
    "opened_pepsi_can": "pepsi can", "pepsi_can": "pepsi can",
    "opened_7up_can": "7up can", "7up_can": "7up can",
    "opened_sprite_can": "sprite can", "sprite_can": "sprite can",
    "redbull_can": "redbull can", "fanta_can": "fanta can",
    "orange": "orange", "apple": "apple", "sponge": "sponge",
    "blue_plastic_bottle": "blue plastic bottle",
    "opened_coke": "coke can", "opened_pepsi": "pepsi can", "opened_7up": "7up can",
}
# deterministic phrase candidates tried for move-near reference replacement.
CANDIDATE_ORDER = ("pepsi can", "coke can", "7up can", "redbull can", "sprite can",
                   "orange", "apple", "sponge", "blue plastic bottle")


def array_sha(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    d = hashlib.sha256()
    d.update(str(a.dtype).encode("ascii"))
    d.update(str(tuple(a.shape)).encode("ascii"))
    d.update(a.tobytes())
    return d.hexdigest()


def snapshot_hashes(snapshot: dict) -> dict:
    return {
        "sim_state_sha256": array_sha(np.asarray(snapshot["sim_state"])),
        "instruction": snapshot.get("instruction"),
    }


def build_scene_manifest(cap: int = MAX_SCENES_PER_TASK, seed_hi: int = 299) -> dict:
    """Deduplicate canonical snapshots per task by (instruction, sim-state).

    Returns {"scenes": {task: [seed,...]}, "stats": {...}} where every chosen
    seed is the first seed of a distinct (instruction, sim-state) group.
    """
    manifest = {"protocol_id": PROTOCOL, "scenes": {}, "stats": {}}
    for task in TASKS:
        groups = {}
        for seed in range(seed_hi + 1):
            path = CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl"
            with path.open("rb") as handle:
                snapshot = pickle.load(handle)
            key = (snapshot.get("instruction"), array_sha(np.asarray(snapshot["sim_state"])))
            groups.setdefault(key, []).append(seed)
        chosen = [seeds[0] for seeds in sorted(groups.values(), key=lambda v: v[0])]
        chosen = chosen[:cap]
        hist = {}
        for seed in chosen:
            with (CANONICAL / "snapshots" / task / f"seed_{seed:03d}.pkl").open("rb") as handle:
                inst = pickle.load(handle).get("instruction")
            hist[inst] = hist.get(inst, 0) + 1
        manifest["scenes"][task] = chosen
        manifest["stats"][task] = {
            "unique_conditions_total": len(groups),
            "selected": len(chosen),
            "cap": cap,
            "instruction_histogram": hist,
        }
    return manifest


def instruction_set(instruction: str) -> dict:
    """Correct/paraphrase/swapped selector instructions for a task instruction.

    Only 'correct' is guaranteed to equal the actual task instruction; the
    other two are fixed, deterministic phrasing rules (never outcome-selected).
    move_near swapped prefers a reference replacement that is *known present*
    (caller passes ``present`` phrases); with no suitable third object it falls
    back to a source<->target swap and is labelled ``rel_swap``.
    """
    m = DRAWER_RE.match(instruction)
    if m:
        verb, level = m.group(1), m.group(2)
        nxt = DRAWER_LEVELS[(DRAWER_LEVELS.index(level) + 1) % len(DRAWER_LEVELS)]
        return {
            "kind": "drawer",
            "correct": instruction,
            "paraphrase": f"{verb} the {level} drawer",
            "swapped": f"{verb} {nxt} drawer",
            "swapped_kind": f"level_{nxt}",
            "target_level": level,
        }
    if instruction.strip().lower() in ("pick coke can", "pick up the coke can"):
        return {
            "kind": "pick_object",
            "correct": "pick coke can",
            "paraphrase": "pick up the coke can",
            "swapped": None,  # filled from the present distractors
            "swapped_kind": "other_present_object",
        }
    m = MOVE_NEAR_RE.match(instruction)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        return {
            "kind": "move_near",
            "correct": instruction,
            "paraphrase": f"move {a} to be near {b}",
            "swapped": f"move {b} near {a}",
            "swapped_kind": "rel_swap",
            "source": a, "target_ref": b,
        }
    raise ValueError(f"unsupported canonical instruction: {instruction!r}")


def phrase_for_actor(name: str) -> str | None:
    base = name
    for key in ("visual_matching_1", "visual_matching_2"):
        base = base.replace(key, "")
    base = base.strip("_ ")
    return OBJECT_PHRASE.get(base) or OBJECT_PHRASE.get(name)


def resolve_present_phrases(env) -> list[str]:
    """Movable object phrases in the restored scene (excluding robot/cabinet)."""
    phrases = []
    try:
        actors = env.unwrapped.get_actors()
    except Exception:
        return phrases
    skip = {"ground", "arena", "goal_site", "", "cabinet"}
    for actor in actors:
        name = str(getattr(actor, "name", ""))
        if name in skip or "drawer" in name or name.startswith("link"):
            continue
        phrase = phrase_for_actor(name)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def swap_for_scene(instruction: str, present: list[str]) -> dict:
    """Fill scene-dependent swapped instruction for coke/move_near scenes."""
    base = instruction_set(instruction)
    if base["kind"] == "pick_object":
        candidates = [p for p in present if p != "coke can"]
        if not candidates:
            candidates = ["pepsi can"]
        base["swapped"] = f"pick {candidates[0]}"
        base["swapped_target_phrase"] = candidates[0]
        return base
    if base["kind"] == "move_near":
        ref = base["target_ref"]
        others = [p for p in present if p not in (base["source"], ref)]
        chosen = None
        for phrase in others:
            if phrase in CANDIDATE_ORDER:
                chosen = phrase
                break
        if chosen is None:
            for phrase in others:  # any other present object
                chosen = phrase
                break
        if chosen is not None:
            base["swapped"] = f"move {base['source']} near {chosen}"
            base["swapped_kind"] = "reference_replaced"
            base["swapped_target_phrase"] = chosen
        else:
            base["swapped_kind"] = "rel_swap_no_third_object"
        return base
    return base


def _json_default(value):
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    import os
    manifest = build_scene_manifest()
    out = ARTIFACT / "scene_manifest.json"
    atomic_json(out, manifest)
    for task, stats in manifest["stats"].items():
        print(task, stats)
    print("manifest:", out)

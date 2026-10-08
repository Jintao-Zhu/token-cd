"""Offline all-layer prompt-attention screen for pi0-FAST on fixed SIMPLER states.

This does not run policy episodes or alter the OpenPI checkout.  It creates a
temporary instrumented copy of OpenPI so we can record the VLM prefix
attention from each of the 18 PaliGemma layers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
OPENPI = ROOT / "openpi"
SOURCE_STATES = REPO / "artifacts/spatialvla_layer_selection_v1b/states"
OUT = REPO / "artifacts/pi0_fast_layer_selection_v1"
TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
SEEDS = (100, 110, 120, 130, 150, 160)


def instrument_openpi() -> Path:
    """Return a temporary OpenPI source overlay with a tiny attention tap."""
    tmp = Path(tempfile.mkdtemp(prefix="pi0_fast_layer_scan_"))
    pkg = tmp / "openpi"
    shutil.copytree(OPENPI / "src/openpi", pkg)
    path = pkg / "models/gemma_fast.py"
    source = path.read_text()
    source = source.replace(
        "import dataclasses\nfrom typing import Literal, TypeAlias",
        "import dataclasses\nimport os\nimport numpy as np\nfrom typing import Literal, TypeAlias",
    )
    anchor = "        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)\n"
    tap = '''        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # Optional diagnostic tap. The configured query mask selects only the
        # textual task phrase; keys 0:256 are the first/overhead image tokens.
        if _PI0FAST_ATTN_CAPTURE is not None:
            query_mask = jnp.asarray(_PI0FAST_ATTN_CAPTURE["query_mask"], dtype=probs.dtype)
            qmass = jnp.maximum(jnp.sum(query_mask, axis=1), 1.0)
            key_start = int(_PI0FAST_ATTN_CAPTURE.get("key_start", 0))
            selected = probs[..., key_start : key_start + 256]
            selected = jnp.einsum("bkgts,bt->bks", selected, query_mask)
            selected = jnp.sum(selected, axis=1) / qmass[:, None]
            jax.debug.callback(_PI0FAST_ATTN_CAPTURE["callback"], selected, ordered=True)
'''
    if source.count(anchor) != 1:
        raise RuntimeError("Could not locate the unique attention softmax insertion point")
    source = source.replace(anchor, tap)
    header = '''
# Process-local offline attention capture; set by the research scanner.
_PI0FAST_ATTN_CAPTURE = None

'''
    source = source.replace("Variant = Literal[", header + "Variant = Literal[", 1)
    path.write_text(source)
    return tmp


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("empty query token sequence")
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    raise ValueError("task phrase token IDs were not found in FAST prefix")


def _prompt_variants(task: str, instruction: str, controls: tuple[str, int, int]) -> tuple[str, str, str]:
    target_switch = controls[0]
    synonyms = {
        "google_robot_open_drawer": "pull the top drawer open",
        "google_robot_close_drawer": "shut the top drawer",
        "google_robot_pick_coke_can": "grasp the coke can",
    }
    if task == "google_robot_move_near":
        source, target = instruction[len("move ") :].split(" near ", 1)
        synonym = f"move {source} close to {target}"
    else:
        synonym = synonyms[task]
    return instruction, synonym, target_switch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="debug state limit; zero scans all 24 states")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    overlay = instrument_openpi()
    sys.path.insert(0, str(OPENPI / "src"))
    sys.path.insert(0, str(overlay))
    sys.path.insert(0, str(REPO))

    import jax
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies import policy_config
    from openpi.shared.normalize import NormStats
    from openpi.training import config as config_lib
    from research.semantic_token_cd.spatialvla_layer_selection_offline import get_controls
    import openpi.models.gemma_fast as gemma_fast

    stats_json = ROOT / "checkpoints/openpi_fast_finetuned/fractal/assets/local/fractal_lerobot/norm_stats.json"
    source_stats = json.loads(stats_json.read_text())["norm_stats"]
    norm_stats = {
        key: NormStats(
            mean=np.asarray(value["mean"], np.float32),
            std=np.asarray(value["std"], np.float32),
            q01=np.asarray(value["q01"], np.float32),
            q99=np.asarray(value["q99"], np.float32),
        )
        for key, value in source_stats.items()
        if key in ("state", "actions")
    }
    cfg = config_lib.get_config("pi0_fast_libero")
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "observation/image": "observation/primary_image",
                    "observation/wrist_image": "observation/wrist_image",
                    "observation/state": "observation/state",
                    "prompt": "prompt",
                }
            )
        ]
    )
    checkpoint = ROOT / "checkpoints/openpi-assets/openpi-assets/checkpoints/pi0_fast_base"
    policy = policy_config.create_trained_policy(
        cfg, checkpoint, repack_transforms=repack, norm_stats=norm_stats
    )
    model = policy._model
    # FAST's PaliGemma backbone is Gemma-2B with 18 language layers.
    depth = int(gemma_fast.get_config("gemma_2b").depth)
    if depth != 18:
        raise RuntimeError(f"unexpected PaliGemma depth {depth}")

    # Locate FAST tokenizer instance from the exact policy input pipeline.
    tokenizer = None
    for transform in policy._input_transform.transforms:
        if hasattr(transform, "tokenizer") and hasattr(transform.tokenizer, "_paligemma_tokenizer"):
            tokenizer = transform.tokenizer
    if tokenizer is None:
        raise RuntimeError("FAST tokenizer not found in policy input transforms")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "capture_overlay.txt").write_text(str(overlay) + "\n")
    processed = 0
    for task in TASKS:
        for seed in SEEDS:
            state_dir = SOURCE_STATES / task / f"seed_{seed:03d}"
            meta = json.loads((state_dir / "meta.json").read_text())
            npz = np.load(state_dir / "input.npz")
            image = np.asarray(npz["image"])
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            instruction = meta["instruction"]
            target_switch, old_id, new_id = get_controls(task, instruction, meta)
            prompts = _prompt_variants(task, instruction, (target_switch, old_id, new_id))

            # Reproduce the deployed SIMPLER adapter's view layout: overhead
            # input plus a zero wrist image. State is held at the same neutral
            # input for the three text variants; this is an offline selector
            # screen, not action evaluation.
            transformed = []
            q_ranges = []
            for prompt in prompts:
                raw = {
                    "observation/primary_image": image,
                    "observation/wrist_image": np.zeros_like(image),
                    "observation/state": np.zeros((8,), dtype=np.float32),
                    "prompt": prompt,
                }
                item = policy._input_transform(raw)
                transformed.append(item)
                clean = prompt.lower().strip().replace("_", " ")
                prefix = f"Task: {clean}, State: "
                proto = tokenizer._paligemma_tokenizer.encode_as_immutable_proto(prefix)
                # SentencePiece offsets locate precisely the task text within
                # the prefix, excluding fixed `Task:`/`, State:` scaffolding.
                task_start = len("Task: ")
                task_end = task_start + len(clean)
                pos = []
                offset = 1  # FAST prefix begins with BOS.
                for piece in proto.pieces:
                    if piece.begin >= task_start and piece.end <= task_end:
                        pos.append(offset)
                    offset += 1
                if not pos:
                    ids = list(np.asarray(item["tokenized_prompt"]).tolist())
                    query_ids = tokenizer._paligemma_tokenizer.encode(clean)
                    start = _find_subsequence(ids, query_ids)
                    pos = list(range(start, start + len(query_ids)))
                q_ranges.append(pos)

            # Model transforms yield fixed-size token arrays. Stack three
            # views/prompts into one batch so each layer is run once per state.
            batched = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *transformed)
            observation = model_lib.Observation.from_dict(batched)
            observation = model_lib.preprocess_observation(
                jax.random.key(seed), observation, train=False, image_keys=list(observation.images)
            )
            embedded, input_mask, ar_mask = model.embed_inputs(observation)
            from openpi.models.pi0_fast import make_attn_mask
            mask = make_attn_mask(input_mask, ar_mask)

            # Image input order in the Fractal adapter is base-0, base-1,
            # wrist. Each SigLIP view contributes 256 visual tokens.
            qmask = np.zeros((3, embedded.shape[1]), dtype=np.float32)
            image_prefix = 3 * 256
            if embedded.shape[1] != image_prefix + cfg.model.max_token_len:
                raise RuntimeError(f"unexpected prefix length {embedded.shape[1]}")
            for i, positions in enumerate(q_ranges):
                for pos in positions:
                    absolute = image_prefix + int(pos)
                    if 0 <= absolute < embedded.shape[1]:
                        qmask[i, absolute] = 1.0
            if not np.all(qmask.sum(axis=1) > 0):
                raise RuntimeError(f"empty prompt query mask for {task} seed {seed}: {q_ranges}")

            captured: list[np.ndarray] = []
            gemma_fast._PI0FAST_ATTN_CAPTURE = {"query_mask": qmask, "callback": lambda x: captured.append(np.asarray(x))}
            try:
                model.PaliGemma.llm(embedded_prefix=embedded, mask=mask, return_prelogits=True)
                jax.effects_barrier()
            finally:
                gemma_fast._PI0FAST_ATTN_CAPTURE = None
            if len(captured) != depth:
                raise RuntimeError(f"captured {len(captured)} layers, expected {depth}")
            # [layer, prompt variant, 256 primary image tokens]
            scores = np.stack(captured).astype(np.float32)

            def actor_mask(actor_id: int) -> np.ndarray:
                seg = np.asarray(npz["seg_actor"])
                seg = cv2.resize((seg == actor_id).astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
                return seg.reshape(16, 14, 16, 14).any(axis=(1, 3)).reshape(256).astype(np.uint8)

            old_mask, new_mask = actor_mask(old_id), actor_mask(new_id)
            if not old_mask.any() or not new_mask.any():
                raise RuntimeError(f"empty actor token mask for {task} seed {seed}")
            target_dir = OUT / "states" / task / f"seed_{seed:03d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target_dir / "layers.npz",
                original=scores[:, 0], synonym=scores[:, 1], target_switch=scores[:, 2],
                old_target_mask=old_mask, new_target_mask=new_mask,
            )
            (target_dir / "meta.json").write_text(json.dumps({
                "protocol": "PI0_FAST_LAYER_SELECTION_OFFLINE_V1",
                "task": task, "seed": seed, "checkpoint": str(checkpoint),
                "model_language_layers": depth, "view": "SIMPLER overhead RGB, zero wrist view",
                "renderer_of_captured_state": "the existing fixed state image; no env re-render",
                "instruction": instruction, "synonym_instruction": prompts[1],
                "target_switch_instruction": prompts[2], "old_actor_id": old_id,
                "new_actor_id": new_id, "old_target_tokens": int(old_mask.sum()),
                "new_target_tokens": int(new_mask.sum()),
                "prompt_query_token_counts": [len(x) for x in q_ranges],
                "attention_capture": "mean over heads and task-text query tokens, to primary-view SigLIP tokens",
                "state_input": "zero raw state transformed with Fractal stats; held constant across prompt variants; attention-screening caveat",
            }, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"task": task, "seed": seed, "layers": depth,
                              "query_tokens": [len(x) for x in q_ranges],
                              "old_tokens": int(old_mask.sum()), "new_tokens": int(new_mask.sum()),
                              "captured": True}), flush=True)
            processed += 1
            if args.limit and processed >= args.limit:
                print("DEBUG_LIMIT_REACHED", flush=True)
                return
    print("EXTRACTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

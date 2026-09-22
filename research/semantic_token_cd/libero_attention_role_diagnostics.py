#!/usr/bin/env python3
"""Layer/query-role diagnostics for LIBERO prompt attention.

This is an offline audit. It does not run closed-loop control.  For each
development state it computes attention maps from several layer groups and
from two query modes:

  full : all instruction tokens
  role : target object / relation phrase / reference object, each normalized
         over visual tokens and then equally averaged

The script uses simulator instance segmentation only as a diagnostic ruler for
target/distractor/reference coverage.  No GT mask enters the rollout policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.libero_runtime import (
    build_prompt,
    load_policy,
    prepare_agentview,
    set_determinism,
)
from research.semantic_token_cd.libero_policy import (
    extract_source_target_entities_libero,
    token_ids_from_mask,
)
from research.semantic_token_cd.libero_matched_rollout import find_instruction_span

N_VISUAL = 256
GRID = 16
EPS = 1e-12

TASK_REFERENCE = {
    0: ("plate_1", "glazed_rim_porcelain_ramekin_1"),
    1: ("glazed_rim_porcelain_ramekin_1",),
    2: (),
    3: ("cookies_1",),
    4: ("wooden_cabinet_1",),
    5: ("glazed_rim_porcelain_ramekin_1",),
    6: ("cookies_1",),
    7: ("flat_stove_1",),
    8: ("plate_1",),
    9: ("wooden_cabinet_1",),
}

SWITCH_TEMPLATES = {
    0: ("pick up the black bowl between the plate and the cookie box and place it on the plate", "cookies_1"),
    1: ("pick up the black bowl next to the cookie box and place it on the plate", "cookies_1"),
    3: ("pick up the black bowl on the ramekin and place it on the plate", "glazed_rim_porcelain_ramekin_1"),
    5: ("pick up the black bowl on the cookie box and place it on the plate", "cookies_1"),
    6: ("pick up the black bowl next to the ramekin and place it on the plate", "glazed_rim_porcelain_ramekin_1"),
    7: ("pick up the black bowl on the plate and place it on the plate", "plate_1"),
    8: ("pick up the black bowl next to the ramekin and place it on the plate", "glazed_rim_porcelain_ramekin_1"),
    9: ("pick up the black bowl on the stove and place it on the plate", "flat_stove_1"),
}


def parse_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_layer_groups(value: str) -> list[tuple[str, tuple[int, ...]]]:
    groups: list[tuple[str, tuple[int, ...]]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            layers = tuple(range(lo, hi + 1))
            groups.append((f"mean{lo}-{hi}", layers))
        else:
            layer = int(part)
            groups.append((str(layer), (layer,)))
    return groups


def find_phrase_span(ids: list[int], tokenizer, phrase: str, search_start: int = 0, search_end: int | None = None) -> tuple[int, int]:
    phrase = phrase.lower().strip()
    if search_end is None:
        search_end = len(ids)
    candidates: list[tuple[int, int]] = []
    for add_special in (False, True):
        sub = tokenizer(phrase, add_special_tokens=add_special)["input_ids"]
        if add_special and sub and int(sub[0]) == int(tokenizer.bos_token_id):
            sub = sub[1:]
        sub = [int(x) for x in sub]
        if not sub or len(sub) > search_end - search_start:
            continue
        for start in range(search_start, search_end - len(sub) + 1):
            if ids[start:start + len(sub)] == sub:
                candidates.append((start, start + len(sub)))
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"phrase span not unique: {phrase!r}: {unique}")
    return unique[0]


def role_texts(instruction: str) -> tuple[str, str, str]:
    """Return target object, relation phrase, and reference phrase."""
    target = extract_source_target_entities_libero(instruction)[0]
    first_clause = re.split(r"\band\s+(?:place|put|move)\b", instruction.lower(), maxsplit=1)[0]
    stripped = re.sub(
        r"^pick\s+up\s+(?:the\s+)?", "", first_clause
    ).strip()
    # Remove the target object and determiners from the front.
    target_words = target.split()
    words = stripped.split()
    while words and words[0] in {"the", "a", "an"}:
        words.pop(0)
    if words[:len(target_words)] == target_words:
        words = words[len(target_words):]
    remainder = " ".join(words).strip()
    for relation in ("next to", "between", "from", "on", "in"):
        if remainder.startswith(relation + " "):
            return target, relation, remainder[len(relation):].strip()
    raise RuntimeError(f"cannot parse relation/reference from {instruction!r}: {remainder!r}")


def span_indices(ids: list[int], tokenizer, texts: tuple[str, ...], search_start: int = 0, search_end: int | None = None) -> list[int]:
    result: set[int] = set()
    for text in texts:
        start, end = find_phrase_span(ids, tokenizer, text, search_start=search_start, search_end=search_end)
        result.update(range(start, end))
    return sorted(result)


def attention_map(attn: torch.Tensor, query_indices: list[int]) -> np.ndarray:
    selected = attn[0, :, [N_VISUAL + i for i in query_indices], 1:1 + N_VISUAL]
    return selected.detach().float().cpu().mean(dim=(0, 1)).numpy()


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("invalid attention map mass")
    return values / total


def topk_indices(values: np.ndarray, k: int) -> set[int]:
    order = np.lexsort((np.arange(N_VISUAL), -values))
    return set(int(x) for x in order[:k])


def mask_tokens(seg: np.ndarray, instance_to_id: dict, name: str) -> set[int]:
    if name not in instance_to_id:
        return set()
    mask = (seg[..., 0] == instance_to_id[name]).astype(np.uint8)
    ids, _ = token_ids_from_mask(mask)
    return set(int(x) for x in ids)


def forward_attention(model, processor, image, instruction: str):
    inputs = processor(build_prompt(instruction), image).to(model.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    if out.attentions is None:
        raise RuntimeError("model did not return attentions")
    return inputs, out.attentions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--tasks", default="0-9")
    p.add_argument("--states", default="0-19")
    p.add_argument("--layer-groups", default="8,11,16,24,31,10-12,15-17,23-25,30-31")
    p.add_argument("--topk", default="16,32,50")
    a = p.parse_args()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    task_ids = parse_ints(a.tasks)
    states = parse_ints(a.states)
    layer_groups = parse_layer_groups(a.layer_groups)
    topk_values = parse_ints(a.topk)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    set_determinism(7)
    model, processor = load_policy(
        a.checkpoint,
        Path("third_party/openvla/prismatic/extern/hf"),
        device=f"cuda:{a.gpu}",
    )

    out_root = a.artifact.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        init_states = suite.get_task_init_states(task_id)
        try:
            for state_idx in states:
                if state_idx >= len(init_states):
                    continue
                env.seed(0)
                env.reset()
                obs = env.set_init_state(init_states[state_idx])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                seg = np.asarray(obs["agentview_segmentation_instance"])
                _, image = prepare_agentview(obs)
                inputs, attentions = forward_attention(model, processor, image, task.language)
                ids = inputs["input_ids"][0].detach().cpu().tolist()
                instruction_span = find_instruction_span(ids, processor.tokenizer, task.language)
                target_text, relation_text, reference_text = role_texts(task.language)
                source_text = re.split(r"\band\s+(?:place|put|move)\b", task.language.lower(), maxsplit=1)[0]
                source_span = find_phrase_span(ids, processor.tokenizer, source_text)
                role_spans = {
                    "target": span_indices(ids, processor.tokenizer, (target_text,), *source_span),
                    "relation": span_indices(ids, processor.tokenizer, (relation_text,), *source_span),
                    "reference": span_indices(ids, processor.tokenizer, (reference_text,), *source_span),
                }
                target_tokens = mask_tokens(seg, env.instance_to_id, "akita_black_bowl_1")
                distractor_tokens = mask_tokens(seg, env.instance_to_id, "akita_black_bowl_2")
                reference_tokens: set[int] = set()
                for name in TASK_REFERENCE[task_id]:
                    reference_tokens |= mask_tokens(seg, env.instance_to_id, name)

                switch_template = SWITCH_TEMPLATES.get(task_id)
                switch_alt_tokens: set[int] = set()
                switch_original_tokens: set[int] = set()
                switch_maps: dict[tuple[str, str], np.ndarray] = {}
                if switch_template is not None:
                    switch_instruction, switch_ref_name = switch_template
                    _, switch_attentions = forward_attention(model, processor, image, switch_instruction)
                    switch_ids = processor(build_prompt(switch_instruction), image).to(
                        model.device, dtype=torch.bfloat16
                    )["input_ids"][0].detach().cpu().tolist()
                    sw_target, sw_relation, sw_reference = role_texts(switch_instruction)
                    sw_source_text = re.split(r"\band\s+(?:place|put|move)\b", switch_instruction.lower(), maxsplit=1)[0]
                    sw_source_span = find_phrase_span(switch_ids, processor.tokenizer, sw_source_text)
                    sw_role_spans = {
                        "target": span_indices(switch_ids, processor.tokenizer, (sw_target,), *sw_source_span),
                        "relation": span_indices(switch_ids, processor.tokenizer, (sw_relation,), *sw_source_span),
                        "reference": span_indices(switch_ids, processor.tokenizer, (sw_reference,), *sw_source_span),
                    }
                    for group_name, layers in layer_groups:
                        full_map = np.mean(
                            [attention_map(attentions[i], list(range(instruction_span[0], instruction_span[1]))) for i in layers],
                            axis=0,
                        )
                        role_map = np.mean(
                            [
                                normalize_map(attention_map(attentions[i], role_spans["target"]))
                                + normalize_map(attention_map(attentions[i], role_spans["relation"]))
                                + normalize_map(attention_map(attentions[i], role_spans["reference"]))
                                for i in layers
                            ],
                            axis=0,
                        )
                        role_map = normalize_map(role_map)
                        sw_full_map = np.mean(
                            [attention_map(switch_attentions[i], list(range(instruction_span[0], instruction_span[1]))) for i in layers],
                            axis=0,
                        )
                        sw_role_map = np.mean(
                            [
                                normalize_map(attention_map(switch_attentions[i], sw_role_spans["target"]))
                                + normalize_map(attention_map(switch_attentions[i], sw_role_spans["relation"]))
                                + normalize_map(attention_map(switch_attentions[i], sw_role_spans["reference"]))
                                for i in layers
                            ],
                            axis=0,
                        )
                        sw_role_map = normalize_map(sw_role_map)
                        for mode, values in (("full", full_map), ("role", role_map)):
                            for k in topk_values:
                                selected = topk_indices(values, k)
                                target_cov = len(selected & target_tokens) / max(1, len(target_tokens))
                                distractor_cov = len(selected & distractor_tokens) / max(1, len(distractor_tokens))
                                target_mass = float(values[list(target_tokens)].sum()) / max(1, len(target_tokens)) if target_tokens else 0.0
                                distractor_mass = float(values[list(distractor_tokens)].sum()) / max(1, len(distractor_tokens)) if distractor_tokens else 0.0
                                reference_mass = float(values[list(reference_tokens)].sum()) / max(1, len(reference_tokens)) if reference_tokens else 0.0
                                rows.append({
                                    "task_id": task_id,
                                    "task": task.name,
                                    "init_state": state_idx,
                                    "layer_group": group_name,
                                    "query_mode": mode,
                                    "topk": k,
                                    "target_tokens": len(target_tokens),
                                    "distractor_tokens": len(distractor_tokens),
                                    "target_coverage": target_cov,
                                    "distractor_coverage": distractor_cov,
                                    "target_discrimination": target_cov - distractor_cov,
                                    "target_mass": target_mass,
                                    "distractor_mass": distractor_mass,
                                    "target_mass_diff": target_mass - distractor_mass,
                                    "reference_mass": reference_mass,
                                    "switch_alt_mass_diff": None,
                                    "switch_alt_coverage_diff": None,
                                })
                        # relation-switch response, using role aggregation
                        alt_tokens = mask_tokens(seg, env.instance_to_id, switch_ref_name)
                        original_tokens = reference_tokens
                        alt_mass = float(sw_role_map[list(alt_tokens)].sum()) / max(1, len(alt_tokens)) if alt_tokens else 0.0
                        orig_mass = float(role_map[list(original_tokens)].sum()) / max(1, len(original_tokens)) if original_tokens else 0.0
                        alt_cov = len(topk_indices(sw_role_map, max(topk_values)) & alt_tokens) / max(1, len(alt_tokens))
                        orig_cov = len(topk_indices(role_map, max(topk_values)) & original_tokens) / max(1, len(original_tokens))
                        rows.append({
                            "task_id": task_id,
                            "task": task.name,
                            "init_state": state_idx,
                            "layer_group": group_name,
                            "query_mode": "role_switch",
                            "topk": max(topk_values),
                            "target_tokens": len(target_tokens),
                            "distractor_tokens": len(distractor_tokens),
                            "target_coverage": None,
                            "distractor_coverage": None,
                            "target_discrimination": None,
                            "target_mass": None,
                            "distractor_mass": None,
                            "target_mass_diff": None,
                            "reference_mass": None,
                            "switch_alt_mass_diff": alt_mass - orig_mass,
                            "switch_alt_coverage_diff": alt_cov - orig_cov,
                        })
                print(json.dumps({"task": task.name, "init_state": state_idx, "done": True}), flush=True)
        finally:
            env.close()

    out_csv = out_root / "ATTENTION_ROLE_DIAGNOSTICS.csv"
    if rows:
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "csv": str(out_csv)}), flush=True)


if __name__ == "__main__":
    main()

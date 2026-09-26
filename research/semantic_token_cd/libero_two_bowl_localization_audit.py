#!/usr/bin/env python3
"""Offline prompt-attention localization audit for two LIBERO-Spatial bowl tasks.

Renders the same initial states used by the archived paired rollout, extracts
all language-layer prompt-to-vision attention, and compares it to simulator
instance masks for the designated and distractor black bowls. No policy action
is executed. Each process handles an explicit list of (task_id, init_state)
pairs and writes one JSONL file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
CHECKPOINT = Path("/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial")
CODE_DIR = ROOT / "third_party/openvla/prismatic/extern/hf"
MATCHED_ROOT = ROOT / "artifacts/libero_official_matched_500_v1"
VANILLA_ROOT = ROOT / "artifacts/libero_official_vanilla_500_v1"
TASKS = {
    2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    8: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
}
ALTERNATE_PROMPTS = {
    2: "pick up the black bowl next to the plate and place it on the plate",
    8: "pick up the black bowl next to the cookie box and place it on the plate",
}
N_VISUAL = 256


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def token_span(ids: list[int], tokenizer, phrase: str, start: int = 0, end: int | None = None) -> tuple[int, int]:
    if end is None:
        end = len(ids)
    found = set()
    for special in (False, True):
        sub = [int(x) for x in tokenizer(phrase, add_special_tokens=special)["input_ids"]]
        if special and sub and sub[0] == int(tokenizer.bos_token_id):
            sub = sub[1:]
        for i in range(start, end - len(sub) + 1):
            if sub and ids[i:i + len(sub)] == sub:
                found.add((i, i + len(sub)))
    if len(found) != 1:
        raise RuntimeError(f"token span not unique: {phrase!r}: {sorted(found)}")
    return next(iter(found))


def indices(span: tuple[int, int]) -> list[int]:
    return list(range(span[0], span[1]))


def query_map(attention, query_indices: list[int]) -> tuple[np.ndarray, float]:
    q = [N_VISUAL + i for i in query_indices]
    selected = attention[0, :, q, 1:1 + N_VISUAL].detach().float().cpu()
    raw = selected.mean(dim=(0, 1)).numpy().astype(np.float64)
    return raw / max(float(raw.sum()), 1e-30), float(raw.sum())


def object_tokens(segmentation: np.ndarray, instance_to_id: dict, name: str) -> set[int]:
    from research.semantic_token_cd.libero_policy import token_ids_from_mask
    if name not in instance_to_id:
        return set()
    mask = (segmentation[..., 0] == instance_to_id[name]).astype(np.uint8)
    ids, _ = token_ids_from_mask(mask)
    return set(map(int, ids))


def metrics(values: np.ndarray, target: set[int], distractor: set[int], k: int) -> dict:
    order = np.lexsort((np.arange(N_VISUAL), -values))
    top = set(map(int, order[:max(0, min(k, N_VISUAL))]))
    target_mass = float(values[list(target)].sum()) if target else 0.0
    distractor_mass = float(values[list(distractor)].sum()) if distractor else 0.0
    target_density = target_mass / max(1, len(target))
    distractor_density = distractor_mass / max(1, len(distractor))
    return {
        "target_tokens": len(target),
        "distractor_tokens": len(distractor),
        "target_attention_mass": target_mass,
        "distractor_attention_mass": distractor_mass,
        "target_attention_density": target_density,
        "distractor_attention_density": distractor_density,
        "target_vs_distractor_density_ratio": target_density / max(distractor_density, 1e-30),
        "topk_target_coverage": len(top & target) / max(1, len(target)),
        "topk_distractor_coverage": len(top & distractor) / max(1, len(distractor)),
        "topk_target_precision": len(top & target) / max(1, len(top)),
        "topk_distractor_precision": len(top & distractor) / max(1, len(top)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="physical inference GPU")
    ap.add_argument("--render-gpu", type=int, default=7)
    ap.add_argument("--cases", required=True, help="comma-separated task_id:init_state pairs")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--worker-id", required=True)
    args = ap.parse_args()

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    visible = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if args.gpu not in visible:
        raise RuntimeError(f"inference GPU {args.gpu} not in CUDA_VISIBLE_DEVICES={visible}")
    device_index = visible.index(args.gpu)
    from research.semantic_token_cd.resolve_mujoco_egl_device import resolve as resolve_egl_device
    egl = resolve_egl_device(args.render_gpu)
    # LIBERO/robosuite checks this variable against CUDA_VISIBLE_DEVICES during import.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)

    import torch
    from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, prepare_agentview, set_determinism
    from research.semantic_token_cd.libero_attention_role_diagnostics import find_phrase_span, role_texts

    set_determinism(7)
    model, processor = load_policy(CHECKPOINT, CODE_DIR, device=f"cuda:{device_index}")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    # Restore the EGL ordinal after robosuite's import-time visibility guard.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl["egl_ordinal"])
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    parsed = []
    for item in args.cases.split(","):
        tid, state = item.split(":", 1)
        parsed.append((int(tid), int(state)))
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "worker_start", "worker": args.worker_id, "gpu": args.gpu,
                      "render_gpu": args.render_gpu, "egl": egl, "device_index": device_index,
                      "cases": len(parsed)}, sort_keys=True), flush=True)

    with out_path.open("w", buffering=1) as out:
        for tid, state_idx in parsed:
            task_name = TASKS[tid]
            task_index = next((i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name), None)
            if task_index is None:
                raise RuntimeError(f"task not present in current LIBERO-Spatial suite: {task_name}")
            task = suite.get_task(task_index)
            matched_path = MATCHED_ROOT / task_name / f"episode_{state_idx:03d}.json"
            vanilla_path = VANILLA_ROOT / task_name / f"episode_{state_idx:03d}.json"
            if not matched_path.exists() or not vanilla_path.exists():
                raise FileNotFoundError(f"missing paired outcome for {task_name} state {state_idx}")
            matched = json.loads(matched_path.read_text())
            vanilla = json.loads(vanilla_path.read_text())
            if matched.get("episode") != state_idx or vanilla.get("episode") != state_idx:
                raise RuntimeError("episode index mismatch")
            m = int(matched["trace"][0]["m_matched"])
            old_selected = set(map(int, matched["trace"][0]["selected_token_ids"]))
            label = ("rescue" if matched["success"] and not vanilla["success"] else
                     "harm" if vanilla["success"] and not matched["success"] else
                     "both_success" if matched["success"] else "both_fail")

            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            try:
                init_states = suite.get_task_init_states(task_index)
                if state_idx >= len(init_states):
                    raise IndexError(f"state {state_idx} outside {len(init_states)} init states")
                env.seed(0)
                env.reset()
                obs = env.set_init_state(init_states[state_idx])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                state = np.asarray(env.get_sim_state()).copy()
                state_hash = sha256_array(state)
                seg = np.asarray(obs["agentview_segmentation_instance"])
                _, image = prepare_agentview(obs)
                target = object_tokens(seg, env.instance_to_id, "akita_black_bowl_1")
                distractor = object_tokens(seg, env.instance_to_id, "akita_black_bowl_2")
                if not target or not distractor:
                    raise RuntimeError(f"missing bowl instance masks target={len(target)} distractor={len(distractor)}")

                def get_attentions(instruction: str):
                    inputs = processor(build_prompt(instruction), image).to(f"cuda:{device_index}", dtype=torch.bfloat16)
                    with torch.inference_mode():
                        result = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                       pixel_values=inputs["pixel_values"], use_cache=False,
                                       output_attentions=True, return_dict=True)
                    if result.attentions is None or len(result.attentions) != 32:
                        raise RuntimeError("expected attention maps for all 32 language layers")
                    ids = inputs["input_ids"][0].detach().cpu().tolist()
                    return ids, result.attentions

                ids, attentions = get_attentions(task.language)
                full_span = find_phrase_span(ids, processor.tokenizer, task.language)
                source_clause = re.split(r"\band\s+(?:place|put|move)\b", task.language.lower(), maxsplit=1)[0]
                source_span = find_phrase_span(ids, processor.tokenizer, source_clause)
                target_text, relation_text, reference_text = role_texts(task.language)
                target_span = find_phrase_span(ids, processor.tokenizer, target_text, *source_span)
                relation_span = find_phrase_span(ids, processor.tokenizer, relation_text, *source_span)
                reference_span = find_phrase_span(ids, processor.tokenizer, reference_text, *source_span)
                modes = {
                    "full_instruction": indices(full_span),
                    "source_clause": indices(source_span),
                    "object_words": indices(target_span),
                    "relation_words": indices(relation_span),
                    "reference_words": indices(reference_span),
                }
                layer_rows = []
                recomputed_l11_topm: set[int] = set()
                for layer, attn in enumerate(attentions):
                    mode_metrics = {}
                    for mode, qidx in modes.items():
                        amap, mass = query_map(attn, qidx)
                        md = metrics(amap, target, distractor, m)
                        md["visual_attention_mass"] = mass
                        mode_metrics[mode] = md
                        if layer == 11 and mode == "full_instruction":
                            order = np.lexsort((np.arange(N_VISUAL), -amap))
                            recomputed_l11_topm = set(map(int, order[:m]))
                    layer_rows.append({"layer": layer, "modes": mode_metrics})

                alt_ids, alt_attentions = get_attentions(ALTERNATE_PROMPTS[tid])
                alt_span = find_phrase_span(alt_ids, processor.tokenizer, ALTERNATE_PROMPTS[tid])
                switch_rows = []
                for layer, attn in enumerate(alt_attentions):
                    amap, mass = query_map(attn, indices(alt_span))
                    switch_rows.append({"layer": layer, "metrics": metrics(amap, target, distractor, m),
                                        "visual_attention_mass": mass})

                rec = {
                    "worker_id": args.worker_id, "gpu": args.gpu, "render_gpu": args.render_gpu,
                    "task_id": tid, "task": task_name, "init_state": state_idx,
                    "outcome": {"matched": bool(matched["success"]), "vanilla": bool(vanilla["success"]), "label": label},
                    "initial_state_sha256": state_hash,
                    "expected_initial_state_sha256": matched["initial_state_sha256"],
                    "state_hash_match": state_hash == matched["initial_state_sha256"] == vanilla["initial_state_sha256"],
                    "matched_budget_m": m, "historical_selected_token_count": len(old_selected),
                    "historical_selected_target_tokens": len(old_selected & target),
                    "historical_selected_distractor_tokens": len(old_selected & distractor),
                    "historical_selected_target_coverage": len(old_selected & target) / len(target),
                    "historical_selected_distractor_coverage": len(old_selected & distractor) / len(distractor),
                    "recomputed_l11_topm_jaccard_to_historical": len(recomputed_l11_topm & old_selected) / max(1, len(recomputed_l11_topm | old_selected)),
                    "recomputed_l11_topm_target_tokens": len(recomputed_l11_topm & target),
                    "recomputed_l11_topm_distractor_tokens": len(recomputed_l11_topm & distractor),
                    "layer_metrics": layer_rows, "alternate_relation_prompt": ALTERNATE_PROMPTS[tid],
                    "alternate_prompt_metrics": switch_rows,
                }
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                print(json.dumps({"event": "case_done", "worker": args.worker_id, "task_id": tid,
                                  "state": state_idx, "state_hash_match": rec["state_hash_match"],
                                  "m": m, "outcome": label}, sort_keys=True), flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()

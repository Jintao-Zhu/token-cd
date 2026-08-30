#!/usr/bin/env python3
"""LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1 — offline alignment gate.

Verifies that the Phase-2B language-selected semantic entity counterfactual
transfers to LIBERO-Object: compute cos(r_entity, r_oracle) where
  r_entity = log_softmax(z_clean) - log_softmax(z_masked_by_entity_set)
  r_oracle = log_softmax(z_clean) - log_softmax(z_masked_by_GT_object_mask)
and the GT object mask is the LIBERO simulator's instance segmentation
(SegmentationRenderEnv), used as a MEASUREMENT RULER only (never in the rollout
method). Mirrors Phase-2B's SAM-as-ruler (align_entity_set_K8 = 0.586).

Gate: mean alignment > 0.55 -> PASS_TO_ROLLOUT; < 0.40 -> STOP_..._NO_GO.

Run (smoke):
  cd /home/leju-suzhou/zjt_ws/token-cd
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3 \
  PYTHONPATH="./LIBERO:$PWD" \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/libero_phase0_align.py \
      --artifact artifacts/libero_object_semantic_entity_cd_phase0_v1 --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
CHECKPOINT = REPO / "checkpoints/openvla-7b-finetuned-libero-object/287d6cfdf12d07b1449505f66d9bf3550257e9b3"
CODE_DIR = REPO / "third_party/openvla/prismatic/extern/hf"

from research.ar_token_counterfactual.libero_runtime import load_policy, prepare_agentview, set_determinism  # noqa: E402
from research.ar_token_closed_loop.common import write_json  # noqa: E402
from research.cw_lpcd.core import action_token_slice, log_softmax_residual, N_VISUAL  # noqa: E402
from research.cw_lpcd.metrics import per_position_cosine  # noqa: E402
from research.semantic_token_cd.libero_policy import (  # noqa: E402
    compute_clean_ids,
    embed_phrase,
    entity_select,
    extract_entities_libero,
    extract_h,
    forward_logits,
    token_ids_from_mask,
)

K = 8
KMEANS_SEED = 0
THRESHOLD_PASS = 0.55
THRESHOLD_STOP = 0.40
PROJ_DIM = 4096


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _npz_value(v):
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else v.tolist()
    return v


def mean_states(tasks):
    """Deterministic small subset of states for the position-conditioned mean."""
    return [(t, e) for t in tasks for e in range(2)]  # 2 states/task


def _build_bddl(get_libero_path, task) -> str:
    return str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--tasks", default=None, help="comma-separated task ids (default all 10)")
    p.add_argument("--episodes", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--limit", type=int)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="1 task x 2 states")
    a = p.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    art = a.artifact.resolve()
    (art / "align_npz").mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task_ids = [int(x) for x in a.tasks.split(",")] if a.tasks else list(range(10))
    if a.smoke:
        task_ids = task_ids[:1]
    tasks = [suite.get_task(t) for t in task_ids]
    episodes = [int(x) for x in a.episodes.split(",") if x]
    if a.smoke:
        episodes = episodes[:2]

    set_determinism(0)
    model, processor = load_policy(CHECKPOINT, CODE_DIR)
    tokenizer = processor.tokenizer
    token_slice = action_token_slice(model)

    # ---- position-conditioned visual mean over a fixed 2-states-per-task subset ----
    mean_acc = np.zeros((N_VISUAL, PROJ_DIM), dtype=np.float64)
    mean_n = 0
    mean_env = None
    mean_env_task = None
    for task, ep in mean_states(tasks):
        if mean_env is None or mean_env_task != task.name:
            if mean_env is not None:
                mean_env.close()
            mean_env = SegmentationRenderEnv(
                bddl_file_name=_build_bddl(get_libero_path, task),
                camera_heights=256, camera_widths=256)
            mean_env_task = task.name
        mean_env.seed(ep)
        obs = mean_env.reset()
        _, image_pil = prepare_agentview(obs)
        clean_ids = compute_clean_ids(model, processor, image_pil, task.language)
        h = extract_h(model, processor, image_pil, task.language, clean_ids)
        mean_acc += h.astype(np.float64)
        mean_n += 1
    if mean_env is not None:
        mean_env.close()
    mean = (mean_acc / mean_n).astype(np.float32)
    torch.save({"mean": torch.from_numpy(mean)}, art / "position_conditioned_visual_mean.pt")
    print(json.dumps({"mean_computed": True, "n_states": mean_n,
                      "mean_shape": list(mean.shape)}), flush=True)

    # ---- per-state alignment ----
    rows = []
    for task in tasks:
        env = SegmentationRenderEnv(bddl_file_name=_build_bddl(get_libero_path, task),
                                    camera_heights=256, camera_widths=256)
        for ep in episodes:
            state_id = f"{task.name}_ep{ep:02d}"
            out_npz = art / "align_npz" / f"{state_id}.npz"
            if out_npz.exists():
                print(json.dumps({"skip": state_id}), flush=True)
                continue
            env.seed(ep)
            obs = env.reset()
            _, image_pil = prepare_agentview(obs)
            instruction = task.language
            entities = extract_entities_libero(instruction)
            emb = [embed_phrase(model, tokenizer, e) for e in entities]

            seg = np.asarray(obs["agentview_segmentation_instance"])
            ooi = list(env.obj_of_interest)
            full_mask = np.zeros(seg.shape[:2], dtype=bool)
            source_mask = np.zeros(seg.shape[:2], dtype=bool)
            target_mask = np.zeros(seg.shape[:2], dtype=bool)
            for idx, name in enumerate(ooi[:2]):
                m = (seg[..., 0] == env.instance_to_id[name])
                full_mask |= m
                (source_mask if idx == 0 else target_mask)[m] = True
            oracle_ids, _ = token_ids_from_mask(full_mask.astype(np.uint8))
            src_ids, _ = token_ids_from_mask(source_mask.astype(np.uint8))
            tgt_ids, _ = token_ids_from_mask(target_mask.astype(np.uint8))

            clean_ids = compute_clean_ids(model, processor, image_pil, instruction)
            h = extract_h(model, processor, image_pil, instruction, clean_ids)
            clean_logits = forward_logits(model, processor, image_pil, instruction, clean_ids)
            selected, meta = entity_select(h, emb, K, KMEANS_SEED)

            if not selected:
                r_entity = np.zeros((7, 256), dtype=np.float64)
            else:
                entity_logits = forward_logits(model, processor, image_pil, instruction,
                                               clean_ids, selected, torch.from_numpy(mean))
                r_entity = log_softmax_residual(clean_logits, entity_logits, token_slice).numpy().astype(np.float64)

            oracle_logits = forward_logits(model, processor, image_pil, instruction,
                                           clean_ids, oracle_ids, torch.from_numpy(mean))
            r_oracle = log_softmax_residual(clean_logits, oracle_logits, token_slice).numpy().astype(np.float64)

            align = per_position_cosine(r_entity, r_oracle)  # [7]
            row = {
                "state_id": state_id, "task_id": task.name, "task": task.language,
                "episode": ep, "entities": entities, "n_entities": len(entities),
                "entity_groups": meta["selected_groups"], "entity_tokens": selected,
                "per_entity_score": meta["per_entity_score"], "language_score": meta["language_score"],
                "n_entity_tokens": len(selected), "n_oracle_tokens": len(oracle_ids),
                "n_source_tokens": len(src_ids), "n_target_tokens": len(tgt_ids),
                "align_mean": float(np.mean(align)), "align": align,
            }
            np.savez_compressed(out_npz, **{k: jsonable(v) for k, v in row.items()})
            rows.append(row)
            print(json.dumps({"state_id": state_id, "align_mean": round(row["align_mean"], 4),
                              "n_entity_tokens": row["n_entity_tokens"],
                              "n_oracle_tokens": row["n_oracle_tokens"],
                              "language_score": round(row["language_score"], 4),
                              "entities": entities}), flush=True)
            if a.limit and len(rows) >= a.limit:
                break
        env.close()
        if a.limit and len(rows) >= a.limit:
            break

    # ---- aggregate ----
    all_rows = []
    for npz in sorted((art / "align_npz").glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        all_rows.append({k: _npz_value(d[k]) for k in d.files})
    if not all_rows:
        all_rows = rows

    align_vals = [r["align_mean"] for r in all_rows]
    align_pooled = []
    for r in all_rows:
        align_pooled.extend(np.asarray(r["align"]).tolist())
    agg = {
        "experiment": "LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1",
        "n_states": len(all_rows),
        "n_tasks": len({r["task_id"] for r in all_rows}),
        "align_mean": float(np.mean(align_vals)),
        "align_median": float(np.median(align_vals)),
        "align_pooled_mean": float(np.mean(align_pooled)),  # matches Phase-2B align_pooled
        "language_score_mean": float(np.mean([r["language_score"] for r in all_rows])),
        "n_entity_tokens_mean": float(np.mean([r["n_entity_tokens"] for r in all_rows])),
        "n_oracle_tokens_mean": float(np.mean([r["n_oracle_tokens"] for r in all_rows])),
    }
    task_bd = {}
    for r in all_rows:
        task_bd.setdefault(r["task"], []).append(r["align_mean"])
    agg["task_alignment"] = {t: float(np.mean(v)) for t, v in sorted(task_bd.items())}

    am = agg["align_mean"]
    if am > THRESHOLD_PASS:
        verdict = "PASS_TO_ROLLOUT"
    elif am < THRESHOLD_STOP:
        verdict = "STOP_LIBERO_OBJECT_SEMANTIC_NO_GO"
    else:
        verdict = "MARGINAL_NEEDS_DECISION"
    agg["verdict"] = verdict
    agg["gate"] = {"PASS": am > THRESHOLD_PASS, "STOP": am < THRESHOLD_STOP,
                   "threshold_pass": THRESHOLD_PASS, "threshold_stop": THRESHOLD_STOP,
                   "simpler_reference": 0.586}

    write_json(art / "decision.json", {
        "experiment": "LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1",
        "status": verdict, "date": "2026-08-24", "type": "phase0_offline_alignment",
        "align_mean": agg["align_mean"], "align_pooled_mean": agg["align_pooled_mean"],
        "language_score_mean": agg["language_score_mean"],
        "simpler_reference_align_entity_set_K8": 0.586,
        "gate": agg["gate"], "closed_loop": "NOT_RUN",
        "core_question": "semantic entity counterfactual 是否迁移到 LIBERO-Object？",
    })

    with (art / "alignment_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task_id", "task", "episode", "n_entity_tokens",
                                           "n_oracle_tokens", "n_source_tokens", "n_target_tokens",
                                           "language_score", "align_mean"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    with (art / "selector_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "entities", "entity_groups", "entity_tokens",
                                           "per_entity_score", "language_score"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (json.dumps(r.get(k)) if isinstance(r.get(k), list) else r.get(k))
                        for k in w.fieldnames})

    (art / "config.yaml").write_text(
        "experiment: LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1\n"
        "benchmark: libero_object\n"
        "checkpoint: checkpoints/openvla-7b-finetuned-libero-object/287d6cfdf12d07b1449505f66d9bf3550257e9b3\n"
        "model: OpenVLAForActionPrediction (HF path, third_party/openvla/prismatic/extern/hf)\n"
        "selector: K=8 kmeans + per-entity top-1 cos (Phase-2B)\n"
        "oracle: LIBERO instance segmentation (measurement-only ruler)\n"
        "gate: align_mean > 0.55 PASS / < 0.40 STOP / else MARGINAL\n"
        "simpler_reference_align_entity_set_K8: 0.586\n")

    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

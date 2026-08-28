#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-1.6: goal-conditioned selector (compute + evaluate).

Reuses Phase-1 group_features_npz (v_i, l(sentence), labels, object_ids,
clean_action, r_pixel, masked, g_obj_idx). Extracts goal/object phrases
(rule-based, no LLM), computes 6 language embeddings per unique instruction
(object / noun_only / goal / goal_phrase / wrong_goal, plus reused sentence l),
scores each against the 8 group visual features, and evaluates oracle recovery /
alignment / task breakdown / gates A-B-C + 3 ablations.

No rollout, no training, no λ. object mask used only for evaluation.

Run:
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/goal_selector.py \
      --artifact artifacts/semantic_token_cd_phase1_6_goal_selector_v1 --split confirmation
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from research.cw_lpcd.core import CHECKPOINT, load_model
from research.cw_lpcd.engine import random_seed_for
from research.cw_lpcd.metrics import per_position_cosine, state_cosine

STATES_LOCK = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl")
FEAT_NPZ = Path("artifacts/semantic_token_cd_phase1_selector_v1/group_features_npz")
SPLIT = Path("artifacts/cat_cd_phase0_v1/FROZEN_SPLIT.json")

PREPOSITIONS = {"into", "in", "on", "onto", "near", "to", "next", "from", "under", "over",
                "behind", "beside", "above", "below", "inside", "at", "with"}
DETERMINERS = {"the", "a", "an"}
K = 8


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extract_goal(instruction: str) -> dict[str, str]:
    """Rule-based goal extraction -> {object, goal, goal_phrase, prep}.
    goal = last noun phrase (destination/container/目标位置); object = first noun.
    No preposition -> single-target/drawer task: goal = object (fallback)."""
    words = instruction.lower().split()
    rest = words[1:]  # drop leading verb
    prep_pos = [i for i, w in enumerate(rest) if w in PREPOSITIONS]
    if not prep_pos:
        obj = " ".join(w for w in rest if w not in DETERMINERS)
        return {"object": obj, "goal": obj, "goal_phrase": obj, "prep": ""}
    first, last = prep_pos[0], prep_pos[-1]
    obj = " ".join(w for w in rest[:first] if w not in DETERMINERS)
    goal = " ".join(w for w in rest[last + 1:] if w not in DETERMINERS)
    prep = rest[last]
    return {"object": obj, "goal": goal, "goal_phrase": f"{prep} {goal}", "prep": prep}


def embed_phrase(model, tokenizer, text: str) -> np.ndarray:
    """Mean-pooled LLM token embedding of a phrase (no special tokens)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    ids_t = torch.tensor([ids], dtype=torch.long, device=model.device)
    emb = model.language_model.model.embed_tokens(ids_t)
    return emb[0].mean(dim=0).detach().float().cpu().numpy()


def log_softmax_residual(clean: np.ndarray, branch: np.ndarray) -> np.ndarray:
    ca = torch.tensor(clean, dtype=torch.float64)
    ba = torch.tensor(branch, dtype=torch.float64)
    return (torch.log_softmax(ca, -1) - torch.log_softmax(ba, -1)).numpy()


def _mean(pool):
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    return float(np.mean(pool)) if pool.size else float("nan")


def _med(pool):
    pool = np.asarray(pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    return float(np.median(pool)) if pool.size else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    a = p.parse_args()
    art = a.artifact.resolve()
    split = json.loads(SPLIT.read_text())
    (art / "FROZEN_SPLIT.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
    state_ids = split[key]

    states = {r["state_id"]: r for r in read_jsonl(STATES_LOCK)}

    model, processor = load_model(CHECKPOINT)
    tokenizer = processor.tokenizer

    # ---- unique instructions -> goal/object phrase extraction ----
    instr_set = {}
    for sid in state_ids:
        instr_set.setdefault(states[sid]["instruction"], None)
    phrases = {instr: extract_goal(instr) for instr in instr_set}

    # ---- wrong-goal mapping: deterministic rotation over sorted unique goals ----
    unique_goals = sorted({phrases[i]["goal"] for i in phrases})
    wrong_map = {g: unique_goals[(k + 1) % len(unique_goals)] for k, g in enumerate(unique_goals)}

    # ---- embeddings ----
    emb = {}  # instruction -> {variant: np.ndarray}
    for instr in instr_set:
        ph = phrases[instr]
        alln = " ".join(w for w in instr.lower().split()[1:] if w not in DETERMINERS and w not in PREPOSITIONS)
        emb[instr] = {
            "object": embed_phrase(model, tokenizer, ph["object"]),
            "noun_only": embed_phrase(model, tokenizer, alln),
            "goal": embed_phrase(model, tokenizer, ph["goal"]),
            "goal_phrase": embed_phrase(model, tokenizer, ph["goal_phrase"]),
            "wrong_goal": embed_phrase(model, tokenizer, wrong_map[ph["goal"]]),
        }

    (art / "goal_phrase.json").write_text(json.dumps({
        "phrases": phrases, "wrong_goal_map": wrong_map, "unique_goals": unique_goals,
        "n_unique_instructions": len(instr_set),
    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"n_unique_instructions": len(instr_set), "n_unique_goals": len(unique_goals)}, ensure_ascii=False))

    # ---- per-state scoring ----
    variants = ("object", "noun_only", "goal", "goal_phrase", "wrong_goal")
    selectors = ("sentence",) + variants + ("random", "oracle")
    rows = []
    n = 0
    for sid in state_ids:
        f = FEAT_NPZ / f"{sid}.npz"
        if not f.exists():
            continue
        n += 1
        d = np.load(f)
        v = d["v"].astype(np.float64)
        l_stored = d["l"].astype(np.float64)
        oracle = int(d["g_obj_idx"][0])
        labels = d["labels"].astype(np.int64)
        obj = set(int(x) for x in d["object_ids"].tolist())
        clean = d["clean_action"]
        r_pixel = d["r_pixel"].astype(np.float64)
        masked = d["masked"]
        instr = states[sid]["instruction"]

        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        scores = {}
        for var in variants:
            e = emb[instr][var]
            en = e / (np.linalg.norm(e) + 1e-8)
            scores[var] = vn @ en
        ln = l_stored / (np.linalg.norm(l_stored) + 1e-8)
        scores["sentence"] = vn @ ln
        rnd_g = int(np.random.default_rng(random_seed_for(sid) + 2000).integers(0, K))

        chosen = {sel: int(np.argmax(scores[sel])) for sel in ("sentence",) + variants}
        chosen["random"] = rnd_g
        chosen["oracle"] = oracle

        for sel in selectors:
            g = chosen[sel]
            grp = set(int(x) for x in np.flatnonzero(labels == g).tolist())
            iou = len(grp & obj) / len(grp | obj) if (grp | obj) else 0.0
            r = log_softmax_residual(clean, masked[g])
            pc = per_position_cosine(r, r_pixel)
            rows.append({
                "state_id": sid, "task": str(d["task"]), "selector": sel, "group": g,
                "hit": int(g == oracle), "iou": iou,
                "cos_pos_mean": float(np.mean(pc)), "cos_state": state_cosine(r, r_pixel),
            })

    # ---- aggregate + pooled alignment ----
    agg = {}
    for sel in selectors:
        rr = [r for r in rows if r["selector"] == sel]
        agg[sel] = {
            "oracle_recovery": _mean([r["hit"] for r in rr]),
            "iou_median": _med([r["iou"] for r in rr]),
            "cos_pos_mean": _mean([r["cos_pos_mean"] for r in rr]),
            "cos_state_mean": _mean([r["cos_state"] for r in rr]),
        }
    pooled_mean = {}
    for sel in selectors:
        vals = []
        for sid in state_ids:
            f = FEAT_NPZ / f"{sid}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            v = d["v"].astype(np.float64); l_stored = d["l"].astype(np.float64)
            vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            instr = states[sid]["instruction"]
            sc = {}
            for var in variants:
                e = emb[instr][var]
                sc[var] = vn @ (e / (np.linalg.norm(e) + 1e-8))
            sc["sentence"] = vn @ (l_stored / (np.linalg.norm(l_stored) + 1e-8))
            g = {"sentence": int(np.argmax(sc["sentence"])),
                 **{var: int(np.argmax(sc[var])) for var in variants},
                 "oracle": int(d["g_obj_idx"][0]),
                 "random": int(np.random.default_rng(random_seed_for(sid) + 2000).integers(0, K))}[sel]
            r = log_softmax_residual(d["clean_action"], d["masked"][g])
            vals.extend(per_position_cosine(r, d["r_pixel"].astype(np.float64)).tolist())
        pooled_mean[sel] = _mean(vals)

    # ---- task breakdown ----
    task_rows = {}
    for r in rows:
        task_rows.setdefault(r["task"], {}).setdefault(r["selector"], []).append(r["hit"])
    task_breakdown = []
    for t in sorted(task_rows):
        task_breakdown.append({
            "task": t, "n_states": len(task_rows[t]["oracle"]),
            "sentence": _mean(task_rows[t]["sentence"]),
            "object": _mean(task_rows[t]["object"]),
            "noun_only": _mean(task_rows[t]["noun_only"]),
            "goal": _mean(task_rows[t]["goal"]),
            "goal_phrase": _mean(task_rows[t]["goal_phrase"]),
            "wrong_goal": _mean(task_rows[t]["wrong_goal"]),
            "random": _mean(task_rows[t]["random"]),
        })

    # ---- gates / stop / verdict ----
    noun_rec = agg["noun_only"]["oracle_recovery"]
    goal_rec = agg["goal"]["oracle_recovery"]
    noun_align = pooled_mean["noun_only"]
    goal_align = pooled_mean["goal"]
    obj_rec = agg["object"]["oracle_recovery"]

    gateA = goal_rec > noun_rec + 0.05
    gateB = goal_align > noun_align
    failed_tasks = [tb for tb in task_breakdown if tb["noun_only"] < 0.30]
    gateC = any(tb["goal"] >= 0.50 for tb in failed_tasks)

    stop_ceiling = abs(goal_rec - noun_rec) <= 0.05
    stop_no_align = (goal_rec > noun_rec + 0.05) and (goal_align <= noun_align)
    stop_multiobj = not gateC

    stop_hit = stop_ceiling or stop_no_align or stop_multiobj
    verdict = "PASS_TO_ROLLOUT" if (gateA and gateB and gateC and not stop_hit) else "STOP_GOAL_SELECTOR_NO_GO"

    # ---- write CSVs ----
    with (art / "selector_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "selector", "group", "hit", "iou",
                                           "cos_pos_mean", "cos_state"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "n_states", "sentence", "object", "noun_only",
                                           "goal", "goal_phrase", "wrong_goal", "random"])
        w.writeheader()
        for tb in task_breakdown:
            w.writerow(tb)

    # alignment ablation A2 (phrase length): goal vs goal_phrase vs sentence
    # recovery ablation A1 (goal vs object): goal vs object
    # wrong-goal ablation A3: goal vs wrong_goal
    results = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE1_6_GOAL_SELECTOR_V1",
        "split": a.split, "n_states": n, "n_unique_instructions": len(instr_set),
        "verdict": verdict, "stop_rule_hit": stop_hit,
        "aggregate": agg,
        "pooled_alignment": pooled_mean,
        "gates": {
            "A_recovery_goal_vs_noun": {"PASS": bool(gateA), "goal": goal_rec, "noun_only": noun_rec, "delta": goal_rec - noun_rec},
            "B_alignment_goal_vs_noun": {"PASS": bool(gateB), "goal": goal_align, "noun_only": noun_align},
            "C_failed_task_rescue": {"PASS": bool(gateC), "failed_tasks": [tb["task"] for tb in failed_tasks],
                                     "failed_goal": {tb["task"]: tb["goal"] for tb in failed_tasks}},
        },
        "ablations": {
            "A1_goal_vs_object": {"goal_rec": goal_rec, "object_rec": obj_rec, "goal_align": goal_align,
                                  "object_align": pooled_mean["object"], "goal_gt_object": goal_rec > obj_rec},
            "A2_phrase_length": {"noun": goal_rec, "phrase": agg["goal_phrase"]["oracle_recovery"],
                                 "full": agg["sentence"]["oracle_recovery"],
                                 "noun_align": goal_align, "phrase_align": pooled_mean["goal_phrase"],
                                 "full_align": pooled_mean["sentence"]},
            "A3_wrong_goal_control": {"goal_rec": goal_rec, "wrong_goal_rec": agg["wrong_goal"]["oracle_recovery"],
                                      "goal_align": goal_align, "wrong_goal_align": pooled_mean["wrong_goal"],
                                      "goal_gt_wrong": goal_rec > agg["wrong_goal"]["oracle_recovery"]},
        },
        "stop_flags": {
            "goal_approx_object": bool(stop_ceiling),
            "recovery_no_align": bool(stop_no_align),
            "multiobject_unresolved": bool(stop_multiobj),
        },
        "task_breakdown": task_breakdown,
    }
    (art / "selector_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE1_6_GOAL_SELECTOR_V1",
        "status": verdict, "split": a.split, "n_states": n, "date": "2026-08-24",
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "core_numbers": {
            "recovery_oracle": agg["oracle"]["oracle_recovery"],
            "recovery_sentence": agg["sentence"]["oracle_recovery"],
            "recovery_object": obj_rec, "recovery_noun_only": noun_rec, "recovery_goal": goal_rec,
            "recovery_goal_phrase": agg["goal_phrase"]["oracle_recovery"],
            "recovery_wrong_goal": agg["wrong_goal"]["oracle_recovery"],
            "alignment_oracle": pooled_mean["oracle"], "alignment_sentence": pooled_mean["sentence"],
            "alignment_object": pooled_mean["object"], "alignment_noun_only": noun_align,
            "alignment_goal": goal_align,
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

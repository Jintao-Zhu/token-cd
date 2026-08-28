#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-1.5: object phrase grounding selector (compute + evaluate).

Reuses Phase-1 group_features_npz (v_i, sentence-l, D, oracle, object_ids,
clean_action, r_pixel, masked). Extracts noun phrases from the 9 fixed
instructions (rule-based; spaCy/nltk unavailable offline), computes 4 language
embeddings (sentence / noun_only / first_noun / last_noun) via the frozen LLM
token embedding layer, scores each against the 8 group visual features, and
evaluates oracle recovery / alignment / task breakdown / gates.

No rollout, no training, no λ. object mask used only for evaluation.

Run:
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/object_phrase.py \
      --artifact artifacts/semantic_token_cd_phase1_5_object_grounding_v1 --split confirmation
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from research.cw_lpcd.core import PCD_ROOT, CHECKPOINT, load_model
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


def extract_phrases(instruction: str) -> dict[str, str]:
    """Rule-based noun-phrase extraction -> {first_noun, last_noun, all_nouns} phrase strings."""
    words = instruction.lower().split()
    rest = words[1:]  # drop leading verb
    prep_pos = [i for i, w in enumerate(rest) if w in PREPOSITIONS]
    if not prep_pos:
        obj = " ".join(w for w in rest if w not in DETERMINERS)
        return {"first_noun": obj, "last_noun": obj, "all_nouns": obj}
    first, last = prep_pos[0], prep_pos[-1]
    obj = " ".join(w for w in rest[:first] if w not in DETERMINERS)
    lst = " ".join(w for w in rest[last + 1:] if w not in DETERMINERS)
    alln = " ".join(w for w in rest if w not in DETERMINERS and w not in PREPOSITIONS)
    return {"first_noun": obj, "last_noun": lst, "all_nouns": alln}


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

    # ---- unique instructions -> phrase embeddings ----
    instr_set = {}
    for sid in state_ids:
        instr_set.setdefault(states[sid]["instruction"], None)
    phrase_emb = {}   # instruction -> {variant: np.ndarray}
    phrase_dump = {}
    for instr in instr_set:
        ph = extract_phrases(instr)
        emb = {
            "sentence": embed_phrase(model, tokenizer, instr),
            "noun_only": embed_phrase(model, tokenizer, ph["all_nouns"]),
            "first_noun": embed_phrase(model, tokenizer, ph["first_noun"]),
            "last_noun": embed_phrase(model, tokenizer, ph["last_noun"]),
        }
        # reuse Phase-1 stored sentence-l for an exact baseline (mean over tokens incl. BOS)
        phrase_emb[instr] = emb
        phrase_dump[instr] = ph
    (art / "phrase_extraction.json").write_text(json.dumps(phrase_dump, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_unique_instructions": len(instr_set), "phrases": phrase_dump}, ensure_ascii=False))

    # ---- per-state scoring ----
    variants = ("sentence", "noun_only", "first_noun", "last_noun")
    selectors = variants + ("random", "oracle")
    rows = []
    n = 0
    for sid in state_ids:
        f = FEAT_NPZ / f"{sid}.npz"
        if not f.exists():
            continue
        n += 1
        d = np.load(f)
        v = d["v"].astype(np.float64)
        l_stored = d["l"].astype(np.float64)  # Phase-1 sentence embedding
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
            e = phrase_emb[instr][var] if var != "sentence" else l_stored
            en = e / (np.linalg.norm(e) + 1e-8)
            scores[var] = vn @ en
        rnd_g = int(np.random.default_rng(random_seed_for(sid) + 2000).integers(0, K))

        chosen = {var: int(np.argmax(scores[var])) for var in variants}
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

    # ---- aggregate ----
    agg = {}
    for sel in selectors:
        rr = [r for r in rows if r["selector"] == sel]
        agg[sel] = {
            "oracle_recovery": _mean([r["hit"] for r in rr]),
            "iou_median": _med([r["iou"] for r in rr]),
            "cos_pos_mean": _mean([r["cos_pos_mean"] for r in rr]),
            "cos_state_mean": _mean([r["cos_state"] for r in rr]),
        }
    # pooled per-position alignment
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
            scores = {}
            for var in variants:
                e = phrase_emb[instr][var] if var != "sentence" else l_stored
                en = e / (np.linalg.norm(e) + 1e-8)
                scores[var] = vn @ en
            g = {"sentence": int(np.argmax(scores["sentence"])), "noun_only": int(np.argmax(scores["noun_only"])),
                 "first_noun": int(np.argmax(scores["first_noun"])), "last_noun": int(np.argmax(scores["last_noun"])),
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
            "noun_only": _mean(task_rows[t]["noun_only"]),
            "first_noun": _mean(task_rows[t]["first_noun"]),
            "last_noun": _mean(task_rows[t]["last_noun"]),
            "random": _mean(task_rows[t]["random"]),
        })

    # ---- gates (first_noun = object phrase) ----
    sen_rec = agg["sentence"]["oracle_recovery"]
    first_rec = agg["first_noun"]["oracle_recovery"]
    sen_align = pooled_mean["sentence"]
    first_align = pooled_mean["first_noun"]

    gateA = first_rec > sen_rec + 0.10
    gateB = first_align > sen_align
    failed_tasks = [tb for tb in task_breakdown if tb["sentence"] < 0.30]
    gateC = any(tb["first_noun"] >= 0.50 for tb in failed_tasks)

    # stop flags
    stop_dilution_not_main = (abs(first_rec - sen_rec) <= 0.05)
    stop_recovery_no_align = (first_rec > sen_rec + 0.10) and (first_align <= sen_align)
    stop_align_few_tasks = (first_align > sen_align) and (not gateC)

    stop_hit = stop_dilution_not_main or stop_recovery_no_align or stop_align_few_tasks
    verdict = "PASS_TO_ROLLOUT" if (gateA and gateB and gateC and not stop_hit) else "STOP_SEMANTIC_SELECTOR_NO_GO"

    # ---- write CSVs ----
    with (art / "selector_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "selector", "group", "hit", "iou",
                                           "cos_pos_mean", "cos_state"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "n_states", "sentence", "noun_only",
                                           "first_noun", "last_noun", "random"])
        w.writeheader()
        for tb in task_breakdown:
            w.writerow(tb)

    results = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE1_5_OBJECT_GROUNDING_V1",
        "split": a.split, "n_states": n,
        "verdict": verdict, "stop_rule_hit": stop_hit,
        "aggregate": agg,
        "pooled_alignment": pooled_mean,
        "gates": {
            "A_recovery": {"PASS": bool(gateA), "first_noun": first_rec, "sentence": sen_rec, "delta": first_rec - sen_rec},
            "B_alignment": {"PASS": bool(gateB), "first_noun": first_align, "sentence": sen_align},
            "C_failed_task_rescue": {"PASS": bool(gateC), "failed_tasks": [tb["task"] for tb in failed_tasks],
                                     "failed_first_noun": {tb["task"]: tb["first_noun"] for tb in failed_tasks}},
        },
        "stop_flags": {
            "dilution_not_main": bool(stop_dilution_not_main),
            "recovery_no_align": bool(stop_recovery_no_align),
            "align_few_tasks": bool(stop_align_few_tasks),
        },
        "task_breakdown": task_breakdown,
    }
    (art / "selector_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE1_5_OBJECT_GROUNDING_V1",
        "status": verdict, "split": a.split, "n_states": n, "date": "2026-08-23",
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "core_numbers": {
            "recovery_sentence": sen_rec, "recovery_first_noun": first_rec,
            "recovery_noun_only": agg["noun_only"]["oracle_recovery"],
            "recovery_last_noun": agg["last_noun"]["oracle_recovery"],
            "alignment_sentence": sen_align, "alignment_first_noun": first_align,
            "alignment_oracle": pooled_mean["oracle"],
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

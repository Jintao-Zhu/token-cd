#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-2A relation grouping (compute, GPU, resumable).

Per Confirmation state:
  1. extract projector h_i [256,4096] (1 clean forward), cache to h_npz/.
  2. build fine groups (K=16/32) from stored labels_kmeans_K; group feature v_i.
  3. construct candidates: single (noun), relation_pair (e_rel), oracle_pair
     (argmax IoU union), random_pair, langfree_pair (argmax cos(v_i,v_j)).
  4. masked branch per candidate via forward_masked on the group/pair-union
     patch index set -> residual r = log_softmax(clean) - log_softmax(branch).
  5. dump per-state npz; a second pass aggregates recovery/alignment/ablations.

No rollout, no training, no λ. object mask used only for oracle definition.

Run (smoke):
  cd /home/leju-suzhou/zjt_ws/token-cd
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/relation_grouping.py \
      --artifact artifacts/semantic_token_cd_phase2a_relation_grouping_v1 \
      --split confirmation --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from research.cw_lpcd.core import PCD_ROOT, CHECKPOINT, load_model, action_token_slice
from research.cw_lpcd.engine import load_position_mean, random_seed_for
from research.cw_lpcd.metrics import per_position_cosine
from research.ar_cat_cd.core import forward_masked
from research.semantic_token_cd.core import extract_features

STATES_LOCK = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl")
MEAN_PATH = Path("artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/position_conditioned_visual_mean.pt")
SEMANTIC_NPZ = Path("artifacts/semantic_token_cd_phase0_v1/semantic_npz")
SPLIT = Path("artifacts/cat_cd_phase0_v1/FROZEN_SPLIT.json")

PREPOSITIONS = {"into", "in", "on", "onto", "near", "to", "next", "from", "under", "over",
                "behind", "beside", "above", "below", "inside", "at", "with"}
DETERMINERS = {"the", "a", "an"}
K_VALUES = [16, 32]
CANDIDATES = ("single", "relation", "oracle", "random", "langfree")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extract_relation(instruction: str) -> dict[str, str]:
    """Decompose instruction -> {action, source, relation, target, rel_phrase, noun_phrase}."""
    words = instruction.lower().split()
    verb = words[0]
    rest = words[1:]
    prep_pos = [i for i, w in enumerate(rest) if w in PREPOSITIONS]
    if not prep_pos:
        obj = " ".join(w for w in rest if w not in DETERMINERS)
        return {"action": verb, "source": obj, "relation": "", "target": obj,
                "rel_phrase": obj, "noun_phrase": obj}
    first, last = prep_pos[0], prep_pos[-1]
    src = " ".join(w for w in rest[:first] if w not in DETERMINERS)
    rel = rest[last]
    tgt = " ".join(w for w in rest[last + 1:] if w not in DETERMINERS)
    rel_phrase = " ".join(w for w in rest if w not in DETERMINERS)            # source rel target
    noun_phrase = " ".join(w for w in rest if w not in DETERMINERS and w not in PREPOSITIONS)  # source target
    return {"action": verb, "source": src, "relation": rel, "target": tgt,
            "rel_phrase": rel_phrase, "noun_phrase": noun_phrase}


def embed_phrase(model, tokenizer, text: str) -> np.ndarray:
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


def group_sets(labels: np.ndarray, K: int) -> list[set]:
    return [set(int(x) for x in np.flatnonzero(labels == k).tolist()) for k in range(K)]


def iou_of(groups: list[set], pair: tuple[int, int], obj: set) -> float:
    u = groups[pair[0]] | groups[pair[1]]
    inter = len(u & obj)
    union = len(u | obj)
    return inter / union if union else 0.0


def pair_scores_rel(v: np.ndarray, e_rel: np.ndarray) -> tuple[int, int]:
    """argmax_{i<j} cos(mean(v_i,v_j), e_rel)."""
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    en = e_rel / (np.linalg.norm(e_rel) + 1e-8)
    K = v.shape[0]
    best, bi, bj = -np.inf, 0, 1
    for i in range(K):
        for j in range(i + 1, K):
            s = (vn[i] + vn[j]) @ en / (np.linalg.norm(vn[i] + vn[j]) + 1e-8)
            if s > best:
                best, bi, bj = s, i, j
    return int(bi), int(bj)


def pair_scores_langfree(v: np.ndarray) -> tuple[int, int]:
    """argmax_{i<j} cos(v_i, v_j) (feature similarity, no language)."""
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    K = v.shape[0]
    best, bi, bj = -np.inf, 0, 1
    for i in range(K):
        for j in range(i + 1, K):
            s = float(vn[i] @ vn[j])
            if s > best:
                best, bi, bj = s, i, j
    return int(bi), int(bj)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pcd-root", type=Path, default=PCD_ROOT)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--split", choices=["selection", "confirmation"], default="confirmation")
    p.add_argument("--limit", type=int)
    p.add_argument("--state-id", type=str)
    p.add_argument("--mean-path", type=Path, default=MEAN_PATH)
    p.add_argument("--eval-only", action="store_true", help="skip compute, only aggregate existing npz")
    a = p.parse_args()

    pcd_root = a.pcd_root.resolve()
    art = a.artifact.resolve()
    rel_dir = art / "relation_npz"
    h_dir = art / "h_npz"
    rel_dir.mkdir(parents=True, exist_ok=True)
    h_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT.read_text())
    (art / "FROZEN_SPLIT.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    if a.state_id:
        state_ids = [a.state_id]
    else:
        key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
        state_ids = split[key][: a.limit] if a.limit else split[key]

    states = {r["state_id"]: r for r in read_jsonl(STATES_LOCK)}

    # ---------- language embedding cache ----------
    model = processor = tokenizer = None
    if not a.eval_only:
        torch.manual_seed(20260823)
        torch.cuda.manual_seed_all(20260823)
        model, processor = load_model(CHECKPOINT)
        tokenizer = processor.tokenizer
        mean = load_position_mean(a.mean_path.resolve()).to(model.device, dtype=torch.bfloat16)
        token_slice = action_token_slice(model)

        instr_set = {}
        for sid in state_ids:
            instr_set.setdefault(states[sid]["instruction"], None)
        emb_cache = {}
        phrases = {}
        for instr in instr_set:
            r = extract_relation(instr)
            phrases[instr] = r
            emb_cache[instr] = {
                "noun": embed_phrase(model, tokenizer, r["noun_phrase"]),
                "rel": embed_phrase(model, tokenizer, r["rel_phrase"]),
                "goal": embed_phrase(model, tokenizer, r["target"]),
            }
        (art / "relation_phrases.json").write_text(json.dumps(phrases, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        print(json.dumps({"model_loaded": True, "n_states": len(state_ids), "split": a.split,
                          "n_unique_instructions": len(instr_set)}), flush=True)

        n_fwd = 0
        t0 = time.time()
        for ordinal, sid in enumerate(state_ids):
            out_npz = rel_dir / f"{sid}.npz"
            if out_npz.exists():
                print(json.dumps({"skip": sid, "done": ordinal + 1}), flush=True)
                continue
            sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
            row = states[sid]
            instruction = row["instruction"]
            clean_img = cv2.cvtColor(cv2.imread(str(pcd_root / row["clean_path"])), cv2.COLOR_BGR2RGB)
            clean_ids = torch.tensor(sem["clean_ids"], dtype=torch.long, device=model.device).unsqueeze(0)
            clean_action = sem["clean_action"].astype(np.float64)
            r_pixel = sem["r_pixel"].astype(np.float64)
            object_ids = set(int(x) for x in sem["object_ids"].tolist())

            # h_i (cached)
            h_npz = h_dir / f"{sid}.npz"
            if h_npz.exists():
                h = np.load(h_npz)["h"].astype(np.float64)
            else:
                h_i, _ = extract_features(model, processor, clean_img, instruction, clean_ids)
                h = h_i.numpy().astype(np.float64)
                np.savez_compressed(h_npz, h=h_i.numpy().astype(np.float16))
            n_fwd += 1

            e_noun = emb_cache[instruction]["noun"]
            e_rel = emb_cache[instruction]["rel"]
            out = {"state_id": sid, "task": str(sem["task"])}

            for K in K_VALUES:
                labels = sem[f"labels_kmeans_{K}"].astype(np.int64)
                masked = sem[f"masked_kmeans_{K}"]           # [K,7,256]
                v = np.zeros((K, h.shape[1]), dtype=np.float64)
                for k in range(K):
                    idx = np.flatnonzero(labels == k)
                    v[k] = h[idx].mean(axis=0) if len(idx) else 0.0
                groups = group_sets(labels, K)

                vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
                e_noun_n = e_noun / (np.linalg.norm(e_noun) + 1e-8)
                single = int(np.argmax(vn @ e_noun_n))
                rel_pair = pair_scores_rel(v, e_rel)
                langfree_pair = pair_scores_langfree(v)

                # oracle pair = argmax IoU(union, PCD mask)
                obest, oi, oj = -1.0, 0, 1
                for i in range(K):
                    for j in range(i + 1, K):
                        io = iou_of(groups, (i, j), object_ids)
                        if io > obest:
                            obest, oi, oj = io, i, j
                oracle_pair = (int(oi), int(oj))

                rng = np.random.default_rng(random_seed_for(sid) + 3000 + K)
                rand_pair = (int(rng.integers(0, K)), int(rng.integers(0, K)))
                while rand_pair[0] == rand_pair[1]:
                    rand_pair = (int(rng.integers(0, K)), int(rng.integers(0, K)))

                picks = {"single": single, "relation": rel_pair, "oracle": oracle_pair,
                         "random": rand_pair, "langfree": langfree_pair}

                oracle_single = int(sem[f"g_obj_idx_kmeans_{K}"][0])
                for cand in CANDIDATES:
                    if cand == "single":
                        branch = masked[single].astype(np.float64)
                        union = groups[single]
                        hit = 1 if single == oracle_single else 0
                    else:
                        pa, pb = picks[cand]
                        union = groups[pa] | groups[pb]
                        ma, _ = forward_masked(model, processor, clean_img, instruction, clean_ids,
                                               sorted(union), mean)
                        branch = ma[:, token_slice].numpy().astype(np.float64)
                        n_fwd += 1
                        hit = 1 if (cand == "oracle" or picks[cand] == oracle_pair) else 0
                    r = log_softmax_residual(clean_action, branch)
                    inter = len(union & object_ids)
                    un = len(union | object_ids)
                    iou = inter / un if un else 0.0
                    out[f"{cand}_sel_{K}"] = np.asarray(picks[cand] if cand != "single" else [single], dtype=np.int64)
                    out[f"{cand}_r_{K}"] = r.astype(np.float32)
                    out[f"{cand}_iou_{K}"] = np.asarray([iou], dtype=np.float64)
                    out[f"{cand}_hit_{K}"] = np.asarray([hit], dtype=np.int64)

            np.savez_compressed(out_npz, **out)
            dt = time.time() - t0
            print(json.dumps({"state_id": sid, "done": ordinal + 1, "of": len(state_ids),
                              "sec": round(dt, 1), "fwd": n_fwd, "rate": round(n_fwd / dt, 1)}), flush=True)

        print(json.dumps({"DONE_COMPUTE": len(state_ids), "total_fwd": n_fwd,
                          "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    # ---------- aggregate (reads all relation_npz) ----------
    rows = []
    for sid in state_ids:
        f = rel_dir / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
        r_pixel = sem["r_pixel"].astype(np.float64)
        task = str(d["task"])
        for K in K_VALUES:
            for cand in CANDIDATES:
                r = d[f"{cand}_r_{K}"].astype(np.float64)
                pc = per_position_cosine(r, r_pixel)
                rows.append({
                    "state_id": sid, "task": task, "K": K, "candidate": cand,
                    "iou": float(d[f"{cand}_iou_{K}"][0]),
                    "hit": int(d[f"{cand}_hit_{K}"][0]),
                    "cos_pos_mean": float(np.mean(pc)),
                })

    # pooled alignment over states×7 (recompute from r for the pooled metric)
    pooled = {}
    for K in K_VALUES:
        for cand in CANDIDATES:
            vals = []
            for sid in state_ids:
                f = rel_dir / f"{sid}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
                r = d[f"{cand}_r_{K}"].astype(np.float64)
                vals.extend(per_position_cosine(r, sem["r_pixel"].astype(np.float64)).tolist())
            pooled[(K, cand)] = _mean(vals)

    agg = {}
    for K in K_VALUES:
        for cand in CANDIDATES:
            rr = [r for r in rows if r["K"] == K and r["candidate"] == cand]
            agg[f"{cand}_K{K}"] = {
                "iou_median": _med([r["iou"] for r in rr]),
                "hit_mean": _mean([r["hit"] for r in rr]),
                "cos_pos_mean": _mean([r["cos_pos_mean"] for r in rr]),
                "align_pooled": pooled[(K, cand)],
            }

    # task breakdown for hard tasks
    task_bd = {}
    for r in rows:
        key = (r["task"], r["K"], r["candidate"])
        task_bd.setdefault(key, []).append(r["iou"])
    task_breakdown = []
    for t in sorted({r["task"] for r in rows}):
        for K in K_VALUES:
            row = {"task": t, "K": K}
            for cand in CANDIDATES:
                vv = task_bd.get((t, K, cand), [])
                row[f"{cand}_iou"] = _mean(vv)
            task_breakdown.append(row)

    # gates (K=16 primary)
    Kp = 16
    single_iou = agg[f"single_K{Kp}"]["iou_median"]
    rel_iou = agg[f"relation_K{Kp}"]["iou_median"]
    single_align = agg[f"single_K{Kp}"]["align_pooled"]
    rel_align = agg[f"relation_K{Kp}"]["align_pooled"]
    oracle_align = agg[f"oracle_K{Kp}"]["align_pooled"]
    rand_align = agg[f"random_K{Kp}"]["align_pooled"]
    langfree_align = agg[f"langfree_K{Kp}"]["align_pooled"]

    gateA = rel_iou > single_iou + 0.10
    gateB = rel_align > 0.60
    hard_tasks = ["widowx_put_eggplant_in_basket", "widowx_spoon_on_towel"]
    htb = {t: next((b for b in task_breakdown if b["task"] == t and b["K"] == Kp), None) for t in hard_tasks}
    gateC = any(htb[t] is not None and htb[t]["relation_iou"] > htb[t]["single_iou"] for t in hard_tasks)

    stop_pair_approx_single = abs(rel_align - single_align) <= 0.02
    stop_mask_area = (rel_align - rand_align) <= 0.02
    stop_no_generalize = not gateC

    stop_hit = stop_pair_approx_single or stop_mask_area or stop_no_generalize
    verdict = "PASS_TO_ROLLOUT" if (gateA and gateB and gateC and not stop_hit) else "STOP_RELATION_GROUPING_NO_GO"

    # single-group oracle recovery reference: noun-only single group at K=8 = 71.1% (Phase-1.5)
    results = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE2A_RELATION_GROUPING_V1",
        "split": a.split, "n_states": len([s for s in state_ids if (rel_dir / f"{s}.npz").exists()]),
        "verdict": verdict, "stop_rule_hit": stop_hit,
        "aggregate": agg,
        "pooled_alignment": {f"{cand}_K{K}": pooled[(K, cand)] for K in K_VALUES for cand in CANDIDATES},
        "gates": {
            "A_recovery_coverage": {"PASS": bool(gateA), "relation_iou": rel_iou, "single_iou": single_iou,
                                    "delta": rel_iou - single_iou},
            "B_alignment": {"PASS": bool(gateB), "relation_align": rel_align, "oracle_align": oracle_align},
            "C_hard_task": {"PASS": bool(gateC),
                            "eggplant": {"single_iou": htb[hard_tasks[0]]["single_iou"] if htb[hard_tasks[0]] else None,
                                         "relation_iou": htb[hard_tasks[0]]["relation_iou"] if htb[hard_tasks[0]] else None},
                            "spoon": {"single_iou": htb[hard_tasks[1]]["single_iou"] if htb[hard_tasks[1]] else None,
                                      "relation_iou": htb[hard_tasks[1]]["relation_iou"] if htb[hard_tasks[1]] else None}},
        },
        "ablations": {
            "A_single_vs_pair": {"single_align": single_align, "relation_align": rel_align, "pair_gt_single": rel_align > single_align},
            "B_random_pair": {"relation_align": rel_align, "random_align": rand_align,
                              "lang_gt_random": rel_align > rand_align + 0.02},
            "C_langfree": {"relation_align": rel_align, "langfree_align": langfree_align,
                           "lang_gt_langfree": rel_align > langfree_align},
        },
        "stop_flags": {
            "pair_approx_single": bool(stop_pair_approx_single),
            "mask_area_redundancy": bool(stop_mask_area),
            "no_generalize": bool(stop_no_generalize),
        },
        "task_breakdown": task_breakdown,
    }
    (art / "selector_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE2A_RELATION_GROUPING_V1",
        "status": verdict, "split": a.split, "date": "2026-08-24",
        "n_states": results["n_states"],
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "core_numbers": {
            "align_single_K16": single_align, "align_relation_K16": rel_align,
            "align_oracle_K16": oracle_align, "align_random_K16": rand_align, "align_langfree_K16": langfree_align,
            "iou_single_K16": single_iou, "iou_relation_K16": rel_iou,
            "align_single_K32": agg["single_K32"]["align_pooled"], "align_relation_K32": agg["relation_K32"]["align_pooled"],
            "align_oracle_K32": agg["oracle_K32"]["align_pooled"],
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    # CSVs
    with (art / "selector_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "K", "candidate", "iou", "hit", "cos_pos_mean"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        cols = ["task", "K"] + [f"{c}_iou" for c in CANDIDATES]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for tb in task_breakdown:
            w.writerow(tb)

    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

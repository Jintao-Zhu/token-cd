#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-2B entity-set (compute, GPU, resumable).

Per Confirmation state:
  1. reuse cached projector h_i [256,4096] (Phase-2A h_npz, fallback recompute).
  2. build groups from stored labels_kmeans_K; group feature v_i.
  3. extract entities (rule-based noun phrases, 1 or 2) from instruction;
     per entity pick top-1 group via cos(v_i, e_entity); entity_set = union.
  4. candidates: single_source / single_goal / entity_set / oracle_set
     (best k-subset IoU) / random_set (random k groups). Masked branch via
     forward_masked on the union (single groups reuse masked_kmeans_K).
  5. residual r = log_softmax(clean) - log_softmax(branch); + IoU + hit + disruption.
  6. dump per-state npz; second pass aggregates alignment/coverage/ablations.

No rollout, no training, no λ, no external VLM. object mask = oracle only.

Run (smoke):
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/entity_set.py \
      --artifact artifacts/semantic_token_cd_phase2b_entity_set_v1 \
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
H_REUSE = Path("artifacts/semantic_token_cd_phase2a_relation_grouping_v1/h_npz")

PREPOSITIONS = {"into", "in", "on", "onto", "near", "to", "next", "from", "under", "over",
                "behind", "beside", "above", "below", "inside", "at", "with"}
DETERMINERS = {"the", "a", "an"}
K_VALUES = [8, 16, 32]
CANDIDATES = ("single_source", "single_goal", "entity_set", "oracle_set", "random_set")
HARD_TASKS = ["widowx_put_eggplant_in_basket", "widowx_spoon_on_towel"]
PRIMARY_K = 16


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extract_relation(instruction: str) -> dict[str, str]:
    """Decompose instruction -> {action, source, relation, target, ...} (Phase-2A verified)."""
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
    rel_phrase = " ".join(w for w in rest if w not in DETERMINERS)
    noun_phrase = " ".join(w for w in rest if w not in DETERMINERS and w not in PREPOSITIONS)
    return {"action": verb, "source": src, "relation": rel, "target": tgt,
            "rel_phrase": rel_phrase, "noun_phrase": noun_phrase}


def extract_entities(instruction: str) -> list[str]:
    """Rule-based entity list = deduped [source, target] (1 or 2 entities, no LLM)."""
    r = extract_relation(instruction)
    ents: list[str] = []
    for e in (r["source"], r["target"]):
        if e and e not in ents:
            ents.append(e)
    return ents


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


def iou_single(g: set, obj: set) -> float:
    inter = len(g & obj)
    union = len(g | obj)
    return inter / union if union else 0.0


def iou_of(groups: list[set], pair: tuple[int, int], obj: set) -> float:
    u = groups[pair[0]] | groups[pair[1]]
    inter = len(u & obj)
    union = len(u | obj)
    return inter / union if union else 0.0


def entity_top1(v: np.ndarray, e: np.ndarray) -> int:
    """argmax_i cos(v_i, e)."""
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    en = e / (np.linalg.norm(e) + 1e-8)
    return int(np.argmax(vn @ en))


def best_subset(groups: list[set], obj: set, k: int) -> list[int]:
    """Best k-subset by IoU with PCD mask (k in {1,2})."""
    K = len(groups)
    if k <= 1:
        best, bi = -1.0, 0
        for i in range(K):
            io = iou_single(groups[i], obj)
            if io > best:
                best, bi = io, i
        return [int(bi)]
    best, bi, bj = -1.0, 0, 1
    for i in range(K):
        for j in range(i + 1, K):
            io = iou_of(groups, (i, j), obj)
            if io > best:
                best, bi, bj = io, i, j
    return [int(bi), int(bj)]


def random_subset(rng: np.random.Generator, K: int, k: int) -> list[int]:
    return [int(x) for x in rng.choice(K, size=k, replace=False)]


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
    ent_dir = art / "entity_npz"
    h_dir = art / "h_npz"
    ent_dir.mkdir(parents=True, exist_ok=True)
    h_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT.read_text())
    (art / "FROZEN_SPLIT.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    if a.state_id:
        state_ids = [a.state_id]
    else:
        key = "selection_state_ids" if a.split == "selection" else "confirmation_state_ids"
        state_ids = split[key][: a.limit] if a.limit else split[key]

    states = {r["state_id"]: r for r in read_jsonl(STATES_LOCK)}

    model = processor = tokenizer = None
    if not a.eval_only:
        torch.manual_seed(20260824)
        torch.cuda.manual_seed_all(20260824)
        model, processor = load_model(CHECKPOINT)
        tokenizer = processor.tokenizer
        mean = load_position_mean(a.mean_path.resolve()).to(model.device, dtype=torch.bfloat16)
        token_slice = action_token_slice(model)

        instr_set = {}
        for sid in state_ids:
            instr_set.setdefault(states[sid]["instruction"], None)
        emb_cache = {}
        entity_json = {}
        for instr in instr_set:
            ents = extract_entities(instr)
            entity_json[instr] = {"entities": ents, "n": len(ents)}
            emb_cache[instr] = [embed_phrase(model, tokenizer, e) for e in ents]
        (art / "entity_extraction.json").write_text(
            json.dumps(entity_json, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        print(json.dumps({"model_loaded": True, "n_states": len(state_ids), "split": a.split,
                          "n_unique_instructions": len(instr_set)}), flush=True)

        n_fwd = 0
        t0 = time.time()
        for ordinal, sid in enumerate(state_ids):
            out_npz = ent_dir / f"{sid}.npz"
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

            # h_i (reuse Phase-2A cache, else compute)
            reuse = H_REUSE / f"{sid}.npz"
            if reuse.exists():
                h = np.load(reuse)["h"].astype(np.float64)
            else:
                h_npz = h_dir / f"{sid}.npz"
                if h_npz.exists():
                    h = np.load(h_npz)["h"].astype(np.float64)
                else:
                    h_i, _ = extract_features(model, processor, clean_img, instruction, clean_ids)
                    h = h_i.numpy().astype(np.float64)
                    np.savez_compressed(h_npz, h=h_i.numpy().astype(np.float16))
            n_fwd += 1

            ents = entity_json[instruction]["entities"]
            emb = emb_cache[instruction]
            out = {"state_id": sid, "task": str(sem["task"])}

            for K in K_VALUES:
                labels = sem[f"labels_kmeans_{K}"].astype(np.int64)
                masked = sem[f"masked_kmeans_{K}"]           # [K,7,256]
                v = np.zeros((K, h.shape[1]), dtype=np.float64)
                for k in range(K):
                    idx = np.flatnonzero(labels == k)
                    v[k] = h[idx].mean(axis=0) if len(idx) else 0.0
                groups = group_sets(labels, K)

                entity_groups: list[int] = []
                for e_ent in emb:
                    g = entity_top1(v, e_ent)
                    if g not in entity_groups:
                        entity_groups.append(g)
                source_group = entity_groups[0]
                goal_group = entity_groups[-1]
                k = len(entity_groups)

                oracle_set = best_subset(groups, object_ids, k)
                rng = np.random.default_rng(random_seed_for(sid) + 4000 + K)
                random_set = random_subset(rng, K, k)

                picks = {"single_source": [source_group], "single_goal": [goal_group],
                         "entity_set": entity_groups, "oracle_set": oracle_set,
                         "random_set": random_set}

                for cand in CANDIDATES:
                    sel = picks[cand]
                    union: set = set()
                    for g in sel:
                        union |= groups[g]
                    if len(sel) == 1:
                        branch = masked[sel[0]].astype(np.float64)
                    else:
                        ma, _ = forward_masked(model, processor, clean_img, instruction, clean_ids,
                                               sorted(union), mean)
                        branch = ma[:, token_slice].numpy().astype(np.float64)
                        n_fwd += 1
                    r = log_softmax_residual(clean_action, branch)
                    inter = len(union & object_ids)
                    un = len(union | object_ids)
                    iou = inter / un if un else 0.0
                    hit = 1 if set(sel) == set(oracle_set) else 0
                    dis = float(np.mean(np.linalg.norm(clean_action - branch, axis=1)))
                    out[f"{cand}_sel_{K}"] = np.asarray(sel, dtype=np.int64)
                    out[f"{cand}_r_{K}"] = r.astype(np.float32)
                    out[f"{cand}_iou_{K}"] = np.asarray([iou], dtype=np.float64)
                    out[f"{cand}_hit_{K}"] = np.asarray([hit], dtype=np.int64)
                    out[f"{cand}_dis_{K}"] = np.asarray([dis], dtype=np.float64)

            np.savez_compressed(out_npz, **out)
            dt = time.time() - t0
            print(json.dumps({"state_id": sid, "done": ordinal + 1, "of": len(state_ids),
                              "sec": round(dt, 1), "fwd": n_fwd, "rate": round(n_fwd / dt, 1)}), flush=True)

        print(json.dumps({"DONE_COMPUTE": len(state_ids), "total_fwd": n_fwd,
                          "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    # ---------- aggregate (reads all entity_npz) ----------
    rows = []
    for sid in state_ids:
        f = ent_dir / f"{sid}.npz"
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
                    "disruption": float(d[f"{cand}_dis_{K}"][0]),
                    "cos_pos_mean": float(np.mean(pc)),
                })

    pooled = {}
    for K in K_VALUES:
        for cand in CANDIDATES:
            vals = []
            for sid in state_ids:
                f = ent_dir / f"{sid}.npz"
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
                "disruption_mean": _mean([r["disruption"] for r in rr]),
            }

    # task breakdown
    task_bd = {}
    for r in rows:
        task_bd.setdefault((r["task"], r["K"], r["candidate"]), []).append(r["iou"])
    task_breakdown = []
    for t in sorted({r["task"] for r in rows}):
        for K in K_VALUES:
            row = {"task": t, "K": K}
            for cand in CANDIDATES:
                vv = task_bd.get((t, K, cand), [])
                row[f"{cand}_iou"] = _mean(vv)
            task_breakdown.append(row)

    # gates (primary K=16)
    Kp = PRIMARY_K
    es = agg[f"entity_set_K{Kp}"]["align_pooled"]
    ss = agg[f"single_source_K{Kp}"]["align_pooled"]
    sg = agg[f"single_goal_K{Kp}"]["align_pooled"]
    best_single = max(ss, sg)
    orc = agg[f"oracle_set_K{Kp}"]["align_pooled"]
    rnd = agg[f"random_set_K{Kp}"]["align_pooled"]

    gateA = es > 0.60
    gateB = es > 0.579                        # noun_only single 0.529 + 0.05
    htb = {t: next((b for b in task_breakdown if b["task"] == t and b["K"] == Kp), None) for t in HARD_TASKS}
    gateC = any(htb[t] is not None and htb[t]["entity_set_iou"] > htb[t]["single_source_iou"] for t in HARD_TASKS)
    gateD = es > rnd + 0.05

    stop_entity_eq_single = abs(es - best_single) <= 0.02
    stop_area = abs(es - rnd) <= 0.02
    stop_selector_fail = gateA and (not gateC)

    stop_hit = stop_entity_eq_single or stop_area or stop_selector_fail
    verdict = "PASS_TO_ROLLOUT" if (gateA and gateB and gateC and gateD and not stop_hit) else "STOP_ENTITY_SET_NO_GO"

    results = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE2B_ENTITY_SET_V1",
        "split": a.split, "n_states": len([s for s in state_ids if (ent_dir / f"{s}.npz").exists()]),
        "verdict": verdict, "stop_rule_hit": stop_hit,
        "aggregate": agg,
        "pooled_alignment": {f"{cand}_K{K}": pooled[(K, cand)] for K in K_VALUES for cand in CANDIDATES},
        "gates": {
            "A_alignment": {"PASS": bool(gateA), "entity_set_align": es, "oracle_align": orc},
            "B_over_single": {"PASS": bool(gateB), "entity_set_align": es, "best_single_align": best_single,
                              "threshold": 0.579},
            "C_hard_task": {"PASS": bool(gateC),
                            "eggplant": {"single_iou": htb[HARD_TASKS[0]]["single_source_iou"] if htb[HARD_TASKS[0]] else None,
                                         "entity_set_iou": htb[HARD_TASKS[0]]["entity_set_iou"] if htb[HARD_TASKS[0]] else None},
                            "spoon": {"single_iou": htb[HARD_TASKS[1]]["single_source_iou"] if htb[HARD_TASKS[1]] else None,
                                      "entity_set_iou": htb[HARD_TASKS[1]]["entity_set_iou"] if htb[HARD_TASKS[1]] else None}},
            "D_over_random": {"PASS": bool(gateD), "entity_set_align": es, "random_align": rnd},
        },
        "ablations": {
            "A_single_vs_set": {"single_source": ss, "single_goal": sg, "entity_set": es,
                                "set_gt_single": es > best_single},
            "B_entity_count": {"one_entity_states": None, "two_entity_states": None},  # filled below
            "C_source_vs_set": {"single_source": ss, "entity_set": es, "source_plus_goal_gt_source": es > ss},
        },
        "stop_flags": {
            "entity_eq_single": bool(stop_entity_eq_single),
            "area_redundancy": bool(stop_area),
            "selector_fail": bool(stop_selector_fail),
        },
        "task_breakdown": task_breakdown,
    }

    # Ablation B: entity count (1 vs 2) — split states by entity_extraction.json
    ent_json = json.loads((art / "entity_extraction.json").read_text())
    n1 = [sid for sid in state_ids if ent_json[states[sid]["instruction"]]["n"] == 1]
    n2 = [sid for sid in state_ids if ent_json[states[sid]["instruction"]]["n"] == 2]
    def _align_pooled(sids, K, cand):
        vv = []
        for sid in sids:
            f = ent_dir / f"{sid}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            sem = np.load(SEMANTIC_NPZ / f"{sid}.npz")
            r = d[f"{cand}_r_{K}"].astype(np.float64)
            vv.extend(per_position_cosine(r, sem["r_pixel"].astype(np.float64)).tolist())
        return _mean(vv)
    results["ablations"]["B_entity_count"] = {
        "one_entity": {"n": len(n1), "entity_set_align": _align_pooled(n1, Kp, "entity_set"),
                       "single_source_align": _align_pooled(n1, Kp, "single_source")},
        "two_entity": {"n": len(n2), "entity_set_align": _align_pooled(n2, Kp, "entity_set"),
                       "single_source_align": _align_pooled(n2, Kp, "single_source")},
    }

    (art / "selector_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "SEMANTIC_TOKEN_CD_PHASE2B_ENTITY_SET_V1",
        "status": verdict, "split": a.split, "date": "2026-08-24",
        "n_states": results["n_states"],
        "gates": {k: ("PASS" if v["PASS"] else "FAIL") for k, v in results["gates"].items()},
        "core_numbers": {
            "align_entity_set_K16": es, "align_single_source_K16": ss, "align_single_goal_K16": sg,
            "align_oracle_set_K16": orc, "align_random_set_K16": rnd,
            "align_entity_set_K8": agg["entity_set_K8"]["align_pooled"],
            "align_entity_set_K32": agg["entity_set_K32"]["align_pooled"],
            "iou_entity_set_K16": agg["entity_set_K16"]["iou_median"],
            "iou_single_source_K16": agg["single_source_K16"]["iou_median"],
        },
        "closed_loop": "NOT_RUN",
    }
    (art / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    # CSVs
    with (art / "selector_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state_id", "task", "K", "candidate", "iou", "hit", "disruption", "cos_pos_mean"])
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

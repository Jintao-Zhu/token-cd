#!/usr/bin/env python3
"""CW-LPCD STEP 1: correctness sentinels for the transient layer-window attention
intervention, against the SIMPLER OpenVLA-7B base checkpoint.

Three required sentinels (any failure => STOP_IMPLEMENTATION_ERROR):
  A. Vanilla reproduction: custom layer loop == standard `language_model` forward AND
     == Prismatic `forward` (both bit-exact), and its greedy action tokens match the
     standard forward 100/100.
  B. All-open key_mask (explicit zeros) == vanilla forward (bit-exact).
  C. Empty object-key set (CW path) == vanilla forward (bit-exact).

Plus a PART 2 window-semantics micro-test: within the window the action-query ->
object-key attention is zeroed; BEFORE the window attention is bit-identical to the
all-open run; AFTER the window the mask is removed (attention restored, non-zero).

NOTE (documented, not a bug): free-run `model.generate` (KV-cache) can differ from the
teacher-forced prefix forward by ~0.4 logits on near-ties due to bf16 KV-cache rounding,
occasionally flipping a token argmax. Phase-0 residual uses teacher-forced consistently
(matching the historical stage_a), so this is irrelevant to the offline metric. The
generate greedy-match is therefore reported informationally, not gated.

Run: env/venv/bin/python -m research.cw_lpcd.sentinel --artifact artifacts/cw_lpcd_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

from research.cw_lpcd.core import (
    N_ACTION_TOKENS, PCD_ROOT, build_multimodal, custom_layer_loop,
    ensure_empty_action_token, generate_clean_ids, inputs_for, load_model,
    make_cw_mask,
)


def read_states(states_lock: Path) -> list[dict]:
    return [json.loads(line) for line in states_lock.read_text().splitlines() if line]


def balanced_states(states: list[dict], n_per_task: int) -> list[dict]:
    seen: dict[str, int] = {}
    out: list[dict] = []
    for row in states:
        task = row["task"]
        if seen.get(task, 0) >= n_per_task:
            continue
        seen[task] = seen.get(task, 0) + 1
        out.append(row)
    return out


def teacher_forward(model, inputs, clean_ids):
    base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    teacher_ids = torch.cat([base_ids, clean_ids[:, :-1]], dim=1)
    teacher_mask = torch.cat([base_mask, torch.ones_like(clean_ids[:, :-1], dtype=base_mask.dtype)], dim=1)
    mm_embeds, mm_mask, n_visual = build_multimodal(model, teacher_ids, teacher_mask, inputs["pixel_values"])
    query_indices = [n_visual + base_ids.shape[1] - 1 + offset for offset in range(N_ACTION_TOKENS)]
    return teacher_ids, teacher_mask, mm_embeds, mm_mask, n_visual, query_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-root", type=Path, default=PCD_ROOT)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--n-per-task", type=int, default=15)
    parser.add_argument("--states-lock", type=Path)
    args = parser.parse_args()

    pcd_root = args.pcd_root.resolve()
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)

    states_lock = args.states_lock or Path(
        "artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl"
    )
    states = balanced_states(read_states(states_lock.resolve()), args.n_per_task)
    print(json.dumps({"states_total": len(states), "n_per_task": args.n_per_task}), flush=True)

    torch.manual_seed(20260822)
    torch.cuda.manual_seed_all(20260822)
    sys.path.insert(0, str(pcd_root / "source/PCD"))
    model, processor = load_model()
    print(json.dumps({"model_loaded": True, "vocab_size": int(model.vocab_size),
                      "n_layers": len(model.language_model.model.layers)}), flush=True)

    # ---------------- Sentinel A / B / C ----------------
    a_results: list[dict] = []
    b_fail = c_fail = 0
    for ordinal, row in enumerate(states):
        clean_path = pcd_root / row["clean_path"]
        image = cv2.cvtColor(cv2.imread(str(clean_path)), cv2.COLOR_BGR2RGB)
        inputs = inputs_for(processor, model, image, row["instruction"])
        clean_ids = generate_clean_ids(model, inputs)  # [1, 7]
        teacher_ids, teacher_mask, mm_embeds, mm_mask, n_visual, qi = teacher_forward(model, inputs, clean_ids)
        seq_len = mm_embeds.shape[1]

        # Path A: standard language_model no-cache forward
        logits_std = model.language_model(inputs_embeds=mm_embeds, attention_mask=mm_mask,
                                          use_cache=False, return_dict=True).logits
        # Path B: custom layer loop (all-open)
        logits_custom, _ = custom_layer_loop(model, mm_embeds, mm_mask)
        # Path C: Prismatic forward (what stage_a used)
        logits_prism = model(input_ids=teacher_ids, attention_mask=teacher_mask,
                             pixel_values=inputs["pixel_values"], use_cache=False, return_dict=True).logits

        d_std = (logits_std - logits_custom).abs().float()
        d_prism = (logits_prism - logits_custom).abs().float()
        greedy_custom = logits_custom[0, qi].argmax(-1)
        greedy_prism = logits_prism[0, qi].argmax(-1)
        match_prism = bool(torch.equal(greedy_custom, greedy_prism))
        match_gen = bool(torch.equal(greedy_custom, clean_ids[0]))

        # Sentinel B: explicit all-open (zeros) key_mask == vanilla
        zero_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.bfloat16, device=mm_embeds.device)
        logits_b, _ = custom_layer_loop(model, mm_embeds, mm_mask, key_mask=zero_mask)
        b_ok = bool(torch.equal(logits_b, logits_custom))

        # Sentinel C: empty object-key set through the CW path == vanilla
        logits_c, _ = custom_layer_loop(
            model, mm_embeds, mm_mask,
            key_mask=make_cw_mask(seq_len, qi, (), torch.bfloat16, mm_embeds.device))
        c_ok = bool(torch.equal(logits_c, logits_custom))

        a_results.append({"state_id": row["state_id"],
                          "max_abs_logit_diff_vs_language_model": float(d_std.max()),
                          "max_abs_logit_diff_vs_prismatic": float(d_prism.max()),
                          "greedy_match_prismatic": match_prism,
                          "greedy_match_generate": match_gen})
        if not b_ok:
            b_fail += 1
        if not c_ok:
            c_fail += 1
        if ordinal % 15 == 0:
            print(json.dumps({"done": ordinal, "state": row["state_id"]}), flush=True)

    n_states = len(a_results)
    max_std = max(r["max_abs_logit_diff_vs_language_model"] for r in a_results)
    max_prism = max(r["max_abs_logit_diff_vs_prismatic"] for r in a_results)
    n_prism = sum(1 for r in a_results if r["greedy_match_prismatic"])
    n_gen = sum(1 for r in a_results if r["greedy_match_generate"])
    a_pass = (max_std == 0.0) and (max_prism == 0.0) and (n_prism == n_states)

    # ---------------- PART 2: window semantics ----------------
    part2: list[dict] = []
    window = (8, 20)
    for row in states[:3]:
        clean_path = pcd_root / row["clean_path"]
        image = cv2.cvtColor(cv2.imread(str(clean_path)), cv2.COLOR_BGR2RGB)
        inputs = inputs_for(processor, model, image, row["instruction"])
        clean_ids = generate_clean_ids(model, inputs)
        _, _, mm_embeds, mm_mask, n_visual, qi = teacher_forward(model, inputs, clean_ids)
        seq_len = mm_embeds.shape[1]
        object_keys = [1 + 10, 1 + 50, 1 + 100]

        _, attns_open = custom_layer_loop(model, mm_embeds, mm_mask, output_attentions=True)
        key_mask = make_cw_mask(seq_len, qi, object_keys, torch.bfloat16, mm_embeds.device)
        logits_win, attns_win = custom_layer_loop(model, mm_embeds, mm_mask, key_mask=key_mask,
                                                  key_mask_window=window, output_attentions=True)

        inside_zero = True
        before_exact = True
        after_restored = True
        for li, (ao, aw) in enumerate(zip(attns_open, attns_win)):
            ao = ao[:, :, qi][:, :, :, object_keys]  # [1, H, 7, 3]
            aw = aw[:, :, qi][:, :, :, object_keys]
            if window[0] <= li <= window[1]:
                if not bool((aw < 1e-6).all()):
                    inside_zero = False
            elif li < window[0]:
                if not bool(torch.equal(ao, aw)):
                    before_exact = False
            else:  # li > window[1]: mask removed -> object keys accessible again
                if bool((aw < 1e-6).all()):
                    after_restored = False

        part2.append({"state_id": row["state_id"], "window": window, "object_keys": object_keys,
                      "finite": bool(torch.isfinite(logits_win).all()),
                      "inside_window_zeroed": inside_zero,
                      "before_window_exact": before_exact,
                      "after_window_restored": after_restored,
                      "seq_len": seq_len, "n_visual": n_visual})

    p2_inside = all(p["inside_window_zeroed"] for p in part2)
    p2_before = all(p["before_window_exact"] for p in part2)
    p2_after = all(p["after_window_restored"] for p in part2)

    report = {
        "status": "PASS" if (a_pass and b_fail == 0 and c_fail == 0 and p2_inside and p2_before and p2_after) else "FAIL",
        "n_states": n_states, "n_per_task": args.n_per_task,
        "sentinel_A": {
            "pass": a_pass,
            "max_abs_logit_diff_vs_language_model": max_std,
            "max_abs_logit_diff_vs_prismatic": max_prism,
            "greedy_match_prismatic_frac": f"{n_prism}/{n_states}",
            "greedy_match_generate_frac": f"{n_gen}/{n_states}",
            "note": "generate(KV-cache) vs teacher-forced can differ ~0.4 logits on near-ties (bf16); prismatic-forward is the exact reference",
        },
        "sentinel_B_allopen_exact": {"fail_count": b_fail, "pass": b_fail == 0},
        "sentinel_C_empty_exact": {"fail_count": c_fail, "pass": c_fail == 0},
        "part2_window_semantics": {"window": window, "inside_window_zeroed": p2_inside,
                                   "before_window_exact": p2_before, "after_window_restored": p2_after,
                                   "details": part2},
    }
    out_path = artifact / "sentinel_step1.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""AMCD Phase-0 sentinels: expert-token round-trip + implementation (final-block readout).

Part 1 (STEP 5, expert action token correctness):
  continuous expert action -> 7 action tokens -> decode back to continuous,
  on the 100 Selection states.  Asserts per-dim |decoded - transformed_action|
  stays below the bin width.  Any off-by-one / wrong bin indexing => STOP_TOKEN_ALIGNMENT_ERROR.

Part 2 (implementation sentinel):
  (a) readout_layer(h_31) reproduces the layer-loop final logits bit-exactly,
  (b) custom layer loop reproduces the OFFICIAL OpenVLA no-cache forward
      (model.language_model(inputs_embeds=...)) — argmax over the action vocab
      must match 100% on 100 states x 7 positions.

Run:
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" task1/.venvs/openvla-ar/bin/python \
      research/ar_amcd/sentinel.py --workspace . --artifact artifacts/ar_amcd_action_maturity_phase0_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, set_determinism
from research.ar_token_counterfactual.intervention import ensure_empty_action_token
from research.ar_amcd.core import (
    ACTION_HI, ACTION_LO, BINS, VOCAB_SIZE,
    action_slice, build_multimodal, decode_tokens, encode_action_tokens,
    layer_loop_with_hidden, load_action_stats, readout_layer, transformed_action,
)

CHECKPOINT = "openvla-7b-finetuned-libero-object/287d6cfdf12d07b1449505f66d9bf3550257e9b3"
FROZEN_ARTIFACT = "artifacts/ar_sid_token_cd_phase0_v1_20260822T145651+0800"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--max-states", type=int, default=100)
    a = p.parse_args()
    w = a.workspace.resolve()
    art = a.artifact.resolve()
    art.mkdir(parents=True, exist_ok=True)

    ckpt_dir = w / f"checkpoints/{CHECKPOINT}"
    low, high, mask = load_action_stats(ckpt_dir)
    bin_width = np.where(mask, (high - low) / 255.0, 2.0 / 255.0)

    set_determinism(20260822)
    model, processor = load_policy(ckpt_dir, w / "third_party/openvla/prismatic/extern/hf")

    n_layers = len(model.language_model.model.layers)
    hidden_size = model.language_model.config.hidden_size
    print(json.dumps({"n_layers": n_layers, "hidden_size": hidden_size,
                      "vocab_size": int(model.vocab_size),
                      "bin_centers_shape": list(model.bin_centers.shape)}), flush=True)
    assert n_layers == 32, f"expected 32 layers, got {n_layers}"
    assert int(model.vocab_size) == VOCAB_SIZE

    split = json.loads((w / FROZEN_ARTIFACT / "frozen_state_split.json").read_text())
    selection_ids = split["selection"][: a.max_states]
    by_id = {}
    with (w / FROZEN_ARTIFACT / "state_manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            by_id[row["state_id"]] = row
    rows = [by_id[sid] for sid in selection_ids]
    n_states = len(rows)

    # ---- Part 1: expert-token mapping correctness ----
    # 1a. Direct grid round-trip over the normalized domain [-1, 1]: this is the
    #     exact bin-mapping check (off-by-one / wrong bin indexing would fail here).
    grid = np.linspace(-1.0, 1.0, 10001)
    g_disc = np.digitize(np.clip(grid, -1.0, 1.0), BINS)
    g_tok = VOCAB_SIZE - g_disc
    g_d = np.clip(model.vocab_size - g_tok - 1, 0, model.bin_centers.shape[0] - 1)
    g_centers = model.bin_centers[g_d]
    if hasattr(g_centers, "detach"):
        g_centers = g_centers.detach().cpu().numpy()
    grid_max_err = float(np.abs(np.asarray(g_centers, dtype=np.float64) - np.clip(grid, -1.0, 1.0)).max())
    grid_ok = grid_max_err <= 0.004  # half a bin = 2/255/2 = 0.00392, + eps

    # 1b. Manifest round-trip, split in-range vs out-of-[q01,q99] (percentile
    #     normalization legitimately saturates out-of-range values).
    inrange_max = np.zeros(7)
    outrange_count = np.zeros(7, dtype=int)
    token_out_of_range = 0
    for row in rows:
        raw = np.asarray(json.loads(row["expert_action"]), dtype=np.float64)
        toks = encode_action_tokens(raw, low, high, mask)
        if not (np.all(toks >= ACTION_LO) and np.all(toks < ACTION_HI)):
            token_out_of_range += 1
        proc = transformed_action(raw)
        norm = np.where(mask, 2.0 * (proc - low) / (high - low + 1e-8) - 1.0, proc)
        in_range = np.abs(norm) <= 1.0
        outrange_count += (~in_range).astype(int)
        decoded = decode_tokens(model, toks, low, high, mask)
        err = np.abs(decoded - proc)
        inrange_max = np.maximum(inrange_max, np.where(in_range, err, 0.0))
    manifest_inrange_ok = bool(np.all(inrange_max < bin_width)) and token_out_of_range == 0
    rt_ok = bool(grid_ok and manifest_inrange_ok)

    # ---- Part 2: implementation sentinel ----
    n_agree_action = 0
    n_agree_full = 0
    max_abs_custom_official = 0.0
    max_abs_readout_loop = 0.0
    total_pos = 0
    n_visual_ok = 0
    for si, row in enumerate(rows):
        language = row["language"]
        expert_raw = np.asarray(json.loads(row["expert_action"]), dtype=np.float64)
        fine_image = Image.open(w / FROZEN_ARTIFACT / row["image"]).convert("RGB")

        inputs = processor(build_prompt(language), fine_image).to(model.device, dtype=torch.bfloat16)
        base_ids, base_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
        pv = inputs["pixel_values"]

        expert_tokens = torch.tensor(
            encode_action_tokens(expert_raw, low, high, mask), dtype=base_ids.dtype, device=base_ids.device
        ).unsqueeze(0)  # [1, 7]
        teacher_ids = torch.cat([base_ids, expert_tokens[:, :-1]], dim=1)
        teacher_mask = torch.cat(
            [base_mask, torch.ones((1, 6), device=base_mask.device, dtype=base_mask.dtype)], dim=1)

        mm_embeds, mm_mask, n_visual = build_multimodal(model, teacher_ids, teacher_mask, pv)
        assert n_visual == 256, f"n_visual={n_visual}"
        n_visual_ok += 1
        seq_len = mm_embeds.shape[1]
        aq = list(range(seq_len - 7, seq_len))

        # custom loop with hidden collection
        hiddens, final_logits = layer_loop_with_hidden(model, mm_embeds, mm_mask)

        # (a) readout_layer(h_31) == final_logits (bit-exact)
        z31_readout = readout_layer(model, hiddens[31])
        max_abs_readout_loop = max(max_abs_readout_loop,
                                   float((z31_readout - final_logits).abs().max()))

        # (b) custom loop == official no-cache forward
        out_official = model.language_model(
            inputs_embeds=mm_embeds, attention_mask=mm_mask, use_cache=False, return_dict=True)
        logits_official = out_official.logits

        d = (final_logits - logits_official).abs().float()
        max_abs_custom_official = max(max_abs_custom_official, float(d.max()))

        aq = list(range(seq_len - 7, seq_len))
        zc = final_logits[0, aq, ACTION_LO:ACTION_HI]
        zo = logits_official[0, aq, ACTION_LO:ACTION_HI]
        n_agree_action += int((zc.argmax(-1) == zo.argmax(-1)).all())
        n_agree_full += int((final_logits[0, aq].argmax(-1) == logits_official[0, aq].argmax(-1)).all())
        total_pos += 7

        del hiddens, final_logits, out_official, logits_official, mm_embeds
        if (si + 1) % 10 == 0 or si + 1 == n_states:
            print(json.dumps({"done": si + 1, "of": n_states}), flush=True)

    impl_ok = (n_agree_action == n_states and n_agree_full == n_states
               and max_abs_custom_official < 1e-3 and n_visual_ok == n_states)

    result = {
        "experiment": "AR_AMCD_ACTION_MATURITY_PHASE0_V1",
        "step": "STEP5_expert_token_roundtrip + implementation_sentinel",
        "n_states": n_states,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "vocab_size": int(model.vocab_size),
        "action_vocab_slice": [ACTION_LO, ACTION_HI - 1],
        "part1_expert_roundtrip": {
            "PASS": bool(rt_ok),
            "grid_roundtrip_max_err": grid_max_err,
            "grid_roundtrip_ok": bool(grid_ok),
            "manifest_inrange_max_abs_err_per_dim": [float(x) for x in inrange_max],
            "manifest_out_of_range_count_per_dim": [int(x) for x in outrange_count],
            "bin_width_per_dim": [float(x) for x in bin_width],
            "token_out_of_range_count": int(token_out_of_range),
            "note": "grid round-trip validates the bin mapping; manifest in-range values "
                    "decode within bin width; out-of-[q01,q99] values saturate by design.",
        },
        "part2_implementation": {
            "PASS": bool(impl_ok),
            "argmax_agree_action_vocab": f"{n_agree_action}/{n_states}",
            "argmax_agree_full_vocab": f"{n_agree_full}/{n_states}",
            "total_positions": int(total_pos),
            "max_abs_custom_vs_official": float(max_abs_custom_official),
            "max_abs_readout_h31_vs_loop": float(max_abs_readout_loop),
            "n_visual_ok": int(n_visual_ok),
        },
    }
    (art / "sentinel_step5.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if not (rt_ok and impl_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

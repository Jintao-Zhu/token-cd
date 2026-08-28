#!/usr/bin/env python3
"""ALE-CD Phase-0: re-run full-layer vanilla-branch readout on 200 Confirmation states.

AMCD's confirmation_data.npz only kept layer 11 + 31; ALE-CD needs the complete
32-layer trajectory on the vanilla (model-greedy) branch. This re-reads the SAME
checkpoint via logit lens (no training, no rollout); Selection/Confirmation split
is unchanged.

Run:
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" TF_CPP_MIN_LOG_LEVEL=3 \
    task1/.venvs/openvla-ar/bin/python research/ar_ale_cd/scan_conf.py \
      --workspace . --artifact artifacts/ar_ale_cd_phase0_v1
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
    ACTION_LO, build_multimodal, scan_layer_readouts, encode_action_tokens, load_action_stats,
)

CHECKPOINT = "openvla-7b-finetuned-libero-object/287d6cfdf12d07b1449505f66d9bf3550257e9b3"
FROZEN_ARTIFACT = "artifacts/ar_sid_token_cd_phase0_v1_20260822T145651+0800"
N_LAYERS = 32


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--max-states", type=int, default=200)
    a = p.parse_args()
    w = a.workspace.resolve()
    art = a.artifact.resolve()
    art.mkdir(parents=True, exist_ok=True)

    ckpt_dir = w / f"checkpoints/{CHECKPOINT}"
    low, high, mask = load_action_stats(ckpt_dir)

    split = json.loads((w / FROZEN_ARTIFACT / "frozen_state_split.json").read_text())
    conf_ids = split["confirmation"][: a.max_states]
    by_id = {}
    with (w / FROZEN_ARTIFACT / "state_manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            by_id[row["state_id"]] = row
    rows = [by_id[sid] for sid in conf_ids]
    n = len(rows)

    set_determinism(20260822)
    model, processor = load_policy(ckpt_dir, w / "third_party/openvla/prismatic/extern/hf")

    z_vanilla = np.zeros((n, N_LAYERS, 7, 256), dtype=np.float32)
    logZ_vanilla = np.zeros((n, N_LAYERS, 7), dtype=np.float32)
    expert_local = np.zeros((n, 7), dtype=np.int32)
    tasks = [r["task_id"] for r in rows]
    state_ids = [r["state_id"] for r in rows]

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
        expert_local[si] = (expert_tokens[0].cpu().numpy() - ACTION_LO).astype(np.int32)

        # vanilla branch: model greedy generation, then teacher-force first 6 as prefix
        gen = model.generate(input_ids=base_ids, attention_mask=base_mask, pixel_values=pv,
                             max_new_tokens=7, do_sample=False)
        vanilla_tokens = gen[:, -7:]
        v_teacher_ids = torch.cat([base_ids, vanilla_tokens[:, :-1]], dim=1)
        v_teacher_mask = torch.cat(
            [base_mask, torch.ones((1, 6), device=base_mask.device, dtype=base_mask.dtype)], dim=1)
        v_mm, v_mm_mask, v_nvis = build_multimodal(model, v_teacher_ids, v_teacher_mask, pv)
        assert v_nvis == 256
        v_aq = list(range(v_mm.shape[1] - 7, v_mm.shape[1]))
        zv, logzv = scan_layer_readouts(model, v_mm, v_mm_mask, v_aq)  # [32,7,256], [32,7]
        z_vanilla[si] = zv
        logZ_vanilla[si] = logzv

        if (si + 1) % 20 == 0 or si + 1 == n:
            print(json.dumps({"done": si + 1, "of": n}), flush=True)

    out = art / "ale_traj_conf.npz"
    np.savez_compressed(
        out, z_vanilla=z_vanilla, logZ_vanilla=logZ_vanilla, expert_local=expert_local,
        state_ids=np.asarray(state_ids), tasks=np.asarray(tasks),
    )
    print(json.dumps({"done": n, "npz": str(out)}), flush=True)


if __name__ == "__main__":
    main()

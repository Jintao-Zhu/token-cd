#!/usr/bin/env python3
"""Freeze a same-state LIBERO checkpoint-transfer mechanism cohort."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/libero_action_value_gate_egl_unseen_task_v1_20260925/cases_manifest.json"
PREFLIGHT_SOURCE = ROOT / "artifacts/libero_action_value_gate_egl_unseen_task_preflight_task18_init49_20260925/PREFLIGHT_CASE.json"
OUT = ROOT / "artifacts/libero_checkpoint_transfer_base_openvla_n50_v1_20260925"
PROTOCOL = "LIBERO90_CHECKPOINT_TRANSFER_BASE_OPENVLA_VS_LIBERO90_V1"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    cases = []
    for row in source["cases"]:
        case = dict(row)
        case["protocol_id"] = PROTOCOL
        case["arms"] = ["vanilla", "matched"]
        case["renderer_backend"] = "egl"
        cases.append(case)
    if len(cases) != 50 or len({c["case_id"] for c in cases}) != 50:
        raise RuntimeError("source cohort is not 50 unique cases")

    preflight = json.loads(PREFLIGHT_SOURCE.read_text())
    preflight["protocol_id"] = PROTOCOL + "_PREFLIGHT"
    preflight["arms"] = ["vanilla", "matched"]
    preflight["renderer_backend"] = "egl"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "pairs").mkdir(exist_ok=True)
    (OUT / "episodes").mkdir(exist_ok=True)
    (OUT / "cases_manifest.json").write_text(json.dumps({
        "protocol_id": PROTOCOL,
        "pair_count": len(cases),
        "sampling": "exact same 50 tasks/init IDs/seeds as the audited LIBERO90 positive-G N=50 reference cohort; no case selected by outcome",
        "arms": ["vanilla", "matched"],
        "checkpoint_under_test": "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source/pretrained/openvla-7b",
        "reference_checkpoint": "/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90",
        "fixed_action_statistics": "/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90/dataset_statistics.json",
        "unnorm_key": "libero_90_no_noops",
        "attention_layers": [11], "entity_mode": "source_target_libero90",
        "query_mode": "instruction_only", "negative": "canonical_harmonic_beta0", "lambda": 0.5,
        "renderer_backend": "egl", "render_gpu": 7, "inference_gpus": [1, 2, 3],
        "reference_manifest": str(SOURCE),
        "cases": cases,
    }, indent=2) + "\n")
    (OUT / "cases" / "all.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (OUT / "PREFLIGHT_CASE.json").write_text(json.dumps(preflight, indent=2) + "\n")
    (OUT / "cases" / "preflight.jsonl").write_text(json.dumps(preflight) + "\n")
    prereg = f"""# Frozen checkpoint-transfer mechanism test

Protocol: `{PROTOCOL}`

## Question and hypothesis

The audited SIMPLER L11-Matched evaluation uses pretrained base `openvla-7b`; the current LIBERO action-value and N=50 cohorts use a LIBERO90-finetuned OpenVLA checkpoint. The same-defined episode-level guidance score predicts Rescue in opposite directions across SIMPLER and LIBERO. This cohort changes only the model checkpoint within the same LIBERO task/state/rendering condition.

**Hypothesis:** checkpoint fine-tuning changes the action geometry through which the frozen L11-Matched counterfactual acts. Therefore, running the SIMPLER base checkpoint on the same LIBERO states will shift the Rescue-versus-Harm relationship of episode-median G toward the SIMPLER direction (lower G in Rescue than Harm), while the existing fine-tuned checkpoint provides the reference.

**Prediction:** compare base-checkpoint and existing LIBERO90-finetuned checkpoint on identical 50 task/init/seed pairs. If the base checkpoint moves the G/Rescue association toward SIMPLER (Rescue-higher AUC below .5), while successes remain sufficiently discordant, checkpoint is a causal contributor. If the relation remains Rescue-higher or the cohort has too few discordant episodes, checkpoint alone does not explain the contrast.

## Frozen design

- 50 paired cases: exactly the same five LIBERO tasks, init IDs 0–9, environment seeds, and case seeds as the audited reference N=50. No case is selected using outcomes.
- New arms: `vanilla`, canonical `matched` (L11 Prompt Attention, historical Matched budget, harmonic negative, λ=.5).
- Existing same-case reference arms: audited `vanilla` and `matched` traces from the LIBERO90-finetuned checkpoint.
- **Changed variable:** model weights only.
- **Held fixed:** task/seed/state, task prompt, code path, camera, EGL renderer (physical GPU7), action-token set, action unnormalization key and statistics file, L11/matched/negative/lambda, GPU inference allocation (only GPUs1–3), and success detector.
- The base checkpoint is the exact `PCD_SOURCE/pretrained/openvla-7b` used by the SIMPLER runner. To preserve the LIBERO action decoding convention, both checkpoints use the same reference `dataset_statistics.json` and `libero_90_no_noops` unnormalization key.

## Outcomes and interpretation

Primary mechanism endpoint: episode-level Rescue-higher AUC for the frozen G metric, with per-dimension summaries and discordant counts. Also report Vanilla/Matched success and paired transitions under each checkpoint. The AUC is descriptive if either discordant class is small; no threshold will be tuned.

The new rollout must match the reference cohort's simulator-state and first-RGB hashes in every case. RGB/state mismatch, CUDA/EGL error, or Xid is a technical stop, not a method outcome. Preflight task18/init49 must pass before formal work. Use EGL by resolved physical GPU PCI BDF, inference on GPUs1–3, render GPU7; do not use GPUs4/5.

This isolates checkpoint as a contributor within LIBERO. A positive result would not by itself establish all of SIMPLER/LIBERO transfer because benchmark task/state distributions and simulator domains still differ. A negative or underpowered result rejects only the prediction that checkpoint alone flips this metric under these LIBERO conditions.
"""
    (OUT / "PREREGISTRATION.md").write_text(prereg)
    print(json.dumps({"artifact": str(OUT), "n": len(cases), "preflight": preflight["case_id"], "protocol": PROTOCOL}, indent=2))


if __name__ == "__main__":
    main()

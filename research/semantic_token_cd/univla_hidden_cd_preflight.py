"""One saved-image functional preflight; no simulator or episode is run."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
os.environ.setdefault("UNIVLA_L14_SOCK", "/tmp/univla_hidden_preflight_unused.sock")
sys.path.insert(0, str(REPO))

from research.semantic_token_cd.univla_hidden_cd_model_service import HiddenPolicy


def main():
    torch.set_grad_enabled(False)
    state = REPO / "artifacts/univla_prompt_attention_layer_scan_v1/states/widowx_carrot_on_plate__seed000.npz"
    if not state.exists():
        candidates = sorted((REPO / "artifacts/univla_prompt_attention_layer_scan_v1/states").glob("widowx_*__seed*.npz"))
        if not candidates:
            raise FileNotFoundError("no saved layer-scan RGB state for preflight")
        state = candidates[0]
    task = state.name.split("__seed")[0]
    run_root = REPO / "artifacts/univla_simpler_l14_matched_rtfull_100_v1/episodes" / task
    sample = next(run_root.glob("l14_matched/seed_*.json"), None)
    if sample is None:
        raise FileNotFoundError(f"no historical instruction for {task}")
    instruction = json.loads(sample.read_text())["instruction"]
    image = np.load(state)["image"]
    policy = HiddenPolicy()
    results = {}
    for arm, seed in (("hidden_cd_025", 20261002), ("hidden_cd_05", 20261002)):
        policy.reset(instruction, seed)
        traces = []
        # Repeating the same saved image once only to put a non-empty history
        # in ActionDecoder's temporal buffer; this is not a simulator rollout.
        for _ in range(2):
            meta = policy.step(image, arm, instruction)["meta"]
            assert meta["action_token_ids"] and len(meta["action_token_ids"]) == 4
            assert len(meta["selected_token_ids"]) == meta["m_t"] > 0
            for key in ("clean_preagg_raw", "guided_preagg_raw", "clean_postagg_raw", "guided_postagg_raw"):
                assert np.isfinite(np.asarray(meta[key])).all()
            assert len(meta["preagg_delta_xyz"]) == len(meta["preagg_delta_rot"]) == 3
            assert len(meta["postagg_delta_xyz"]) == len(meta["postagg_delta_rot"]) == 3
            traces.append({k: meta[k] for k in (
                "lambda_h", "action_token_ids", "m_t", "action_hidden_rel_l2",
                "action_hidden_norm_ratio", "preagg_delta_l2", "postagg_delta_l2",
                "preagg_delta_xyz", "preagg_delta_rot", "preagg_delta_grip",
                "postagg_delta_xyz", "postagg_delta_rot", "postagg_delta_grip",
                "grip_threshold_flip")})
        results[arm] = {"first_step": traces[0], "nonempty_buffer_step": traces[1]}
    if (results["hidden_cd_025"]["first_step"]["action_token_ids"] !=
            results["hidden_cd_05"]["first_step"]["action_token_ids"] or
            results["hidden_cd_025"]["nonempty_buffer_step"]["action_token_ids"] !=
            results["hidden_cd_05"]["nonempty_buffer_step"]["action_token_ids"]):
        raise AssertionError("clean action-token prefix changed across Hidden-CD lambdas")
    print(json.dumps({"status": "PASS", "state": state.name, "task": task,
                      "instruction": instruction, "arms": results}, indent=2))


if __name__ == "__main__":
    main()

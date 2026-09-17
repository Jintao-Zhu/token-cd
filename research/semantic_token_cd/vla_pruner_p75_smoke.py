"""Four-step full-model structural gate for code-L15 and paper-L16:31 P75."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
OUT = ROOT / "artifacts/vla_pruner_openvla_reproduction/paper_v5_gates"
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def run(policy, image, instruction):
    policy.reset(instruction, seed=0)
    rows = []
    for _ in range(4):
        policy.step(image, None, instruction)
        trace = policy._episode_trace[-1]
        rows.append({
            "tokens": trace["token_ids"],
            "history_before": trace["history_len"],
            "history_after": trace.get("history_after"),
            "effective_r": trace.get("fastv_r_effective"),
            "kept_image_count": trace.get("kept_image_count"),
            "pruned_count": trace.get("pruned_count"),
            "pruning_layer": trace.get("pruning_layer"),
            "history_layers": trace.get("history_layers"),
            "kept_visual_token_ids": trace.get("kept_visual_token_ids"),
        })
    return rows


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.vla_pruner_policy import build_vla_pruner_policy

    task = "google_robot_open_drawer"
    instruction = "open top drawer"
    state = np.load(ROOT / "artifacts/prompt_attn_layer_selection_v1/states/"
                    "google_robot_open_drawer/seed_000/step_011.npz")
    image = state["image"]
    checkpoint = str(SOURCE / "pretrained/openvla-7b")
    base = OpenVLAInference(**get_policy_config("openvla", checkpoint, task, {}, False))
    arms = ("vanilla", "vla_pruner_prune75_code_l15", "vla_pruner_prune75_paper_l16_31")
    rows = {arm: run(build_vla_pruner_policy(base, arm, task), image, instruction) for arm in arms}

    vanilla = rows["vanilla"]
    checks = {}
    for arm in arms[1:]:
        current = rows[arm]
        checks[arm] = {
            "warmup_tokens_equal_vanilla": all(current[i]["tokens"] == vanilla[i]["tokens"] for i in range(3)),
            "warmup_unpruned": all(current[i]["kept_image_count"] == 256 and current[i]["pruned_count"] == 0 for i in range(3)),
            "active_step_keep64": current[3]["kept_image_count"] == 64 and current[3]["pruned_count"] == 192,
            "active_step_layer3": current[3]["pruning_layer"] == 3,
            "history_full_at_active_step": current[3]["history_before"] == 3,
        }
    payload = {"task": task, "instruction": instruction, "rows": rows, "checks": checks}
    payload["all_passed"] = all(all(v.values()) for v in checks.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P75_FULL_MODEL_SMOKE.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"checks": checks, "all_passed": payload["all_passed"]}, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

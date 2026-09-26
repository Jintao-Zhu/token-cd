#!/usr/bin/env python3
"""Build the frozen 50-pair unseen-task EGL positive-G gate cohort."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
OUT = ROOT / "artifacts/libero_action_value_gate_egl_unseen_task_v1_20260925"
TASKS = {
    18: {
        "name": "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "source": "chefmate_8_frypan_1",
        "target": "flat_stove_1",
    },
    24: {
        "name": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
        "source": "akita_black_bowl_1",
        "target": "white_cabinet_1",
    },
    36: {
        "name": "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
        "source": "white_bowl_1",
        "target": "plate_1",
    },
    50: {
        "name": "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "source": "alphabet_soup_1",
        "target": "basket_1",
    },
    74: {
        "name": "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "source": "black_book_1",
        "target": "desk_caddy_1",
    },
}
GPUS = (1, 2, 3)


def main() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    cases = []
    for task_id, task in TASKS.items():
        for init_id in range(10):
            cases.append({
                "protocol_id": "LIBERO90_ACTION_VALUE_GPOSITIVE_GATE_EGL_UNSEEN_TASK_V1",
                "case_id": f"task{task_id:02d}__init{init_id:03d}",
                "task_id": task_id,
                "task_name": task["name"],
                "init_state_id": init_id,
                "case_seed": 20261001 + task_id * 100000 + init_id,
                "env_seed": 0,
                "settle_steps": 10,
                "max_policy_steps": 400,
                "arms": ["vanilla", "matched", "positive_gated"],
                "renderer_backend": "egl",
            })
    manifest = {
        "protocol_id": "LIBERO90_ACTION_VALUE_GPOSITIVE_GATE_EGL_UNSEEN_TASK_V1",
        "pair_count": len(cases),
        "sampling": "five previously unused LIBERO-90 tasks x init_state_id 0..9",
        "tasks": TASKS,
        "case_seed_rule": "20261001 + task_id*100000 + init_state_id",
        "arms": ["vanilla", "matched", "positive_gated"],
        "lambda": 0.5,
        "gate": "median current-step G over first six action dimensions > 0",
        "attention_layers": [11],
        "entity_mode": "source_target_libero90",
        "query_mode": "instruction_only",
        "negative": "canonical_harmonic_beta0",
        "renderer_backend": "egl",
        "render_gpu": 7,
        "inference_gpus": list(GPUS),
        "preflight_case_excluded": "task18__init049",
        "cases": cases,
    }
    (OUT / "cases_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for gpu in GPUS:
        assigned = [case for i, case in enumerate(cases) if i % len(GPUS) == GPUS.index(gpu)]
        (OUT / "cases" / f"gpu{gpu}.jsonl").write_text(
            "".join(json.dumps(case, sort_keys=True) + "\n" for case in assigned)
        )
    print(json.dumps({
        "manifest": str(OUT / "cases_manifest.json"),
        "case_count": len(cases),
        "per_gpu": {
            str(gpu): len((OUT / "cases" / f"gpu{gpu}.jsonl").read_text().splitlines())
            for gpu in GPUS
        },
    }, indent=2))


if __name__ == "__main__":
    main()

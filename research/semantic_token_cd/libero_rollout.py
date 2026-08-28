#!/usr/bin/env python3
"""LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1 — closed-loop rollout (one task).

Four arms — vanilla / entity_cd λ0.25 / entity_cd λ0.50 / random-mask control —
on one LIBERO-Object task with exact snapshot pairing (all arms start from the
identical initial sim state per episode_id) so Rescue/Harm and the semantic-vs-
random gate (D) can be computed per-pair. Shares one frozen OpenVLAForActionPrediction
(model, processor) across arms.

The CD logic mirrors the SIMPLER pilot (research/semantic_token_cd/rollout_policy.py):
  clean_scores   = AR-generate logits (output_scores), projector hook captures h_i
  selected       = KMeans(K=8) + per-entity top-1 cos -> entity_set (deduped union)
  negative_scores = AR-generate logits with selected visual tokens replaced by mean
  final[:-1]     = (1+λ)·clean[:-1] - λ·negative[:-1]; argmax; decode (libero_object)

Run (per task):
  cd /data/docker/dev_zjt/data/code
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3 \
  PYTHONPATH="./LIBERO:$PWD" OMP_NUM_THREADS=1 \
    task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/libero_rollout.py \
      --artifact artifacts/libero_object_semantic_entity_cd_phase0_v1 \
      --task pick_up_the_alphabet_soup_and_place_it_in_the_basket \
      --episodes 0,1 --gpu 0 --smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

REPO = Path("/data/docker/dev_zjt/data/code")
CHECKPOINT = REPO / "checkpoints/openvla-7b-finetuned-libero-object/287d6cfdf12d07b1449505f66d9bf3550257e9b3"
CODE_DIR = REPO / "third_party/openvla/prismatic/extern/hf"

from research.ar_token_counterfactual.libero_runtime import (  # noqa: E402
    build_prompt, load_policy, prepare_agentview, prepare_env_action, set_determinism,
)
from research.ar_token_counterfactual.intervention import (  # noqa: E402
    decode_action_ids, ensure_empty_action_token, projector_intervention,
)
from research.cw_lpcd.core import N_VISUAL  # noqa: E402
from research.semantic_token_cd.libero_policy import (  # noqa: E402
    embed_phrase, entity_select, extract_entities_libero,
)

ARMS = ("vanilla", "entity_cd_025", "entity_cd_050", "random_mask")
LAMBDA_BY_ARM = {"entity_cd_025": 0.25, "entity_cd_050": 0.50, "random_mask": 0.50}
K = 8
KMEANS_SEED = 0
UNNORM_KEY = "libero_object"
MAX_STEPS = 300
PROTOCOL = "LIBERO_OBJECT_SEMANTIC_ENTITY_CD_PHASE0_V1"


def jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def generate_scores(model, inputs):
    """AR-generate the 7 action tokens, returning per-position logits [n, vocab]."""
    input_ids, attention_mask = ensure_empty_action_token(inputs["input_ids"], inputs["attention_mask"])
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs["pixel_values"],
        max_new_tokens=model.get_action_dim(UNNORM_KEY),
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    return torch.cat(out["scores"], dim=0)  # [n, vocab]


def _decode(model, token_ids):
    return decode_action_ids(model, token_ids.unsqueeze(0).detach().cpu(), UNNORM_KEY)


def predict_arm(model, processor, tokenizer, image, instruction, mean, arm, rng):
    """Return (raw_action [7], meta dict) for one arm at the current obs."""
    inputs = processor(build_prompt(instruction), image).to(model.device, dtype=torch.bfloat16)
    with projector_intervention(model) as trace:
        clean_scores = generate_scores(model, inputs).float()
    h = trace.before[0].detach().float().cpu().numpy()
    meta: dict = {"arm": arm}

    if arm == "vanilla":
        token_ids = clean_scores.argmax(dim=-1)
        return _decode(model, token_ids), meta

    entities = extract_entities_libero(instruction)
    emb = [embed_phrase(model, tokenizer, e) for e in entities]
    selected, sel_meta = entity_select(h, emb, K, KMEANS_SEED)
    meta["entities"] = entities
    meta["n_entities"] = len(entities)

    if arm == "random_mask":
        # Token-count-matched random control: mask the same number of visual tokens
        # as the entity selector, but at random patch positions (per-episode seed).
        # Isolates the *semantic* selection from the mere effect of masking N tokens.
        n = len(selected)
        selected = sorted(rng.choice(N_VISUAL, size=n, replace=False).tolist())
        meta["selected_groups"] = None
        meta["selected_token_ids"] = selected
        meta["language_score"] = None
        meta["random_control"] = True
    else:
        meta.update(sel_meta)

    if not selected:
        token_ids = clean_scores.argmax(dim=-1)
        meta["degenerate"] = True
        return _decode(model, token_ids), meta

    with projector_intervention(model, selected, mean) as trace2:
        negative_scores = generate_scores(model, inputs).float()
    n_neg = negative_scores.shape[0]
    meta["degenerate"] = False
    meta["negative_truncated"] = n_neg < 7
    if n_neg < 7:
        negative_scores = torch.cat([negative_scores, clean_scores[n_neg:]], dim=0)
    elif n_neg > 7:
        negative_scores = negative_scores[:7]

    lambd = LAMBDA_BY_ARM[arm]
    final_scores = clean_scores.clone()
    final_scores[:-1] = (1 + lambd) * clean_scores[:-1] - lambd * negative_scores[:-1]
    if not torch.isfinite(final_scores).all():
        raise FloatingPointError("Non-finite entity-CD logits")
    token_ids = final_scores.argmax(dim=-1)
    meta["lambd"] = lambd
    meta["n_selected"] = len(selected)
    return _decode(model, token_ids), meta


def run_episode(env, obs, model, processor, tokenizer, mean, instruction, arm, rng):
    """Roll out one arm from the restored obs until success or MAX_STEPS."""
    trace: list[dict] = []
    steps = 0
    done = False
    while steps < MAX_STEPS and not done:
        _, image = prepare_agentview(obs)
        raw_action, meta = predict_arm(model, processor, tokenizer, image, instruction,
                                       mean, arm, rng)
        action = prepare_env_action(raw_action)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Non-finite env action (arm={arm}, step={steps})")
        obs, _, done, _ = env.step(action.tolist())
        steps += 1
        trace.append(meta)
    success = bool(env.check_success())
    return success, steps, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--task", required=True, help="LIBERO task name (Task.name)")
    p.add_argument("--episodes", required=True, help="comma-separated episode ids")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    art = a.artifact.resolve()
    episodes = [int(x) for x in a.episodes.split(",") if x]
    if a.smoke:
        episodes = episodes[:1]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = next(t for t in (suite.get_task(i) for i in range(10)) if t.name == a.task)

    mean = torch.load(art / "position_conditioned_visual_mean.pt", map_location="cpu",
                      weights_only=True)["mean"]
    set_determinism(7)
    model, processor = load_policy(CHECKPOINT, CODE_DIR)
    tokenizer = processor.tokenizer

    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    instruction = task.language
    task_root = art / "episodes" / a.task
    task_root.mkdir(parents=True, exist_ok=True)

    for ep in episodes:
        # Capture the canonical initial state for this episode.
        env.seed(ep)
        env.reset()
        state = env.get_sim_state()

        pair = {}
        for arm in ARMS:
            arm_dir = task_root / arm
            arm_dir.mkdir(exist_ok=True)
            out = arm_dir / f"episode_{ep:03d}.json"
            if out.exists():
                print(json.dumps({"task": a.task, "episode": ep, "arm": arm, "skip": True}), flush=True)
                pair[arm] = json.loads(out.read_text())
                continue
            rng = np.random.default_rng(1000 + ep)
            env.reset()
            obs = env.set_init_state(state)
            restored = np.asarray(env.get_sim_state()).copy()
            success, steps, trace = run_episode(env, obs, model, processor, tokenizer,
                                                mean, instruction, arm, rng)
            summary = {
                "protocol_id": PROTOCOL, "task": a.task, "episode": ep, "arm": arm,
                "success": success, "control_steps": steps,
                "initial_state_sha256": None, "selector_trace": jsonable(trace),
            }
            if arm != "vanilla":
                lang = [t.get("language_score") for t in trace if t.get("language_score") is not None]
                summary["mean_language_score"] = float(np.mean(lang)) if lang else None
                summary["n_degenerate_steps"] = sum(1 for t in trace if t.get("degenerate"))
            out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            pair[arm] = summary
            print(json.dumps({"task": a.task, "episode": ep, "arm": arm,
                              "success": success, "steps": steps}), flush=True)

        # Pairing sanity: all arms share the same canonical state (restored deterministically).
        print(json.dumps({"task": a.task, "episode": ep, "done": True,
                          "n_arms": len(pair)}), flush=True)

    env.close()
    print(json.dumps({"task": a.task, "DONE": True, "n_episodes": len(episodes)}), flush=True)


if __name__ == "__main__":
    main()

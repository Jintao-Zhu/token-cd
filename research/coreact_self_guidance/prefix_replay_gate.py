"""Small no-guidance gate for replaying vanilla action prefixes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare

TASKS = (4, 7)
INIT_IDS = (0, 1)
MILESTONES = ((20, "approach"), (60, "pre_grasp"), (120, "transport"))


def digest(value) -> str:
    h = hashlib.sha256()
    if isinstance(value, dict):
        for k in sorted(value):
            h.update(k.encode()); h.update(digest(value[k]).encode())
    elif isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value); h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    else:
        h.update(repr(value).encode())
    return h.hexdigest()


def state_digest(env) -> str:
    return digest(np.asarray(env.envs[0]._env.get_sim_state()))


def action_digest(actions) -> str:
    return digest(np.asarray(actions, dtype=np.float32))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    workspace, out = args.workspace.resolve(), args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    (out / "reference").mkdir(parents=True); (out / "replay").mkdir(); (out / "status").mkdir()
    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    records = []
    try:
        for task_id in TASKS:
            language = __import__("libero.libero", fromlist=["benchmark"]).benchmark.get_benchmark_dict()["libero_spatial"]().get_task(task_id).language
            for init_id in INIT_IDS:
                seed = 90000000 + task_id * 10000 + init_id * 10
                env, env_pre, env_post = make_task_env("libero_spatial", task_id, config)
                try:
                    inner = env.envs[0]; inner.init_state_id = init_id; obs, _ = env.reset(seed=seed)
                    initial_state = np.asarray(inner._env.get_sim_state()).copy(); actions=[]; refs={}
                    for step in range(max(x[0] for x in MILESTONES)):
                        if not actions or step % 10 == 0:
                            batch = prepare(policy, preprocessor, env_pre, obs, language)
                            gen = torch.Generator(device=batch["state"].device).manual_seed(seed * 1000 + step // 10)
                            noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=gen, device=batch["state"].device, dtype=batch["state"].dtype)
                            with torch.inference_mode():
                                chunk = policy.model.sample_actions(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise=noise)
                            queue = [x.detach().cpu() for x in chunk[:, :10, :7].transpose(0, 1)]
                        action = queue.pop(0); legal = env_post({"action": postprocessor(action)})["action"]; obs, _, terminated, _, _ = env.step(legal.detach().cpu().numpy()); actions.append(np.asarray(legal[0]));
                        control_step = step + 1
                        for milestone, phase in MILESTONES:
                            if control_step == milestone:
                                batch = prepare(policy, preprocessor, env_pre, env_post and env_pre(obs) if False else obs, language)
                                refs[str(milestone)] = {"phase": phase, "control_step": control_step, "state_hash": state_digest(env), "observation_hash": digest(obs), "prepared_hash": digest({k: v.detach().cpu().numpy() for k,v in batch.items() if torch.is_tensor(v)}), "action_prefix_sha256": action_digest(actions), "action_count": len(actions)}
                    ref_path = out / "reference" / f"task{task_id:02d}__init{init_id:02d}.json"; ref_path.write_text(json.dumps({"task_id":task_id,"init_state_id":init_id,"seed":seed,"language":language,"initial_state_hash":digest(initial_state),"actions":[a.tolist() for a in actions],"milestones":refs},indent=2,sort_keys=True)+"\n")
                    # Replay each saved prefix from the same initial condition.
                    for milestone, phase in MILESTONES:
                        replay_env, _, replay_post = make_task_env("libero_spatial", task_id, config)
                        try:
                            ri = replay_env.envs[0]; ri.init_state_id = init_id; robs, _ = replay_env.reset(seed=seed); prefix=actions[:milestone]
                            for a in prefix: robs, _, _, _, _ = replay_env.step(np.asarray(a)[None, :])
                            rb = prepare(policy, preprocessor, env_pre, robs, language)
                            got={"task_id":task_id,"init_state_id":init_id,"phase":phase,"control_step":milestone,"state_equal":digest(np.asarray(ri._env.get_sim_state()))==refs[str(milestone)]["state_hash"],"observation_equal":digest(robs)==refs[str(milestone)]["observation_hash"],"prepared_equal":digest({k:v.detach().cpu().numpy() for k,v in rb.items() if torch.is_tensor(v)})==refs[str(milestone)]["prepared_hash"],"action_prefix_equal":action_digest(prefix)==refs[str(milestone)]["action_prefix_sha256"]}
                            records.append(got)
                        finally: replay_env.close()
                finally: env.close()
    finally:
        pass
    (out/"replay_audit.json").write_text(json.dumps(records,indent=2,sort_keys=True)+"\n")
    passed=all(x["state_equal"] and x["observation_equal"] and x["prepared_equal"] and x["action_prefix_equal"] for x in records)
    decision="PREFIX_REPLAY_VALIDATED" if passed else "PREFIX_REPLAY_INTEGRITY_FAILED"
    (out/"decision.json").write_text(json.dumps({"decision":decision,"points":len(records),"passed":passed},indent=2)+"\n")
    (out/"status"/("integrity.pass" if passed else "integrity.fail")).write_text(decision+"\n")
    print(json.dumps({"decision":decision,"points":len(records),"passed":passed},indent=2))


if __name__ == "__main__": main()

"""Paired SpatialVLA Vanilla / VTCG rollout on remaining SIMPLER tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
TOKEN_CD = WORKSPACE / "token-cd"
MODEL_DIR = WORKSPACE / "checkpoints/spatialvla-4b-224-sft-fractal"
BRIDGE_MODEL_DIR = WORKSPACE / "checkpoints/spatialvla-4b-224-sft-bridge"
PCD_SOURCE = WORKSPACE / "pcd_openvla_simpler_box_31b027e/source"
SPATIAL_REPO = WORKSPACE / "SpatialVLA-SimplerEnv"
MANISKILL_REPO = SPATIAL_REPO / "ManiSkill2_real2sim"
OVERLAY = WORKSPACE / "spatialvla-runtime-overlay-20260928"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
)
REMAINING_TASKS = TASKS[4:]
ARMS = ("vanilla", "l11_matched")
DEPTH = {task: (8 if task in (TASKS[0], TASKS[1]) else 0) for task in TASKS}
POLICY_SETUP = {task: ("widowx_bridge" if task.startswith("widowx_") else "google_robot")
                for task in TASKS}
MODEL_BY_SETUP = {"google_robot": MODEL_DIR, "widowx_bridge": BRIDGE_MODEL_DIR}
PROTOCOL = "SPATIALVLA_SIMPLER_L11_MATCHED_REMAINING5_PAIRED_V1"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jwrite(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def bootstrap() -> None:
    for p in (OVERLAY, SPATIAL_REPO, MANISKILL_REPO,
              TOKEN_CD / "task1/shim_site", TOKEN_CD, PCD_SOURCE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


_PATCHED_DEPTH = None


def patch_runtime(depth: int) -> None:
    global _PATCHED_DEPTH
    _PATCHED_DEPTH = int(depth)
    from mani_skill2_real2sim.envs.sapien_env import BaseEnv
    import gymnasium
    from gymnasium.core import Wrapper

    if not getattr(BaseEnv, "_spatialvla_depth_patch_installed", False):
        init = BaseEnv.__init__
        def pinned_init(self, *args, **kwargs):
            if kwargs.get("shader_dir") == "rt":
                rc = dict(kwargs.get("render_config") or {})
                rc.update(rt_samples_per_pixel=32, rt_max_path_depth=_PATCHED_DEPTH,
                          rt_use_denoiser=True)
                kwargs["render_config"] = rc
            return init(self, *args, **kwargs)
        BaseEnv.__init__ = pinned_init
        BaseEnv._spatialvla_depth_patch_installed = True

    if not hasattr(Wrapper, "__getattr__"):
        def forward_custom_attribute(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            try:
                inner = object.__getattribute__(self, "env")
            except AttributeError as exc:
                raise AttributeError(name) from exc
            return getattr(inner, name)
        Wrapper.__getattr__ = forward_custom_attribute


def make_env(task: str, gpu: int):
    import gymnasium as gym
    import simpler_env
    rkwargs = {"device": "cuda:0", "offscreen_only": True}
    if task == "google_robot_pick_coke_can":
        env = gym.make("GraspSingleOpenedCokeCanDistractorInScene-v0", obs_mode="rgbd",
                       prepackaged_config=True, distractor_config="less",
                       shader_dir="rt", renderer_kwargs=rkwargs,
                       render_config={"rt_samples_per_pixel": 32,
                                      "rt_max_path_depth": 0,
                                      "rt_use_denoiser": True})
        return env, "GraspSingleOpenedCokeCanDistractorInScene-v0"
    env_id, base = simpler_env.ENVIRONMENT_MAP[task]
    kwargs = dict(base)
    kwargs.update(prepackaged_config=True, shader_dir="rt", renderer_kwargs=rkwargs,
                  render_config={"rt_samples_per_pixel": 32,
                                 "rt_max_path_depth": DEPTH[task],
                                 "rt_use_denoiser": True})
    return gym.make(env_id, obs_mode="rgbd", **kwargs), env_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seeds", default="100-199")
    ap.add_argument("--model-setup", choices=("google_robot", "widowx_bridge"), required=True)
    ap.add_argument("--preflight", action="store_true", help="one paired seed and decoder parity audit")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    bootstrap()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0); np.random.seed(0); random.seed(0); torch.cuda.manual_seed_all(0)
    from PIL import Image
    from transforms3d.euler import euler2axangle
    from transformers.cache_utils import HybridCache
    from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from utils import convert_numpy_or_torch_to_python, stat_first, stat_final, summarize
    from research.semantic_token_cd.rollout_policy import extract_entities
    from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct
    from research.semantic_token_cd.rollout_pilot import (
        capture_snapshot, restore_snapshot, snapshot_sha, clone, flatten_action,
    )

    TASK_CD = tuple(x for x in args.tasks.split(",") if x)
    if any(t not in REMAINING_TASKS for t in TASK_CD):
        raise ValueError(f"unsupported task list: {TASK_CD}")
    if any(POLICY_SETUP[t] != args.model_setup for t in TASK_CD):
        raise ValueError(f"tasks {TASK_CD} do not match policy setup {args.model_setup}")
    model_dir = MODEL_BY_SETUP[args.model_setup]
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"SpatialVLA checkpoint is not installed: {model_dir}")
    seeds = [100] if args.preflight else parse_seeds(args.seeds)
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_id": PROTOCOL, "tasks": list(REMAINING_TASKS),
        "seeds_by_task": {t: list(range(100, 200)) for t in REMAINING_TASKS},
        "arms": list(ARMS),
        "models_by_policy_setup": {k: str(v) for k, v in MODEL_BY_SETUP.items()},
        "policy_setup_by_task": POLICY_SETUP,
        "renderer": {t: {"backend": "SAPIEN RT", "depth": DEPTH[t],
                         "spp": 32, "denoiser": True} for t in REMAINING_TASKS},
        "pairing": "snapshot restore; exact simulator-state and initial-RGB hashes required",
        "selector": "SpatialVLA post-softmax layer-11 instruction-to-image attention top-m",
        "matched_m": "KMeans(K=8, random_state=0, n_init=10), phrase embedding cosine entity-cluster union count",
        "negative": "16x16 4-neighbor harmonic reconstruction beta=0 on post-projector/post-Ego3D features",
        "contrastive": "greedy full-vocabulary scores (1+0.5)*positive - 0.5*negative; recompute negative on shared guided prefix",
        "action_chunk_size": 4, "lambda": 0.5,
        "seed_allocation": {
            "gpu2": "seeds 100-149 for all five tasks",
            "gpu3": "seeds 150-199 for all five tasks",
        },
    }
    lock = artifact / "CONFIG_LOCK.json"
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise RuntimeError("SpatialVLA experiment CONFIG_LOCK mismatch")
    jwrite(lock, config)

    patch_runtime(DEPTH[TASK_CD[0]])
    base = SpatialVLAInference(saved_model_path=str(model_dir), policy_setup=args.model_setup,
                               action_ensemble_temp=-0.8)
    proc = base.processor
    model = base.vla
    model.eval()

    def get_inp(image, instruction):
        image = base._resize_image(image)
        base._add_image_to_history(image)
        images = base._obtain_image_history()
        raw = proc(images=images, text=instruction, unnorm_key=base.unnorm_key,
                   return_tensors="pt", do_normalize=False)
        return raw.to(torch.bfloat16).to(model.device)

    def vision_and_attention(inputs, instruction):
        with torch.inference_mode():
            visual_cache = HybridCache(
                config=model.config.text_config, batch_size=1,
                max_cache_len=inputs["input_ids"].shape[1] + 256,
                device=model.device, dtype=model.dtype,
            )
            out = model(**inputs, past_key_values=visual_cache,
                        use_cache=True, output_attentions=True, return_dict=True)
        features = out.image_hidden_states[0].detach().float()
        ids = inputs["input_ids"][0]
        image_token = int(model.config.image_token_index)
        visual_pos = torch.nonzero(ids == image_token, as_tuple=False).flatten()
        if visual_pos.numel() != 256:
            raise RuntimeError(f"expected 256 image positions, got {visual_pos.numel()}")
        instruction_ids = proc.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        sequence = ids.detach().cpu().tolist()
        start = None
        for i in range(max(0, len(sequence)-len(instruction_ids)+1)):
            if sequence[i:i+len(instruction_ids)] == instruction_ids:
                start = i; break
        if start is None:
            raise RuntimeError(f"cannot locate instruction token sequence in SpatialVLA input: {sequence[-30:]}")
        qpos = list(range(start, start+len(instruction_ids)))
        attn = out.attentions[11][0, :, qpos, :][:, :, visual_pos]
        score = attn.float().mean(dim=(0, 1)).cpu().numpy()
        if score.shape != (256,) or not np.isfinite(score).all():
            raise RuntimeError(f"invalid L11 attention shape/values: {score.shape}")
        return features, score, {"query_positions": qpos, "visual_positions": visual_pos.cpu().tolist(),
                                 "attention_sum": float(score.sum())}, out

    def matched_count(features, instruction):
        from sklearn.cluster import KMeans
        hd = features.detach().float().cpu().numpy().astype(np.float64)
        labels = KMeans(n_clusters=8, random_state=0, n_init=10).fit(hd).labels_
        vectors = np.stack([hd[labels == k].mean(0) for k in range(8)])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        groups = []
        for phrase in extract_entities(instruction):
            ids = proc.tokenizer(phrase, add_special_tokens=False)["input_ids"]
            if not ids: continue
            it = torch.tensor([ids], device=model.device, dtype=torch.long)
            emb = model.language_model.model.embed_tokens(it)[0].float().mean(0).detach().cpu().numpy()
            emb /= np.linalg.norm(emb) + 1e-8
            g = int(np.argmax(vectors @ emb))
            if g not in groups: groups.append(g)
        selected_cluster_ids = [int(i) for g in groups for i in np.flatnonzero(labels == g)]
        if not selected_cluster_ids:
            raise RuntimeError("matched entity union selected no visual tokens")
        return len(set(selected_cluster_ids)), labels, groups

    def gen(inputs, negative_features=None, positive_prefill=None):
        """Greedy cached decoding; mirrors generate() while running two KV caches."""
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"], dtype=torch.long)
        ids = inputs["input_ids"]
        pos_kwargs = {k: v for k, v in inputs.items()
                      if k in {"pixel_values", "intrinsic", "token_type_ids"}}
        def new_cache():
            return HybridCache(config=model.config.text_config, batch_size=1,
                               max_cache_len=ids.shape[1] + 256,
                               device=model.device, dtype=model.dtype)
        with torch.inference_mode():
            pos = positive_prefill
            if pos is None:
                pos = model(input_ids=ids, attention_mask=attention_mask,
                            past_key_values=new_cache(), use_cache=True,
                            return_dict=True, **pos_kwargs)
            pos_cache = pos.past_key_values
            neg_cache = None
            if negative_features is not None:
                original = model.get_image_features
                model.get_image_features = lambda *_a, **_kw: negative_features
                try:
                    neg = model(input_ids=ids, attention_mask=attention_mask,
                                past_key_values=new_cache(), use_cache=True,
                                return_dict=True, **pos_kwargs)
                finally:
                    model.get_image_features = original
                neg_cache = neg.past_key_values
                neg_scores = neg.logits[:, -1, :].float()
            pos_scores = pos.logits[:, -1, :].float()
            generated = []
            eos = int(proc.tokenizer.eos_token_id)
            for _ in range(256):
                scores = pos_scores if neg_cache is None else 1.5 * pos_scores - 0.5 * neg_scores
                token = scores.argmax(dim=-1, keepdim=True)
                generated.append(token)
                ids = torch.cat((ids, token), dim=1)
                if int(token.item()) == eos:
                    break
                attention_mask = torch.cat((attention_mask,
                                            torch.ones_like(token, dtype=attention_mask.dtype)), dim=1)
                one_kwargs = {}
                if "token_type_ids" in inputs:
                    # SpatialVLA inference processors normally omit these; preserve
                    # a zero-valued generated suffix if they are supplied.
                    one_kwargs["token_type_ids"] = torch.cat((inputs["token_type_ids"],
                                                               torch.zeros_like(ids[:, inputs["input_ids"].shape[1]:])), dim=1)
                pos = model(input_ids=token, attention_mask=attention_mask,
                            past_key_values=pos_cache, use_cache=True, return_dict=True,
                            **one_kwargs)
                pos_cache = pos.past_key_values
                pos_scores = pos.logits[:, -1, :].float()
                if neg_cache is not None:
                    neg = model(input_ids=token, attention_mask=attention_mask,
                                past_key_values=neg_cache, use_cache=True, return_dict=True,
                                **one_kwargs)
                    neg_cache = neg.past_key_values
                    neg_scores = neg.logits[:, -1, :].float()
        return torch.cat(generated, dim=1)

    def action_from_generated(generated, arm):
        decoded = base.processor.decode_actions(generation_outputs=generated,
                                                unnorm_key=base.unnorm_key)["actions"]
        raw_actions = decoded.detach().cpu().numpy() if torch.is_tensor(decoded) else np.asarray(decoded)
        if base.action_ensemble:
            raw_actions = base.action_ensembler.ensemble_action(raw_actions)[None]
        raw_action = {"world_vector": np.array(raw_actions[0, :3]),
                      "rotation_delta": np.array(raw_actions[0, 3:6]),
                      "open_gripper": np.array(raw_actions[0, 6:7])}
        action = {"world_vector": raw_action["world_vector"] * base.action_scale}
        roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        axis, angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = axis * angle * base.action_scale
        if base.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        else:
            if base.previous_gripper_action is None:
                relative = np.array([0]); base.previous_gripper_action = raw_action["open_gripper"]
            else:
                relative = base.previous_gripper_action - raw_action["open_gripper"]
            if np.abs(relative) > 0.5 and not base.sticky_action_is_on:
                base.sticky_action_is_on = True
                base.sticky_gripper_action = relative
                base.previous_gripper_action = raw_action["open_gripper"]
            if base.sticky_action_is_on:
                base.gripper_action_repeat += 1
                relative = base.sticky_gripper_action
            if base.gripper_action_repeat == base.sticky_gripper_num_repeat:
                base.sticky_action_is_on = False; base.gripper_action_repeat = 0
                base.sticky_gripper_action = 0.0
            action["gripper"] = relative
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def run_policy(env, obs, arm, instruction, seed, single_step=False):
        base.reset(instruction)
        if arm == "vanilla":
            image = get_image_from_maniskill2_obs_dict(env, obs)
            with torch.inference_mode():
                raw, action = base.step(image, task_description=instruction)
            obs, _, terminated, truncated, info = env.step(flatten_action(action))
            infos = [convert_numpy_or_torch_to_python(info)]
            steps = 1
        else:
            infos = []; terminated = truncated = False; steps = 0
        trace = []
        # First vanilla step already computed above. Run the rest of vanilla loop; L11 all steps.
        while not (terminated or truncated) and not (single_step and steps >= 1):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            if arm == "l11_matched":
                inputs = get_inp(image, instruction)
                features, score, ameta, positive_prefill = vision_and_attention(inputs, instruction)
                m, labels, groups = matched_count(features, instruction)
                ids = np.lexsort((np.arange(256), -score))[:m]
                neg_np = features.detach().float().cpu().numpy().copy()
                neg_np[ids] = harmonic_reconstruct(neg_np, ids, beta=0.0)
                neg = torch.as_tensor(neg_np, device=features.device, dtype=features.dtype)[None]
                trace.append({"step": steps, "m_t": m, "selected_token_ids": ids.tolist(),
                              "entity_clusters": groups, "l11_attention_sha256": sha(score.tobytes()),
                              "l11_attention": ameta})
            if arm == "vanilla":
                with torch.inference_mode():
                    raw, action = base.step(image, task_description=instruction)
            else:
                generated = gen(inputs, neg, positive_prefill=positive_prefill)
                raw, action = action_from_generated(generated, arm)
            obs, _, terminated, truncated, info = env.step(flatten_action(action))
            infos.append(convert_numpy_or_torch_to_python(info)); steps += 1
            instruction = env.unwrapped.get_language_instruction()
        result = summarize(infos); result.update(stat_first(infos)); result.update(stat_final(infos))
        return result, obs, steps, trace

    from research.semantic_token_cd.rollout_pilot import wrapped_observation
    for task in TASK_CD:
        # each GPU owns an independent worker, but each worker handles disjoint seeds
        task_dir = artifact / "episodes" / task
        snap_dir = artifact / "snapshots" / task
        task_dir.mkdir(parents=True, exist_ok=True); snap_dir.mkdir(parents=True, exist_ok=True)
        patch_runtime(DEPTH[task])
        env, env_id = make_env(task, args.gpu)
        try:
            for seed in seeds:
                snap_path = snap_dir / f"seed_{seed:03d}.pkl"
                if snap_path.exists():
                    snapshot = pickle.loads(snap_path.read_bytes())
                else:
                    snapshot = capture_snapshot(env, seed)
                    snap_path.write_bytes(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL))
                canonical_sha = snapshot_sha(snapshot)
                if args.preflight and task == TASK_CD[0]:
                    obs0, _, _ = restore_snapshot(env, seed, snapshot)
                    image0 = get_image_from_maniskill2_obs_dict(env, obs0)
                    base.reset(snapshot["instruction"])
                    inputs0 = get_inp(image0, snapshot["instruction"])
                    with torch.inference_mode():
                        official_ids = model.predict_action(inputs0).detach().cpu()
                    base.reset(snapshot["instruction"])
                    inputs1 = get_inp(image0, snapshot["instruction"])
                    custom_ids = gen(inputs1, None).detach().cpu()
                    if not torch.equal(official_ids, custom_ids):
                        raise RuntimeError(
                            "custom greedy decoder does not match SpatialVLA official predict_action: "
                            f"official={official_ids.tolist()} custom={custom_ids.tolist()}"
                        )
                    print(json.dumps({"preflight": "decoder-parity-PASS",
                                      "task": task, "seed": seed,
                                      "generated_action_token_ids": official_ids.tolist(),
                                      "gpu": args.gpu}), flush=True)
                summaries = {}
                starts = {}
                for arm in ARMS:
                    outdir = task_dir / arm; outdir.mkdir(parents=True, exist_ok=True)
                    summary_path = outdir / f"episode_{seed:03d}_summary.json"
                    if summary_path.exists():
                        summaries[arm] = json.loads(summary_path.read_text())
                        starts[arm] = (summaries[arm]["initial_state_sha256"], summaries[arm]["initial_rgb_sha256"])
                        continue
                    obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
                    starts[arm] = (state_hash, rgb_hash)
                    instruction = env.unwrapped.get_language_instruction()
                    base.reset(instruction)
                    started = time.monotonic()
                    result, final_obs, steps, trace = run_policy(
                        env, obs, arm, instruction, seed, single_step=args.preflight
                    )
                    row = {"protocol_id": PROTOCOL, "task": task, "environment_id": env_id,
                           "seed": seed, "arm": arm, "success": bool(result["success"]),
                           "result": result, "control_steps": steps,
                           "runtime_seconds": time.monotonic()-started,
                           "canonical_snapshot_sha256": canonical_sha,
                           "initial_state_sha256": state_hash, "initial_rgb_sha256": rgb_hash,
                           "renderer": {"backend": "SAPIEN RT", "depth": DEPTH[task], "spp": 32,
                                        "denoiser": True}, "gpu": args.gpu,
                           "checkpoint": str(model_dir), "policy_setup": args.model_setup,
                           "l11_trace": trace if arm == "l11_matched" else None}
                    jwrite(summary_path, row); summaries[arm] = row
                    print(json.dumps({"task":task,"seed":seed,"arm":arm,"success":row["success"],
                                      "steps":steps,"runtime_s":round(row["runtime_seconds"],1),
                                      "worker":args.worker_id}), flush=True)
                if len(set(starts.values())) != 1:
                    raise RuntimeError(f"paired initial state/RGB mismatch: {task} {seed} {starts}")
                if any(r["canonical_snapshot_sha256"] != canonical_sha for r in summaries.values()):
                    raise RuntimeError("paired canonical snapshot mismatch")
        finally:
            env.close()


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a,b = map(int, part.split("-",1)); out.extend(range(a,b+1))
        elif part.strip(): out.append(int(part))
    return sorted(set(out))


if __name__ == "__main__":
    main()

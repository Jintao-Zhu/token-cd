"""Fractal Pi0 SIMPLER rollout with L2 token mask and action-distribution PCD.

The image counterfactual is applied to projected SigLIP patch features before
the PiZero language stack. Positive and negative flow branches are sampled
independently with shared initial action noise; PCD reweights positive action
chunk candidates by a KDE likelihood ratio. Flow velocities are never mixed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws")
REPO = ROOT / "token-cd"
OPENPI = ROOT / "open-pi-zero"
CHECKPOINT = ROOT / "checkpoints/pi0/fractal_beta_step29576_2024-12-29_13-10_42.pt"
OUT_DEFAULT = REPO / "artifacts/pi0_fractal_l2_matched_google800_v2"
TASKS = {
    "google_robot_open_drawer": {"depth": 8},
    "google_robot_close_drawer": {"depth": 8},
    "google_robot_pick_coke_can": {"depth": 8},
    "google_robot_move_near": {"depth": 8},
}
LAYER = 2
N_CANDIDATES = 24
PCD_ALPHA = 0.2


def sha_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def clone_state(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return type(value)((k, clone_state(v)) for k, v in value.items())
    if isinstance(value, list):
        return [clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_state(v) for v in value)
    if hasattr(value, "p") and hasattr(value, "q"):
        return type(value)(np.asarray(value.p).copy(), np.asarray(value.q).copy())
    return copy.deepcopy(value)


def wrapped_observation(env):
    obs = env.unwrapped.get_obs()
    wrappers, current = [], env
    while hasattr(current, "env"):
        if hasattr(current, "observation"):
            wrappers.append(current)
        current = current.env
    for wrapper in reversed(wrappers):
        obs = wrapper.observation(obs)
    return obs


def capture_snapshot(env, seed):
    env.reset(seed=seed)
    inner = env.unwrapped
    return {"sim_state": np.asarray(inner.get_state()).copy(),
            "agent_state": clone_state(inner.agent.get_state()),
            "rng_state": clone_state(inner._episode_rng.get_state()),
            "instruction": inner.get_language_instruction()}


def snapshot_sha(snapshot):
    h = hashlib.sha256()
    h.update(sha_array(snapshot["sim_state"]).encode())
    h.update(repr(snapshot["agent_state"]).encode())
    h.update(repr(snapshot["rng_state"]).encode())
    h.update(snapshot["instruction"].encode())
    return h.hexdigest()


def restore_snapshot(env, seed, snapshot, get_image):
    env.reset(seed=seed)
    inner = env.unwrapped
    inner.set_state(snapshot["sim_state"].copy())
    inner.agent.set_state(clone_state(snapshot["agent_state"]))
    inner._episode_rng.set_state(clone_state(snapshot["rng_state"]))
    inner._elapsed_steps = 0
    obs = wrapped_observation(env)
    state_hash = sha_array(np.asarray(inner.get_state()))
    rgb_hash = sha_array(np.asarray(get_image(env, obs)))
    return obs, state_hash, rgb_hash


def parse_entities(instruction: str) -> list[str]:
    preps = {"into", "in", "on", "onto", "near", "to", "next", "from", "under", "over", "behind", "beside", "above", "below", "inside", "at", "with"}
    dets = {"the", "a", "an"}
    words = instruction.lower().replace("_", " ").split()
    rest = words[1:]
    pos = [i for i, token in enumerate(rest) if token in preps]
    if not pos:
        phrase = " ".join(w for w in rest if w not in dets)
        return [phrase or instruction.lower()]
    source = " ".join(w for w in rest[:pos[0]] if w not in dets)
    target = " ".join(w for w in rest[pos[-1] + 1:] if w not in dets)
    entities = list(dict.fromkeys(x for x in (source, target) if x))
    return entities or [instruction.lower()]


def entity_budget_and_attention(model, adapter, inputs, instruction: str, layer_index: int):
    import torch
    from sklearn.cluster import KMeans
    from src.model.vla import joint_model as joint_model_module

    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    prefix = model._forward_siglip_and_text_embedding(input_ids, pixel_values)
    image_positions = torch.nonzero(input_ids[0] == model.image_token_index, as_tuple=False).flatten()
    if image_positions.numel() != 256:
        raise RuntimeError(f"expected 256 image tokens, got {image_positions.numel()}")
    image_features = prefix[0, image_positions].detach().float().cpu().numpy()
    labels = KMeans(n_clusters=8, random_state=0, n_init=10).fit_predict(image_features)
    group = np.stack([image_features[labels == k].mean(0) for k in range(8)])
    group /= np.linalg.norm(group, axis=1, keepdims=True) + 1e-8
    selected_groups = []
    entities = parse_entities(instruction)
    for phrase in entities:
        token_ids = adapter.tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        ids = torch.as_tensor(token_ids, device=device, dtype=torch.long)[None]
        with torch.inference_mode():
            vector = model.embed_tokens(ids)[0].mean(0).float().cpu().numpy()
        vector /= np.linalg.norm(vector) + 1e-8
        group_id = int(np.argmax(group @ vector))
        if group_id not in selected_groups:
            selected_groups.append(group_id)
    selected = sorted(int(i) for g in selected_groups for i in np.flatnonzero(labels == g))
    if not selected:
        raise RuntimeError(f"Matched token budget is empty for {instruction!r}")
    text_ids = adapter.tokenizer(instruction, add_special_tokens=False)["input_ids"]
    all_ids = input_ids[0].detach().cpu().tolist()
    query_starts = [i for i in range(len(all_ids)-len(text_ids)+1) if all_ids[i:i+len(text_ids)] == text_ids]
    if not query_starts:
        raise RuntimeError(f"cannot find task phrase token positions for {instruction!r}")
    query_positions = list(range(query_starts[-1], query_starts[-1] + len(text_ids)))

    causal, vlm_pos, prop_pos, _ = model.build_causal_mask_and_position_ids(inputs["attention_mask"], dtype=pixel_values.dtype)
    causal = causal.to(device); vlm_pos = vlm_pos.to(device); prop_pos = prop_pos.to(device)
    vlm_prop_mask, _ = model.split_full_mask_into_submasks(causal)
    proprio_embeds = model.proprio_encoder(inputs["proprios"])
    capture = {"attention": None}
    previous_callback = joint_model_module.ATTENTION_CAPTURE_CALLBACK

    def callback(index, active_names, q_lens, weights):
        if index == layer_index:
            qidx = torch.as_tensor(query_positions, device=weights.device, dtype=torch.long)
            capture["attention"] = weights[0, :, qidx, :256].float().mean(dim=(0, 1)).detach().cpu().numpy()

    joint_model_module.ATTENTION_CAPTURE_CALLBACK = callback
    try:
        with torch.inference_mode():
            caches = model.joint_model.build_mixture_caches()
            model.joint_model(
                attention_mask=vlm_prop_mask,
                position_ids_all={"vlm": vlm_pos, "proprio": prop_pos},
                embeds_all={"vlm": prefix.clone(), "proprio": proprio_embeds.clone()},
                kv_caches=caches,
                return_caches=True,
            )
    finally:
        joint_model_module.ATTENTION_CAPTURE_CALLBACK = previous_callback
    attention = capture["attention"]
    if attention is None or attention.shape != (256,) or not np.isfinite(attention).all():
        raise RuntimeError("failed to capture finite L2 prompt-to-image attention")
    ids = sorted(int(i) for i in np.lexsort((np.arange(256), -attention))[:len(selected)])
    return prefix, image_positions, proprio_embeds, (causal, vlm_pos, prop_pos), selected, ids, attention


def make_prefill(model, prefix, proprio_embeds, masks, candidate_count: int):
    import torch
    from src.model.kv_cache import KVCache

    causal, vlm_pos, prop_pos = masks
    vlm_prop_mask, _ = model.split_full_mask_into_submasks(causal)
    cache = model.joint_model.build_mixture_caches()
    with torch.inference_mode():
        _, cache = model.joint_model(
            attention_mask=vlm_prop_mask,
            position_ids_all={"vlm": vlm_pos, "proprio": prop_pos},
            embeds_all={"vlm": prefix.clone(), "proprio": proprio_embeds.clone()},
            kv_caches=cache,
            return_caches=True,
        )
    expanded = {}
    for name, old in cache.items():
        new = KVCache()
        new.key_cache = [v.expand(candidate_count, *v.shape[1:]) for v in old.key_cache]
        new.value_cache = [v.expand(candidate_count, *v.shape[1:]) for v in old.value_cache]
        expanded[name] = new
    return expanded


def sample_branch(model, inputs, prefix, proprio_embeds, masks, noise, *, clip_output=True):
    import torch
    count = noise.shape[0]
    caches = make_prefill(model, prefix, proprio_embeds, masks, count)
    _, action_mask = model.split_full_mask_into_submasks(masks[0])
    action_mask = action_mask.expand(count, *action_mask.shape[1:])
    # Match PiZeroInference.build_causal_mask_and_position_ids instead of
    # assuming one proprio token and a particular action horizon.
    _, _, _, action_positions = model.build_causal_mask_and_position_ids(
        inputs["attention_mask"], dtype=inputs["pixel_values"].dtype
    )
    action_positions = action_positions.to(noise.device).expand(count, -1)
    state_batch = inputs["proprios"].expand(count, *inputs["proprios"].shape[1:])
    x = noise.clone()
    dt = 1.0 / model.num_inference_steps
    t = torch.zeros(count, device=x.device, dtype=x.dtype)
    with torch.inference_mode():
        for _ in range(model.num_inference_steps):
            time_cond = model.time_embedding(t)
            if model.action_expert_adaptive_mode:
                action_embeds = model.action_encoder(x)
            else:
                action_embeds = model.action_encoder(x, time_cond)
            action_embeds = model.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_positions},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=caches,
                cache_mode="append_non_active",
            )["action"]
            x = x + dt * model.action_decoder(action_embeds)
            t = t + dt
    if clip_output and model.final_action_clip_value is not None:
        x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
    return x[:, : model.horizon_steps, : model.action_dim]


class FractalPi0PCD:
    def __init__(self, model, adapter, *, candidates=N_CANDIDATES, alpha=PCD_ALPHA, layer=LAYER,
                 decoder="whole_chunk"):
        self.model, self.adapter = model, adapter
        self.candidates, self.alpha, self.layer = int(candidates), float(alpha), int(layer)
        if decoder not in {
            "whole_chunk", "whole_chunk_raw", "official_coordinate_shared",
            "official_coordinate_independent", "positive_coordinate",
        }:
            raise ValueError(f"unknown action decoder: {decoder}")
        self.decoder = decoder
        self.step_index = 0

    def predict(self, env, obs, instruction, seed):
        import torch
        from src.model.vla import joint_model as joint_model_module
        from research.semantic_token_cd.pi0_intact_token_pcd import harmonic_reconstruct, _kde_log_densities

        inputs = self.adapter.preprocess(env, obs, instruction)
        inputs = {k: (v.to(next(self.model.parameters()).device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        prefix, image_positions, proprio_embeds, masks, selected, ids, attention = entity_budget_and_attention(
            self.model, self.adapter, inputs, instruction, self.layer
        )
        features = prefix[0, image_positions].detach().float().cpu().numpy()
        reconstructed = harmonic_reconstruct(features, ids)
        negative_prefix = prefix.clone()
        negative_prefix[0, image_positions[ids]] = torch.as_tensor(reconstructed, device=prefix.device, dtype=prefix.dtype)
        # The caller already folds the current replan index into `seed`.
        # Re-adding step_index here would give guided sampling a different,
        # double-offset noise schedule from the paired vanilla replan.
        action_seed = int(seed)
        generator = torch.Generator(device=prefix.device).manual_seed(action_seed)
        noise_shape = (self.candidates, self.model.horizon_steps, self.model.action_dim)
        if self.decoder == "official_coordinate_independent":
            # The official PiZero wrapper concatenates positive and negative
            # inputs along batch before drawing [2*N,T,D] independent noise.
            all_noise = torch.randn((2 * self.candidates, *noise_shape[1:]),
                                    generator=generator, device=prefix.device, dtype=prefix.dtype)
            positive_noise, negative_noise = all_noise[:self.candidates], all_noise[self.candidates:]
        else:
            # Common random numbers make the shared-noise decoder comparison
            # lower variance; this is deliberately logged as a separate arm.
            positive_noise = torch.randn(noise_shape, generator=generator, device=prefix.device, dtype=prefix.dtype)
            negative_noise = positive_noise
        clip_candidates = self.decoder == "whole_chunk"
        positive = sample_branch(self.model, inputs, prefix, proprio_embeds, masks, positive_noise,
                                 clip_output=clip_candidates)
        if self.decoder == "positive_coordinate":
            negative = positive
        else:
            negative = sample_branch(self.model, inputs, negative_prefix, proprio_embeds, masks, negative_noise,
                                     clip_output=clip_candidates)
        pos = positive.detach().float().cpu().numpy()
        neg = negative.detach().float().cpu().numpy()
        decoder_trace = {}
        if self.decoder in {"official_coordinate_shared", "official_coordinate_independent", "positive_coordinate"}:
            from research.semantic_token_cd.pi0_pcd_coordinate_decoder import official_pcd_coordinate_decode

            decoded, decoder_trace = official_pcd_coordinate_decode(
                positive,
                negative,
                alpha=0.0 if self.decoder == "positive_coordinate" else self.alpha,
                keep_threshold=0.5,
                motion_dims=6,
                clip_value=self.model.final_action_clip_value,
            )
            chosen = decoded[0].detach().float().cpu().numpy()
            chosen_index = None
            weights = None
            bandwidth = None
        else:
            log_pos, bandwidth = _kde_log_densities(pos, pos)
            log_neg, _ = _kde_log_densities(pos, neg)
            log_w = self.alpha * (log_pos - log_neg)
            log_w -= float(log_w.max())
            weights = np.exp(log_w); weights /= weights.sum()
            chosen_index = int(np.random.default_rng(action_seed ^ 0x504344).choice(self.candidates, p=weights))
            chosen = positive[chosen_index].detach().float().cpu().numpy()
            if self.decoder == "whole_chunk_raw" and self.model.final_action_clip_value is not None:
                chosen = np.clip(chosen, -self.model.final_action_clip_value, self.model.final_action_clip_value)
        env_actions = self.adapter.postprocess(chosen)
        if not np.isfinite(env_actions).all():
            raise FloatingPointError("nonfinite decoded SIMPLER action")
        trace = {
            "pcd_action_seed": action_seed,
            "pcd_replan_index": self.step_index,
            "layer_index_0based": self.layer, "m_t": len(selected),
            "selected_token_ids": ids, "attention_sha256": sha_array(attention),
            "harmonic_reconstruction_finite": bool(np.isfinite(reconstructed).all()),
            "positive_candidates_sha256": sha_array(pos),
            "negative_candidates_sha256": sha_array(neg),
            "pcd_candidate_weights": None if weights is None else weights.tolist(), "candidate_count": self.candidates,
            "pcd_alpha": 0.0 if self.decoder == "positive_coordinate" else self.alpha,
            "selected_positive_candidate": chosen_index,
            "flow_velocity_mixing": False, "guidance_method": "PCD action-distribution KDE decoding",
            "action_decoder": self.decoder, "decoder_diagnostics": decoder_trace,
            "matched_groups": selected,
        }
        self.step_index += 1
        return env_actions, trace


def renderer_depth(task: str) -> int:
    return int(TASKS[task]["depth"])


def make_env(task: str):
    import gymnasium as gym
    import simpler_env
    if task == "google_robot_pick_coke_can":
        return gym.make("GraspSingleOpenedCokeCanDistractorInScene-v0", obs_mode="rgbd",
                        prepackaged_config=True, distractor_config="less", shader_dir="rt",
                        renderer_kwargs={"device": "cuda:0", "offscreen_only": True},
                        render_config={"rt_samples_per_pixel": 32, "rt_max_path_depth": renderer_depth(task), "rt_use_denoiser": True}), \
               "GraspSingleOpenedCokeCanDistractorInScene-v0"
    info = simpler_env.ENVIRONMENT_MAP[task]
    env_id, kwargs = info
    kwargs = dict(kwargs)
    kwargs.update(prepackaged_config=True, shader_dir="rt",
                  renderer_kwargs={"device": "cuda:0", "offscreen_only": True},
                  render_config={"rt_samples_per_pixel": 32, "rt_max_path_depth": renderer_depth(task), "rt_use_denoiser": True})
    return gym.make(env_id, obs_mode="rgbd", **kwargs), env_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS), required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--artifact", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--candidates", type=int, default=N_CANDIDATES)
    ap.add_argument("--layer", type=int, choices=range(18), default=LAYER,
                    help="zero-based PaliGemma language layer used for prompt attention")
    ap.add_argument("--max-steps", type=int, default=0, help="nonzero limits a technical smoke")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError(f"launch worker with CUDA_VISIBLE_DEVICES={args.gpu}")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "checkpoints/pi0"))
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path[:0] = [str(OPENPI), str(REPO / "task1/shim_site"), str(REPO)]
    os.chdir(OPENPI)

    import torch
    import hydra
    from omegaconf import OmegaConf
    from src.model.vla.pizero import PiZeroInference
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(0); np.random.seed(0); random.seed(0); torch.cuda.manual_seed_all(0)
    cfg = OmegaConf.load("config/eval/fractal_drawer.yaml")
    cfg.use_torch_compile = False
    cfg.use_bf16 = False
    cfg.checkpoint_path = str(CHECKPOINT)
    cfg.gpu_id = 0
    cfg.env.adapter.dataset_statistics_path = str(OPENPI / "config/fractal_statistics.json")
    cfg.env.adapter.pretrained_model_path = str(ROOT / "checkpoints/pi0/paligemma-3b-pt-224")
    model = PiZeroInference(cfg, use_ddp=False)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.freeze_all_weights(); model.to(torch.float32).to(torch.device("cuda:0")).eval()
    adapter = hydra.utils.instantiate(cfg.env.adapter)
    env, env_id = make_env(args.task)

    seeds = []
    for item in args.seeds.split(","):
        if "-" in item:
            start, stop = map(int, item.split("-", 1)); seeds.extend(range(start, stop + 1))
        elif item.strip(): seeds.append(int(item))
    seeds = sorted(set(seeds))
    artifact = args.artifact.resolve()
    guided_arm = f"l{args.layer}_matched_pcd"
    protocol_id = f"PI0_FRACTAL_L{args.layer}_MATCHED_PCD_SIMPLER_GOOGLE_V2"
    config = {
        "protocol_id": protocol_id,
        "sampler_implementation": "official_euler_dynamic_action_positions_paired_replan_seed_v2",
        "tasks": list(TASKS), "seeds_by_task": {task: list(range(100)) for task in TASKS},
        "arms": ["vanilla", guided_arm],
        "checkpoint": str(CHECKPOINT), "renderer": {"shader": "rt", "depth": renderer_depth(args.task), "spp": 32, "denoiser": True},
        "prompt_attention_layer_0based": args.layer, "pcd_alpha": PCD_ALPHA,
        "candidate_count": args.candidates, "action_guidance": "KDE candidate reweighting; no velocity-field mixing",
        "policy_action_steps": int(cfg.act_steps), "flow_steps": int(cfg.num_inference_steps),
    }
    artifact.mkdir(parents=True, exist_ok=True)
    lock = artifact / "CONFIG_LOCK.json"
    if lock.exists():
        prior = json.loads(lock.read_text())
        for key in ("protocol_id", "sampler_implementation", "checkpoint", "renderer", "prompt_attention_layer_0based", "pcd_alpha", "candidate_count"):
            if prior.get(key) != config[key]:
                raise RuntimeError(f"configuration lock mismatch on {key}")
    else:
        lock.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event":"model_ready","gpu":args.gpu,"worker":args.worker_id,"task":args.task,"seeds":len(seeds),"candidates":args.candidates,"layer":args.layer,"guided_arm":guided_arm}), flush=True)
    outdir = artifact / "episodes" / args.task
    snapshot_dir = artifact / "snapshots" / args.task
    try:
        for seed in seeds:
            snapshot = capture_snapshot(env, seed)
            canonical = snapshot_sha(snapshot)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            (snapshot_dir / f"seed_{seed:03d}.pkl").write_bytes(pickle.dumps(snapshot))
            pair_hashes = []
            for arm in ("vanilla", guided_arm):
                summary_path = outdir / arm / f"episode_{seed:03d}_summary.json"
                if summary_path.exists():
                    saved = json.loads(summary_path.read_text())
                    pair_hashes.append((saved["initial_state_sha256"], saved["initial_rgb_sha256"]))
                    continue
                obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot, get_image_from_maniskill2_obs_dict)
                pair_hashes.append((state_hash, rgb_hash))
                instruction = env.unwrapped.get_language_instruction()
                adapter.reset()
                pcd = FractalPi0PCD(model, adapter, candidates=args.candidates, layer=args.layer)
                control_steps = 0; infos = []; trace = []; terminated = truncated = False
                started = time.monotonic(); control_replan = 0
                while not (terminated or truncated):
                    action_seed = seed * 1009 + control_replan
                    torch.manual_seed(action_seed); torch.cuda.manual_seed_all(action_seed)
                    if arm == "vanilla":
                        inputs = adapter.preprocess(env, obs, instruction)
                        inputs = {k:(v.to("cuda:0") if torch.is_tensor(v) else v) for k,v in inputs.items()}
                        causal, vlm_pos, prop_pos, action_pos = model.build_causal_mask_and_position_ids(inputs["attention_mask"], dtype=torch.float32)
                        causal = causal.to("cuda:0"); vlm_pos = vlm_pos.to("cuda:0")
                        prop_pos = prop_pos.to("cuda:0"); action_pos = action_pos.to("cuda:0")
                        vlm_prop, action_mask = model.split_full_mask_into_submasks(causal)
                        inputs = {"input_ids":inputs["input_ids"], "pixel_values":inputs["pixel_values"], "image_text_proprio_mask":vlm_prop, "action_mask":action_mask,
                                  "vlm_position_ids":vlm_pos, "proprio_position_ids":prop_pos, "action_position_ids":action_pos, "proprios":inputs["proprios"]}
                        with torch.inference_mode(): actions = model(**inputs)
                        env_actions = adapter.postprocess(actions[0].float().cpu().numpy())
                    else:
                        env_actions, step_trace = pcd.predict(env, obs, instruction, action_seed)
                        trace.append(step_trace)
                    for env_action in env_actions[: int(cfg.act_steps)]:
                        obs, reward, terminated, truncated, info = env.step(env_action)
                        control_steps += 1
                        success = bool(info.get("success", terminated)) if isinstance(info, dict) else bool(terminated)
                        infos.append({"success":success,"reward":float(reward),"terminated":bool(terminated),"truncated":bool(truncated),"info":str(info)})
                        if terminated and not env.unwrapped.is_final_subtask():
                            terminated = False
                            env.advance_to_next_subtask()
                        instruction = env.unwrapped.get_language_instruction()
                        if truncated or terminated or (args.max_steps and control_steps >= args.max_steps): break
                    control_replan += 1
                    if args.max_steps and control_steps >= args.max_steps: break
                result = {"success":bool(infos[-1]["success"]) if infos else False,
                          "final_success":bool(infos[-1]["success"]) if infos else False,
                          "first_success":next((i for i,x in enumerate(infos) if x["success"]),None),
                          "steps":control_steps,"replans":control_replan}
                row = {"protocol_id":config["protocol_id"],"task":args.task,"environment_id":env_id,"seed":seed,"arm":arm,
                       "success":result["success"],"result":result,"control_steps":control_steps,"runtime_seconds":time.monotonic()-started,
                       "canonical_snapshot_sha256":canonical,"initial_state_sha256":state_hash,"initial_rgb_sha256":rgb_hash,
                       "renderer":config["renderer"],"checkpoint":str(CHECKPOINT),"gpu":args.gpu,"attention_layer":args.layer,"attention_trace":trace}
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = summary_path.with_suffix(f".json.{os.getpid()}.tmp"); tmp.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n"); os.replace(tmp,summary_path)
                print(json.dumps({"task":args.task,"seed":seed,"arm":arm,"success":row["success"],"steps":control_steps,"runtime_s":round(row["runtime_seconds"],1),"worker":args.worker_id}),flush=True)
            if len(set(pair_hashes)) != 1:
                raise RuntimeError(f"paired initialization mismatch for {args.task} seed {seed}: {pair_hashes}")
    finally:
        env.close()
    print(json.dumps({"event":"worker_complete","worker":args.worker_id,"gpu":args.gpu}),flush=True)


if __name__ == "__main__":
    main()

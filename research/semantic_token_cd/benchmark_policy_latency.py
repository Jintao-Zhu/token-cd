#!/usr/bin/env python3
"""Paired fixed-observation inference-cost benchmark for OpenVLA/SpatialVLA.

No environment actions/episodes are run. The script captures one canonical
RGB observation, then times vanilla and the locked full contrastive policy on
the exact same image. Reported frequency is policy-only 1/latency.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
TOKEN_CD = WORKSPACE / "token-cd"
PCD_SOURCE = WORKSPACE / "pcd_openvla_simpler_box_31b027e"
SPATIAL_REPO = WORKSPACE / "SpatialVLA-SimplerEnv"
OVERLAY = WORKSPACE / "spatialvla-runtime-overlay-20260928"


def capture_image(task: str, seed: int, gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path[:0] = [str(TOKEN_CD / "task1/shim_site"), str(TOKEN_CD), str(PCD_SOURCE / "source")]
    from research.semantic_token_cd import l11_table2_openvla_rollout as runner  # lock RT map
    from research.semantic_token_cd.prompt_attn_l11_budget_causal_rollout import make_environment
    from research.semantic_token_cd.distractor_rollout import restore_snapshot
    env, env_id = make_environment(task)
    snapshot = pickle.loads((TOKEN_CD / "artifacts/vanilla_recon_shr_canonical_0_299_v2" /
                             "snapshots" / task / f"seed_{seed:03d}.pkl").read_bytes())
    try:
        obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
        image = obs["image"]["overhead_camera"]["rgb"]
        instruction = env.unwrapped.get_language_instruction()
    finally:
        env.close()
    out = TOKEN_CD / "artifacts/paper_readiness_audit_v1/latency"
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(out / "canonical_observation.npz", image=image,
                        instruction=np.asarray(instruction), state_hash=np.asarray(state_hash),
                        rgb_hash=np.asarray(rgb_hash), env_id=np.asarray(env_id), task=np.asarray(task))
    print(json.dumps({"captured": True, "task": task, "seed": seed, "shape": image.shape,
                      "state_hash": state_hash, "rgb_hash": rgb_hash,
                      "output": str(out / "canonical_observation.npz")}), flush=True)


def _measure(fn, torch, count: int, warmup: int):
    values = []
    for i in range(warmup + count):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            values.append(dt)
    return {"n": len(values), "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p90_ms": float(__import__("numpy").percentile(values, 90)),
            "p95_ms": float(__import__("numpy").percentile(values, 95)),
            "policy_only_hz_from_mean": 1000 / statistics.fmean(values)}


def bench_openvla(args, image, instruction):
    sys.path[:0] = [str(TOKEN_CD / "task1/shim_site"), str(TOKEN_CD), str(PCD_SOURCE / "source")]
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.l11_table2_openvla_rollout import build_policies
    cfg = get_policy_config("openvla", str(PCD_SOURCE / "source/pretrained/openvla-7b"),
                            "google_robot_pick_coke_can", {}, False)
    base = OpenVLAInference(**cfg)
    guided = build_policies(base, "google_robot_pick_coke_can", "6", ("full",))["full"]
    def run(policy):
        policy.reset(instruction)
        return policy.step(image, task_description=instruction)
    results = {}
    for label, policy in (("vanilla", base), ("l11_matched_full", guided)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        result = _measure(lambda: run(policy), torch, args.steps, args.warmup)
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
        results[label] = result
    results["forward_branches_per_replan"] = {"vanilla": 1, "l11_matched_full": 2}
    return results


def bench_spatialvla(args, image, instruction):
    # Use the exact L10+Matched+harmonic+contrastive cached decoder from the
    # archived rollout, with no action sent to the simulator.
    sys.path[:0] = [str(OVERLAY), str(SPATIAL_REPO), str(SPATIAL_REPO / "ManiSkill2_real2sim"),
                    str(TOKEN_CD / "task1/shim_site"), str(TOKEN_CD), str(PCD_SOURCE / "source")]
    from research.semantic_token_cd import spatialvla_l10_matched_remaining_rollout as sr
    sr.bootstrap()
    import numpy as np
    import torch
    from transformers.cache_utils import HybridCache
    from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
    from research.semantic_token_cd.rollout_policy import extract_entities
    from research.semantic_token_cd.st_shr_policy import harmonic_reconstruct
    from sklearn.cluster import KMeans
    torch.set_num_threads(1)
    torch.manual_seed(0); np.random.seed(0)
    sr.patch_runtime(0)
    model_dir = str(sr.MODEL_DIR)
    base = SpatialVLAInference(saved_model_path=model_dir, policy_setup="google_robot",
                               action_ensemble_temp=-0.8)
    model, proc = base.vla, base.processor
    model.eval()
    def make_inputs():
        base.reset(instruction)
        resized = base._resize_image(image)
        base._add_image_to_history(resized)
        raw = proc(images=base._obtain_image_history(), text=instruction,
                   unnorm_key=base.unnorm_key, return_tensors="pt", do_normalize=False)
        return raw.to(torch.bfloat16).to(model.device)
    def new_cache(length):
        return HybridCache(config=model.config.text_config, batch_size=1,
                           max_cache_len=length + 256, device=model.device, dtype=model.dtype)
    def prefill(inputs, output_attentions=False, features=None):
        original = model.get_image_features
        if features is not None:
            model.get_image_features = lambda *_a, **_kw: features
        try:
            with torch.inference_mode():
                return model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                             pixel_values=inputs.get("pixel_values"), intrinsic=inputs.get("intrinsic"),
                             token_type_ids=inputs.get("token_type_ids"), past_key_values=new_cache(inputs["input_ids"].shape[1]),
                             use_cache=True, output_attentions=output_attentions, return_dict=True)
        finally:
            model.get_image_features = original
    def get_negative(inputs, pos):
        feats = pos.image_hidden_states[0].detach().float()
        ids = inputs["input_ids"][0]
        visual_pos = torch.nonzero(ids == int(model.config.image_token_index), as_tuple=False).flatten()
        tok = proc.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        seq = ids.detach().cpu().tolist(); start = None
        for i in range(max(0, len(seq)-len(tok)+1)):
            if seq[i:i+len(tok)] == tok: start=i; break
        if start is None: raise RuntimeError("instruction tokens not found")
        qpos = list(range(start, start+len(tok)))
        score = pos.attentions[sr.ATTENTION_LAYER][0, :, qpos, :][:, :, visual_pos].float().mean((0,1)).cpu().numpy()
        if score.shape != (256,): raise RuntimeError(f"visual token count {score.shape}")
        hd = feats.detach().cpu().numpy().astype(np.float64)
        labels = KMeans(n_clusters=8, random_state=0, n_init=10).fit(hd).labels_
        vectors = np.stack([hd[labels == k].mean(0) for k in range(8)])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        groups=[]
        for phrase in extract_entities(instruction):
            t=proc.tokenizer(phrase,add_special_tokens=False)["input_ids"]
            if not t: continue
            emb=model.language_model.model.embed_tokens(torch.tensor([t],device=model.device))[0].float().mean(0).detach().cpu().numpy()
            emb/=np.linalg.norm(emb)+1e-8
            g=int(np.argmax(vectors@emb))
            if g not in groups: groups.append(g)
        selected=sorted(int(i) for g in groups for i in np.flatnonzero(labels==g))
        selected=sorted(np.lexsort((np.arange(256),-score))[:len(selected)].tolist())
        neg=feats.detach().float().cpu().numpy().copy()
        neg[selected]=harmonic_reconstruct(neg,selected,beta=0.0)
        return torch.as_tensor(neg,device=feats.device,dtype=feats.dtype)[None]
    def guided_call():
        inputs=make_inputs()
        pos=prefill(inputs,output_attentions=True)
        negative=get_negative(inputs,pos)
        neg=prefill(inputs,features=negative)
        pos_cache,neg_cache=pos.past_key_values,neg.past_key_values
        pos_scores=pos.logits[:,-1,:].float(); neg_scores=neg.logits[:,-1,:].float()
        ids=inputs["input_ids"]; mask=inputs.get("attention_mask",torch.ones_like(ids))
        eos=int(proc.tokenizer.eos_token_id); generated=[]
        with torch.inference_mode():
            for _ in range(256):
                token=(1.5*pos_scores-0.5*neg_scores).argmax(-1,keepdim=True)
                generated.append(token); ids=torch.cat((ids,token),1)
                if int(token.item())==eos: break
                mask=torch.cat((mask,torch.ones_like(token)),1)
                pos=model(input_ids=token,attention_mask=mask,past_key_values=pos_cache,use_cache=True,return_dict=True)
                neg=model(input_ids=token,attention_mask=mask,past_key_values=neg_cache,use_cache=True,return_dict=True)
                pos_cache,neg_cache=pos.past_key_values,neg.past_key_values
                pos_scores=pos.logits[:,-1,:].float(); neg_scores=neg.logits[:,-1,:].float()
        outputs=torch.cat(generated,1)
        return proc.decode_actions(generation_outputs=outputs,unnorm_key=base.unnorm_key)
    def vanilla_call():
        base.reset(instruction)
        return base.step(image,task_description=instruction)
    # Validate the exact generated path and warm up with an identical input.
    results={}
    for label,fn in (("vanilla",vanilla_call),("l10_matched_full",guided_call)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        result=_measure(fn,torch,args.steps,args.warmup)
        result["peak_allocated_gib"]=torch.cuda.max_memory_allocated()/(1024**3)
        result["peak_reserved_gib"]=torch.cuda.max_memory_reserved()/(1024**3)
        results[label]=result
    results["forward_branches_per_replan"]={"vanilla":1,"l10_matched_full":"positive+negative prefill and paired autoregressive token passes"}
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=("capture","openvla","spatialvla"),required=True)
    ap.add_argument("--gpu",type=int,required=True)
    ap.add_argument("--steps",type=int,default=500)
    ap.add_argument("--warmup",type=int,default=50)
    ap.add_argument("--image",type=Path,default=TOKEN_CD/"artifacts/paper_readiness_audit_v1/latency/canonical_observation.npz")
    args=ap.parse_args()
    if args.model=="capture":
        capture_image("google_robot_pick_coke_can",100,args.gpu); return
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    import numpy as np, torch
    d=np.load(args.image,allow_pickle=False)
    image=np.asarray(d["image"],dtype=np.uint8); instruction=str(d["instruction"].item())
    result=(bench_openvla(args,image,instruction) if args.model=="openvla" else bench_spatialvla(args,image,instruction))
    out=args.image.parent/f"{args.model}_latency_gpu{args.gpu}.json"
    payload={"model":args.model,"task":"google_robot_pick_coke_can","instruction":instruction,
             "image_shape":image.shape,"image_state_hash":str(d["state_hash"].item()),
             "image_rgb_hash":str(d["rgb_hash"].item()),"warmup":args.warmup,"measured_steps_per_arm":args.steps,
             "timing_scope":"policy inference, preprocessing, counterfactual branch and action decoding; no simulator step/render",
             "results":result}
    out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"output":str(out),"results":result},indent=2),flush=True)

if __name__=="__main__": main()

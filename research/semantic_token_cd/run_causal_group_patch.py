#!/usr/bin/env python3
"""Canonical L11-Matched fork-state spatial-group patch study.

Each worker owns a fixed modulo shard of the frozen episode manifest. The only
selector used here is PromptAttentionSHRInference.step(selection_budget_source=
"matched"); this module only observes its selected mask and applies feature
patches after canonical reconstruction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
HIST = REPO / "artifacts/prompt_attn_l11_token_count_v1"
VAN = REPO / "artifacts/vanilla_recon_shr_canonical_0_299_v2"


def sha(x):
    a = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def logsoftmax(x):
    x=np.asarray(x,dtype=np.float32)
    return x-np.logaddexp.reduce(x,axis=-1,keepdims=True)


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def spatial_groups(selected, max_size=6):
    selected = set(map(int, selected)); remaining = set(selected); comps = []
    while remaining:
        start = min(remaining); remaining.remove(start); stack=[start]; comp=[]
        while stack:
            i=stack.pop(); comp.append(i); r,c=divmod(i,16)
            for j in (i-16 if r else -1, i+16 if r<15 else -1, i-1 if c else -1, i+1 if c<15 else -1):
                if j in remaining: remaining.remove(j); stack.append(j)
        comps.append(sorted(comp))
    def split(comp):
        if len(comp)<=max_size: return [comp]
        rc=np.array([divmod(i,16) for i in comp],dtype=int)
        spans=rc.max(0)-rc.min(0); axis=0 if spans[0]>=spans[1] else 1
        order=sorted(comp,key=lambda i:(divmod(i,16)[axis],divmod(i,16)[1-axis],i))
        vals=[divmod(i,16)[axis] for i in order]; cut=float(np.median(vals))
        left=[i for i in order if divmod(i,16)[axis] < cut]; right=[i for i in order if divmod(i,16)[axis] > cut]
        if not left or not right:
            order=sorted(comp,key=lambda i:(i//16,i%16)); mid=len(order)//2; left,right=order[:mid],order[mid:]
        return split(sorted(left))+split(sorted(right))
    return [g for comp in comps for g in split(comp)]


def main():
    global WORKER_CONFIG_HASH
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",type=int,choices=(1,2,3),required=True)
    p.add_argument("--worker-id",required=True)
    p.add_argument("--worker-index",type=int,required=True)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--shader-dir",choices=("ibl","rt"),default=None)
    a=p.parse_args()
    WORKER_CONFIG_HASH=hashlib.sha256(
        Path(__file__).read_bytes()+b"|canonical-l11matched|fork-patch|grid4-components|recursive-max6|lambda0.5|sampling20260923|v1|shader-dir="+str(a.shader_dir).encode()
    ).hexdigest()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu); os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
    for q in (REPO/"task1/shim_site",REPO,PCD_SOURCE):
        if str(q) not in sys.path: sys.path.insert(0,str(q))
    import torch
    from parallel_inference import get_image_from_maniskill2_obs_dict
    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    from research.semantic_token_cd.distractor_rollout import restore_snapshot
    from research.semantic_token_cd.spatial_grid_rollout import make_environment
    from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
    from research.semantic_token_cd.l11_causal_group_preflight import build_historical_policy
    from research.semantic_token_cd.global_merge_policy import projector_merge_intervention, guided_forward_scores
    import research.semantic_token_cd.prompt_attn_shr_policy as oldmod

    manifest=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
    jobs=[(i,r) for i,r in enumerate(manifest) if i%a.workers==a.worker_index]
    out=a.output.resolve(); failures=0
    current=None; env=None; clean=None; matched=None; base=None
    for jid,row in jobs:
        task=row["task"]; seed=int(row["seed"])
        dest=out/"states"/task/f"seed_{seed:03d}"
        result=dest/"patch.json"
        retry_path=out/"logs/technical_attempts"/task/f"seed_{seed:03d}.json"
        previous_config_hash="760949a58c481fa2e18ee94cec4610c2fa5148a99737d2632a61616aac402154"
        attempts=json.loads(retry_path.read_text()).get("attempts",0) if retry_path.exists() else 0
        if attempts>=3:
            atomic_json(out/"logs/technical_failed"/task/f"seed_{seed:03d}.json",{"task":task,"seed":seed,"status":"FAILED_TECHNICAL","attempts":attempts,"reason":"three canonical restart attempts exhausted"})
            print(json.dumps({"worker":a.worker_id,"task":task,"seed":seed,"status":"FAILED_TECHNICAL"}),flush=True)
            continue
        if result.exists():
            try:
                old=json.loads(result.read_text())
                if old.get("complete") is True and old.get("worker_config_hash") in {WORKER_CONFIG_HASH,previous_config_hash}:
                    print(json.dumps({"skip":row["task"],"seed":seed}),flush=True); continue
            except Exception: pass
        try:
            if current!=task:
                if env is not None: env.close()
                env,_=make_environment(task,shader_dir=a.shader_dir)
                cfg=get_policy_config("openvla",str(PCD_SOURCE/"pretrained/openvla-7b"),task,{},False)
                base=OpenVLAInference(**cfg); clean=copy.copy(base); clean.__class__=AuditedVanillaInference
                clean._episode_trace=[]; clean._episode_logits=[]
                matched=build_historical_policy(base,task); current=task
            is_rt_switched = a.shader_dir == "rt" and task in ("google_robot_pick_coke_can", "google_robot_move_near")
            vanilla_root = out/"baselines" if is_rt_switched else VAN
            matched_root = out/"baselines" if is_rt_switched else HIST
            snap_path=VAN/"snapshots"/task/f"seed_{seed:03d}.pkl"
            with snap_path.open("rb") as f: snap=pickle.load(f)
            va=np.load(vanilla_root/"episodes"/task/"vanilla"/f"episode_{seed:03d}_arrays.npz")
            ca=np.load(matched_root/"episodes"/task/"l11_matched"/f"episode_{seed:03d}_arrays.npz")
            target_step=int(row["first_action_divergence_step"])
            obs,state_hash,rgb_hash=restore_snapshot(env,seed,snap)
            instruction=env.unwrapped.get_language_instruction(); clean.reset(instruction,seed=seed); matched.reset(instruction,seed=seed)
            fork=None
            for step in range(target_step+1):
                image=get_image_from_maniskill2_obs_dict(env,obs)
                _r,ca_clean,_=clean.step(image,None,instruction,proprio=obs["agent"]["eef_pos"])
                _r,ca_c,meta=matched.step(image,None,instruction,proprio=obs["agent"]["eef_pos"])
                ca_clean = ca_clean[0] if isinstance(ca_clean,list) else ca_clean
                ca_c = ca_c[0] if isinstance(ca_c,list) else ca_c
                aclean=np.concatenate([np.asarray(ca_clean["world_vector"]),np.asarray(ca_clean["rot_axangle"]),np.asarray(ca_clean["gripper"])])
                ac=np.concatenate([np.asarray(ca_c["world_vector"]),np.asarray(ca_c["rot_axangle"]),np.asarray(ca_c["gripper"])])
                if step < target_step:
                    hclean=np.asarray(va["executed_actions"][step]); hc=np.asarray(ca["executed_actions"][step])
                    if np.max(np.abs(hclean[:6]-hc[:6]))>0.0: raise RuntimeError(f"historical prefix not shared at {step}")
                    if np.max(np.abs(aclean[:6]-hclean[:6]))>3e-4 or np.max(np.abs(ac[:6]-hc[:6]))>3e-4:
                        raise RuntimeError(f"live/historical prefix action mismatch at {step}")
                    # Execute the live canonical clean action in its original
                    # float64 form. Historical NPZ trajectories are float32
                    # summaries and rounding those back into the simulator can
                    # perturb later fork observations.
                    obs,_,_,truncated,_=env.step(aclean)
                    if truncated: raise RuntimeError("episode truncated before historical fork")
                    instruction=env.unwrapped.get_language_instruction(); continue
                if np.array_equal(aclean[:6],ac[:6]): raise RuntimeError("recorded first-divergence is not a live divergence")
                if np.max(np.abs(aclean[:6]-np.asarray(va["executed_actions"][step,:6])))>3e-4 or np.max(np.abs(ac[:6]-np.asarray(ca["executed_actions"][step,:6])))>3e-4:
                    raise RuntimeError(f"live fork actions differ from historical canonical actions: clean_max={float(np.max(np.abs(aclean[:6]-np.asarray(va['executed_actions'][step,:6])))):.8g}, c_max={float(np.max(np.abs(ac[:6]-np.asarray(ca['executed_actions'][step,:6])))):.8g}")
                fork={"obs":obs,"image":image,"instruction":instruction,"clean_action":aclean,"c_action":ac,"meta":meta}
            if fork is None: raise RuntimeError("fork state missing")
            captured={}
            original=oldmod.harmonic_reconstruct
            def cap_harmonic(features,region,*args,**kwargs):
                ans=original(features,region,*args,**kwargs); captured["vplus"]=np.asarray(features).copy(); captured["vminus"]=np.asarray(features).copy(); captured["vminus"][np.asarray(region,dtype=int)]=ans; return ans
            oldmod.harmonic_reconstruct=cap_harmonic
            try:
                matched._episode_trace=[]; matched._episode_logits=[]
                _r,_act,meta=matched.step(fork["image"],None,fork["instruction"],proprio=fork["obs"]["agent"]["eef_pos"])
            finally: oldmod.harmonic_reconstruct=original
            rec=matched._episode_logits[-1]; vp=captured["vplus"]; vm=captured["vminus"]
            selected=np.flatnonzero(rec["selected_mask"]>0).tolist()
            if selected!=[int(i) for i in meta["selected_token_ids"]]: raise RuntimeError("canonical selected mask mismatch")
            groups=spatial_groups(selected)
            inputs=matched.process_inputs(fork["image"],task_description=fork["instruction"])
            vplus=torch.from_numpy(vp).unsqueeze(0).to(device="cuda:0",dtype=torch.bfloat16)
            vminus=torch.from_numpy(vm).unsqueeze(0).to(device="cuda:0",dtype=torch.bfloat16)
            pids=torch.as_tensor(meta["positive_token_ids"],device="cuda:0")
            lplus=logsoftmax(rec["positive"])
            lminus=logsoftmax(rec["negative"])
            residual=(lplus-lminus)[:6].astype(np.float64); denom=float(np.sum(residual*residual))+1e-12
            group_rows=[]; restored={}; only={}
            for gi,g in enumerate(groups):
                patched=vm.copy(); patched[g]=vp[g]
                with projector_merge_intervention(matched.vla,torch.from_numpy(patched).unsqueeze(0)):
                    scores=guided_forward_scores(matched.vla,inputs,pids,256)
                logits=scores[:6,-256:].float().cpu().numpy(); lp=logsoftmax(logits)
                d=(lp-lminus[:6]).astype(np.float64)
                R=float(np.sum(d*residual)/denom); Q=float(np.linalg.norm(d)/(np.linalg.norm(residual)+1e-12))
                action=(lplus[:6]+0.5*(lplus[:6]-lp)).argmax(-1).astype(int)
                restored[gi]={"logits":logits,"R":R,"Q":Q,"action":action.tolist()}
                coords=[divmod(i,16) for i in g]
                norms=np.linalg.norm(vp[g]-vm[g],axis=1)
                attn=np.asarray(rec.get("prompt_attention",np.zeros(256)))
                group_rows.append({"group_id":gi,"token_ids":g,"size":len(g),"centroid":[float(np.mean([x[0] for x in coords])),float(np.mean([x[1] for x in coords]))],"bbox":[min(x[0] for x in coords),min(x[1] for x in coords),max(x[0] for x in coords),max(x[1] for x in coords)],"mean_l11_attention":float(np.mean(attn[g])) if attn.shape==(256,) else None,"mean_l11_rank":None,"mean_reconstruction_magnitude":float(np.mean(norms)),"max_reconstruction_magnitude":float(np.max(norms)),"R":R,"Q":Q,"patched_action_tokens":action.tolist()})
            top=sorted(range(len(groups)),key=lambda i:(-restored[i]["R"],i))[:2]
            for gi in top:
                g=groups[gi]
                if len(g)<=2: continue
                children=spatial_groups(g,max_size=2)
                for ci,child in enumerate(children):
                    patched=vm.copy(); patched[child]=vp[child]
                    with projector_merge_intervention(matched.vla,torch.from_numpy(patched).unsqueeze(0)):
                        scores=guided_forward_scores(matched.vla,inputs,pids,256)
                    lp=logsoftmax(scores[:6,-256:].float().cpu().numpy()); d=(lp-lminus[:6]).astype(np.float64)
                    restored[f"fine_{gi}_{ci}"]={"token_ids":child,"R":float(np.sum(d*residual)/denom),"Q":float(np.linalg.norm(d)/(np.linalg.norm(residual)+1e-12)),"action":(lplus[:6]+.5*(lplus[:6]-lp)).argmax(-1).astype(int).tolist()}
            for gi in top:
                g=groups[gi]; patched=vp.copy(); patched[g]=vm[g]
                with projector_merge_intervention(matched.vla,torch.from_numpy(patched).unsqueeze(0)):
                    scores=guided_forward_scores(matched.vla,inputs,pids,256)
                lp=logsoftmax(scores[:6,-256:].float().cpu().numpy()); d=(lp-lplus[:6]).astype(np.float64)
                only[gi]={"token_ids":g,"R_vs_original":float(np.sum(d*residual)/denom),"Q":float(np.linalg.norm(d)/(np.linalg.norm(residual)+1e-12))}
            # Save simulator state at the fork for the closed-loop stage.
            inner=env.unwrapped
            sim={"sim_state":np.asarray(inner.get_state()).copy(),"agent_state":copy.deepcopy(inner.agent.get_state()),"rng_state":copy.deepcopy(inner._episode_rng.get_state()),"instruction":fork["instruction"]}
            restored_json={str(k):{kk:vv for kk,vv in v.items() if kk!="logits"} for k,v in restored.items()}
            saved={"protocol_id":"L11_MATCHED_CAUSAL_GROUP_PATCH_V1","complete":True,"worker_id":a.worker_id,"worker_config_hash":WORKER_CONFIG_HASH,"task":task,"seed":seed,"category":row["category"],"fork_step":target_step,"first_action_divergence_step":target_step,"canonical_snapshot_sha256":row["canonical_snapshot_sha256"],"initial_state_sha256":state_hash,"fork_rgb_sha256":sha(fork["image"]),"matched_m":int(meta["m_t"]),"matched_cluster_ids":meta.get("matched_budget_group_ids"),"matched_entity_parse":meta.get("selected_entities"),"selected_token_ids":selected,"attention_sha256":meta.get("attention_sha256"),"harmonic_sha256":sha(vm),"vplus_sha256":sha(vp),"vminus_sha256":sha(vm),"positive_logits_sha256":sha(rec["positive"]),"negative_logits_sha256":sha(rec["negative"]),"clean_action":fork["clean_action"].tolist(),"c_action":fork["c_action"].tolist(),"groups":group_rows,"top_group_ids":top,"fine_and_sufficiency":restored_json,"only_group":{str(k):v for k,v in only.items()},"clean_action_tokens":meta["positive_token_ids"],"c_action_tokens":meta["final_token_ids"],"action_vocab_token_ids":list(range(int(matched.vla.vocab_size)-256,int(matched.vla.vocab_size)))}
            # Large arrays are kept separately for restartable patch/branch work.
            npz=dest/"fork_features.npz"; npz.parent.mkdir(parents=True,exist_ok=True)
            t=npz.with_name(npz.name+f".{os.getpid()}.tmp.npz")
            np.savez_compressed(t,rgb=fork["image"],vplus=vp,vminus=vm,positive=rec["positive"],negative=rec["negative"],selected_mask=rec["selected_mask"],prompt_attention=rec.get("prompt_attention",np.zeros(256,dtype=np.float32)))
            os.replace(t,npz)
            sp=dest/"fork_sim.pkl"; st=sp.with_name(sp.name+f".{os.getpid()}.tmp")
            with st.open("wb") as f: pickle.dump(sim,f,protocol=pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
            os.replace(st,sp); saved["fork_feature_file"]=str(npz); saved["fork_sim_file"]=str(sp)
            atomic_json(result,saved)
            print(json.dumps({"worker":a.worker_id,"task":task,"seed":seed,"category":row["category"],"fork":target_step,"groups":len(groups),"status":"DONE"}),flush=True)
        except Exception as e:
            failures+=1
            atomic_json(out/"logs"/f"failure_{a.worker_id}_{task}_{seed}_{int(time.time())}.json",{"task":task,"seed":seed,"error":repr(e),"traceback":traceback.format_exc(),"worker":a.worker_id})
            print(json.dumps({"worker":a.worker_id,"task":task,"seed":seed,"status":"FAILED","error":repr(e)}),flush=True)
            if any(s in repr(e).lower() for s in ("devicelost","vulkan","out of memory","outofmemory")):
                retry_path=out/"logs/technical_attempts"/task/f"seed_{seed:03d}.json"
                old=json.loads(retry_path.read_text()) if retry_path.exists() else {"task":task,"seed":seed,"attempts":0}
                old["attempts"]+=1; old["last_error"]=repr(e); old["worker_id"]=a.worker_id
                atomic_json(retry_path,old)
                # Restart this worker process from its frozen shard. Completed jobs
                # are hash-checked and skipped; this failed case starts at snapshot.
                os._exit(75)
    if env is not None: env.close()
    raise SystemExit(1 if failures else 0)


WORKER_CONFIG_HASH=""

if __name__=="__main__": main()

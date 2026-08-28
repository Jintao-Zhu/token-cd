"""Single-snapshot four-arm branch integrity dry-run; no environment continuation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_self_guidance.timestep_sampler import sample_timestep_shift_actions
from research.coreact_self_guidance.sampler import apply_self_guidance_velocity

ARMS=("Vanilla","N0","W05","W20")

def digest(x):
    h=hashlib.sha256()
    if isinstance(x,dict):
        for k in sorted(x): h.update(k.encode()); h.update(digest(x[k]).encode())
    elif isinstance(x,torch.Tensor):
        x=x.detach().cpu().contiguous(); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    elif isinstance(x,np.ndarray):
        x=np.ascontiguousarray(x); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    else: h.update(repr(x).encode())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--reference',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--language',required=True); p.add_argument('--milestone',type=int,default=60); args=p.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    ref=json.loads(args.reference.read_text()); actions=[np.asarray(x,dtype=np.float32) for x in ref['actions']][:args.milestone]; task_id=int(ref['task_id']); init_id=int(ref['init_state_id']); seed=int(ref['seed']); config,policy,pre,post=load_policy_and_processors(args.workspace.resolve()); results={}; traces={}; prepared_hashes={}; state_hashes={}; obs_hashes={}; noise_hashes={}; chunks={}
    means=[]
    try:
        for arm in ARMS:
            env,env_pre,env_post=make_task_env('libero_spatial',task_id,config)
            try:
                inner=env.envs[0]; inner.init_state_id=init_id; obs,_=env.reset(seed=seed)
                for action in actions: obs,_,_,_,_=env.step(action[None,:])
                batch=prepare(policy,pre,env_pre,obs,args.language); prepared_hashes[arm]=digest({k:v for k,v in batch.items() if torch.is_tensor(v)}); state_hashes[arm]=digest(np.asarray(inner._env.get_sim_state())); obs_hashes[arm]=digest(obs)
                gen=torch.Generator(device=batch['state'].device).manual_seed(seed*1000+args.milestone//10); noise=torch.randn((1,config.chunk_size,config.max_action_dim),generator=gen,device=batch['state'].device,dtype=batch['state'].dtype); noise_hashes[arm]=digest(noise)
                if arm=='Vanilla':
                    with torch.inference_mode(): chunk=policy.model.sample_actions(batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise=noise)
                    chunks[arm]=chunk.detach().float().cpu()
                else:
                    w={'N0':0.0,'W05':0.5,'W20':2.0}[arm]
                    with torch.inference_mode(): chunk,trace=sample_timestep_shift_actions(policy.model,batch['images'],batch['image_masks'],batch['lang_tokens'],batch['lang_masks'],batch['state'],noise,shift=0.2,w=w,pure_negative=arm=='N0',record_correction_vectors=True)
                    chunks[arm]=chunk.detach().float().cpu(); traces[arm]=trace
            finally: env.close()
        same_inputs=all(len(set(values.values()))==1 for values in (state_hashes,obs_hashes,prepared_hashes,noise_hashes))
        clean={a:traces[a]['step_traces'] for a in ('N0','W05','W20')}
        # Only flow step 0 is pre-intervention for every arm. Later x_tau/v_clean
        # values are expected to differ because the previous operator changed x.
        clean_equal=all(digest(clean[a][0]['_clean_velocity'])==digest(clean['N0'][0]['_clean_velocity']) for a in ('W05','W20'))
        shift_equal=all(digest(clean[a][0]['_shifted_velocity'])==digest(clean['N0'][0]['_shifted_velocity']) for a in ('W05','W20'))
        x_equal=all(digest(clean[a][0]['_x_tau'])==digest(clean['N0'][0]['_x_tau']) for a in ('W05','W20'))
        d_equal=all(digest(clean[a][i]['_correction_vector'])==digest((clean[a][i]['_clean_velocity']-clean[a][i]['_shifted_velocity'])) for a in ('N0','W05','W20') for i in range(10))
        formula_equal=True
        for arm,w in (("N0",0.0),("W05",0.5),("W20",2.0)):
            for s in clean[arm]:
                cv=s['_clean_velocity']; sv=s['_shifted_velocity']; expected,_,applied,_=apply_self_guidance_velocity(cv,sv,w=w,skipped=False,pure_negative=arm=='N0',action_dim=7,trust_region_kappa=0.25)
                formula_equal = formula_equal and torch.allclose(applied,s['_applied_correction_vector'],atol=1e-5,rtol=1e-5)
        finite=all(torch.isfinite(c).all() for c in chunks.values())
        # Independent vanilla continuation reproduction from the same replay point.
        venv, vpre, vpost = make_task_env('libero_spatial',task_id,config)
        try:
            vi=venv.envs[0]; vi.init_state_id=init_id; vo,_=venv.reset(seed=seed)
            for action in actions: vo,_,_,_,_=venv.step(action[None,:])
            vb=prepare(policy,pre,vpre,vo,args.language); vg=torch.Generator(device=vb['state'].device).manual_seed(seed*1000+args.milestone//10); vn=torch.randn((1,config.chunk_size,config.max_action_dim),generator=vg,device=vb['state'].device,dtype=vb['state'].dtype)
            with torch.inference_mode(): vc=policy.model.sample_actions(vb['images'],vb['image_masks'],vb['lang_tokens'],vb['lang_masks'],vb['state'],noise=vn)
            continuation_equal=digest(vc)==digest(chunks['Vanilla']) and digest(vc[:,:1,:7])==digest(chunks['Vanilla'][:,:1,:7])
        finally: venv.close()
        gate=all((same_inputs,clean_equal,shift_equal,x_equal,d_equal,formula_equal,finite,continuation_equal))
        results={"task_id":task_id,"init_state_id":init_id,"milestone":args.milestone,"same_branch_point_inputs":same_inputs,"clean_velocity_equal_step0":clean_equal,"shift_velocity_equal_step0":shift_equal,"x_tau_equal_step0":x_equal,"raw_delta_recomputed_equal":d_equal,"applied_formula_equal":formula_equal,"vanilla_continuation_equal":continuation_equal,"all_actions_finite":bool(finite),"noise_hashes":noise_hashes,"state_hashes":state_hashes,"observation_hashes":obs_hashes,"prepared_hashes":prepared_hashes,"first_divergence":"operator_only" if gate else "before_or_unresolved","decision":"CAUSAL_FOUR_ARM_DRYRUN_VALIDATED_READY_FOR_ROLLOUT" if gate else "CAUSAL_FOUR_ARM_DRYRUN_FAILED_NO_ROLLOUT"}
        (out/'geometry_sidecar.pt').write_bytes(torch.save({"traces":traces},_use_new_zipfile_serialization=True) if False else b'')
        torch.save({"traces":traces},out/'geometry_sidecar.pt')
    finally: pass
    (out/'dryrun.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n'); print(json.dumps(results,indent=2))

if __name__=='__main__': main()

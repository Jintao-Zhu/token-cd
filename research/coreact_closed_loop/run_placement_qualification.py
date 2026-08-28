#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare

def main():
    p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); a=p.parse_args()
    ws=a.workspace.resolve(); art=a.artifact.resolve(); art.mkdir(parents=True,exist_ok=True)
    cfg,policy,pre,post=load_policy_and_processors(ws)
    env,ep,_=make_task_env('libero_spatial',0,cfg)
    try:
        inner=env.envs[0]; inner.init_state_id=0; obs,_=env.reset(seed=42000001); prep=prepare(policy,pre,ep,obs,inner.task_description)
    finally: env.close()
    means=torch.load(ws/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu')
    g=torch.Generator(device=prep['state'].device).manual_seed(42000002)
    noise=torch.randn((1,cfg.chunk_size,cfg.max_action_dim),generator=g,dtype=prep['state'].dtype,device=prep['state'].device)
    args=(policy.model,prep['images'],prep['image_masks'],prep['lang_tokens'],prep['lang_masks'],prep['state'],noise,means['visual_position_mean'])
    vanilla=policy.model.sample_actions(prep['images'],prep['image_masks'],prep['lang_tokens'],prep['lang_masks'],prep['state'],noise=noise)
    arms={'full':GuidanceConfig(branch='acg'),'early':GuidanceConfig(branch='acg',flow_step_end=3),'mid':GuidanceConfig(branch='acg',flow_step_start=3,flow_step_end=7),'late':GuidanceConfig(branch='acg',flow_step_start=7),'near':GuidanceConfig(branch='acg',action_end=25),'far':GuidanceConfig(branch='acg',action_start=25)}
    outputs={}; traces={}
    for name,c in arms.items(): outputs[name],traces[name]=sample_coreact_actions(*args,config=c,selection_seed=42000003)
    checks={'model_config':{'num_steps':cfg.num_steps,'chunk_size':cfg.chunk_size},'lambda_zero_parity':float((sample_coreact_actions(*args,config=GuidanceConfig(branch='acg',guidance_scale=0),selection_seed=42000003)[0]-vanilla).abs().max()),'finite':all(bool(torch.isfinite(x).all()) for x in outputs.values()),'deterministic':True,'acg_changes_output':any(float((x-vanilla).abs().max())>1e-6 for x in outputs.values()),'placement_ranges':{k:{'flow':v['flow_step_range'],'action':v['action_range']} for k,v in traces.items()}}
    payload={'qualification':'guidance_placement_existence_gate','checks':checks,'output_hashes':{k:tensor_sha256(v) for k,v in outputs.items()},'trace_summary':{k:{'correction_norms':[x['applied_guidance_norm'] for x in v['step_traces']]} for k,v in traces.items()}}
    payload['pass']=checks['model_config']=={'num_steps':10,'chunk_size':50} and checks['lambda_zero_parity']<=1e-6 and checks['finite'] and checks['acg_changes_output']
    (art/'qualification.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2)); raise SystemExit(0 if payload['pass'] else 2)
if __name__=='__main__': main()

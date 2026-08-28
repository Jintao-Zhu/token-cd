#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
import torch
from research.coreact_closed_loop.guidance import GuidanceConfig,sample_coreact_actions
from research.coreact_closed_loop.runtime import load_policy_and_processors,make_task_env,prepare

SPECS={'full':{},'early':{'flow_step_end':3},'mid':{'flow_step_start':3,'flow_step_end':7},'late':{'flow_step_start':7},'near':{'action_end':25},'far':{'action_start':25}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();ws=a.workspace.resolve();art=a.artifact.resolve()
 cfg,policy,pre,_=load_policy_and_processors(ws);means=torch.load(ws/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt',weights_only=True,map_location='cpu'); sums={k:[] for k in SPECS}
 for task in range(10):
  env,ep,_=make_task_env('libero_spatial',task,cfg)
  try:
   inner=env.envs[0]
   for init in (30,31):
    inner.init_state_id=init;obs,_=env.reset(seed=900000000+task*1000+init);x=prepare(policy,pre,ep,obs,inner.task_description);g=torch.Generator(device=x['state'].device).manual_seed(910000000+task*1000+init);noise=torch.randn((1,50,cfg.max_action_dim),generator=g,device=x['state'].device,dtype=x['state'].dtype);args=(policy.model,x['images'],x['image_masks'],x['lang_tokens'],x['lang_masks'],x['state'],noise,means['visual_position_mean'])
    for name,kw in SPECS.items():
     _,tr=sample_coreact_actions(*args,config=GuidanceConfig(branch='acg',record_correction_vectors=True,**kw),selection_seed=1)
     e=sum(float((z['_applied_correction_vector']**2).sum()) for z in tr['step_traces']);sums[name].append(e)
  finally:env.close()
 means_e={k:sum(v)/len(v) for k,v in sums.items()};full=means_e['full'];cap=2.0;mult={'full':1.0};
 for k in SPECS:
  if k!='full': mult[k]=min(cap,math.sqrt(full/max(means_e[k],1e-12)))
 out={'calibration_states':[{'task_id':t,'init_state_ids':[30,31]} for t in range(10)],'outcome_blind':True,'energy_definition':'sum squared applied correction over flow steps, action positions, real action dims','unscaled_mean_energy':means_e,'multiplier_cap':cap,'locked_multipliers':mult,'expected_matched_energy':{k:means_e[k]*mult[k]**2 for k in SPECS}}
 (art/'calibration.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

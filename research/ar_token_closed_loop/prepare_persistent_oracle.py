from __future__ import annotations
import argparse,hashlib,json,platform,socket
from datetime import datetime
from pathlib import Path
import yaml
TASKS=range(10); INITS=range(50); ARMS=('V','A4','A8','A16','R4','R8','R16')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 for d in ('phase0','episodes','invalid_pairs','logs','status'):(out/d).mkdir(parents=True)
 ck=w/'checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873';mean=w/'artifacts/ar_token_counterfactual_qualification_v1_20260808_231044/position_conditioned_visual_mean.pt'
 protocol={'experiment_name':'ar_persistent_token_candidate_oracle_v1','created_at':datetime.now().astimezone().isoformat(),'stage':'phase0_integrity_pending_no_formal_rollout','tasks':list(TASKS),'official_init_states':list(INITS),'arms':list(ARMS),'planned_pairs':500,'planned_episodes':3500,'phase0':{'tasks':list(TASKS),'init_state_ids':[0,1],'matched_states':20,'outcome_blind':True},'persistent_policy':{'every_control_step_recompute_selector_and_mean_replace_then_greedy_free_run':True,'openvla_output':'7 AR action tokens = one 7D control action','actions_executed_per_replan':1,'maximum_control_steps':220,'settle_noop_steps':10},'selectors':{'A4':'attention top 4','A8':'attention top 8','A16':'attention top 16','R4':'deterministic random 4','R8':'deterministic random 8','R16':'deterministic random 16'},'random_seed':'sha256(task,init,control_step,arm)','operator':'v8 locked position-conditioned post-projector mean replacement','guidance_interpolation':False,'decoding':'greedy deterministic','oracle_metrics':['max(V,A4,A8,A16)','max(V,R4,R8,R16)','max(all arms)','vanilla-failure rescue','unique rescue','initial action-token diversity'],'checkpoint_revision':'962318cec55ac10993ff0f5f43eda9a270b4c873','checkpoint_config_sha256':sha(ck/'config.json'),'replacement_sha256':sha(mean),'environment':{'host':socket.gethostname(),'platform':platform.platform()}}
 (out/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False));rows=[]
 for t in TASKS:
  for i in INITS:
   pair=f'task{t:02d}__init{i:02d}'
   for arm in ARMS:rows.append({'pair_id':pair,'episode_id':f'{pair}__{arm}','task_id':t,'init_state_id':i,'arm':arm})
 (out/'episode_manifest.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));(out/'decision.json').write_text(json.dumps({'decision':'PHASE0_INTEGRITY_PENDING_NO_FORMAL_ROLLOUT','planned_episodes':3500},indent=2)+'\n');print(out)
if __name__=='__main__':main()

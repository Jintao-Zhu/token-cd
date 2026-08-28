#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, platform, socket
from pathlib import Path
import yaml
ARMS=('V_vanilla','W2_full','W2_early25','W2_start25','W2_early40','W2_start40')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();ws=a.workspace.resolve();out=a.output.resolve();parent=a.parent.resolve()
 if out.exists(): raise FileExistsError(out)
 for d in ('episodes','geometry','invalid_pairs','status','logs'): (out/d).mkdir(parents=True,exist_ok=True)
 pp=yaml.safe_load((parent/'protocol.lock.yaml').read_text()); ck=ws/'task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de'
 protocol={'experiment_name':'coreact_task4_early_trajectory_accumulation_v1','stage':'development_adjudication_not_confirmation','question':'is the prior Task 4 full-W2 gain reproducible and caused by cumulative early trajectory shaping?','task':{'suite':'libero_spatial','id':4,'instruction':'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate'},'init_state_ids':list(range(50)),'arms':list(ARMS),'planned_pairs':50,'planned_episodes':300,'fresh_seed_replication':True,'reset_seed_rule':'130400000 + init_state_id','noise_seed_rule':'202608110000 + 1000*init_state_id + replan_index','shared':{'same_reset_initial_observation_preprocessing_noise_per_pair':True,'flow_steps':10,'chunk_size':50,'executed_actions_per_chunk':10,'maximum_control_steps':280,'shift':0.2,'trust_region_kappa':0.25,'W2':2.0,'real_action_dim':7},'schedule':{'reference_length':'L from the paired fresh-seed vanilla trajectory','cutoff25':'max(10, min(270, 10*floor((0.25*L)/10)))','cutoff40':'max(10, min(270, 10*floor((0.40*L)/10)))','boundary':'operator chosen at replan start; cutoffs are exact 10-step replan boundaries','V_vanilla':'V throughout','W2_full':'W2 throughout','W2_early25':'W2 before cutoff25, then V','W2_start25':'V before cutoff25, then W2','W2_early40':'W2 before cutoff40, then V','W2_start40':'V before cutoff40, then W2'},'analysis':{'no_outcome_summary_before_300':True,'paired_bootstrap_unit':'init_state_id','bootstrap_replicates':2000,'exact_mcnemar':True,'selector_training':False,'primary_adjudication':'W2_full versus vanilla; early arms versus start arms'},'prior_result_for_context_only':{'artifact':'artifacts/coreact_timestep_self_guidance_closed_loop_v1_20260809_194030','task4_vanilla':0.54,'task4_W2_full':0.66,'difference':0.12,'not_used_to_change_parameters':True},'checkpoint':{'repo':'lerobot/smolvla_libero','revision':'31d453f7edd78c839a8bbc39744a292686daf0de','config_sha256':sha(ck/'config.json'),'weights_sha256':sha(ck/'model.safetensors')},'parent_temporal_artifact':str(parent),'parent_protocol_sha256':sha(parent/'protocol.lock.yaml'),'environment':{'host':socket.gethostname(),'platform':platform.platform()}}
 (out/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
 rows=[]
 for init in range(50):
  for arm in ARMS: rows.append({'pair_id':f'task04__init{init:02d}','episode_id':f'task04__init{init:02d}__{arm}','task_id':4,'init_state_id':init,'reset_seed':130400000+init,'noise_seed_base':202608110000+1000*init,'arm':arm})
 (out/'episode_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)); (out/'decision.json').write_text(json.dumps({'decision':'PROTOCOL_LOCKED_NO_ROLLOUT','planned_pairs':50,'planned_episodes':300},indent=2)+'\n'); print(out)
if __name__=='__main__':main()

from __future__ import annotations
import argparse, hashlib, json, platform, socket
from datetime import datetime
from pathlib import Path
import yaml

TASKS=(0,1,2,3,5,6,7,8,9); ARMS=('vanilla','random8_common','attention8_common'); CAP=3.0661711077317473
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--phase0-artifact',type=Path,required=True); a=p.parse_args(); w=a.workspace.resolve(); out=a.output.resolve(); phase=a.phase0_artifact.resolve()
 if out.exists(): raise FileExistsError(out)
 for d in ('episodes','invalid_pairs','status','logs'):(out/d).mkdir(parents=True)
 ck=w/'task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de'; means=w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt'
 protocol={'experiment_name':'coreact_common_min_selector_heldout_confirmation_v1','created_at':datetime.now().astimezone().isoformat(),'stage':'heldout_confirmation_development','benchmark':'LIBERO','suite':'libero_spatial','heldout_tasks':list(TASKS),'development_task_excluded':4,'init_state_ids':list(range(50)),'arms':list(ARMS),'planned_pairs':len(TASKS)*50,'planned_episodes':len(TASKS)*50*len(ARMS),'operator':{'definition':'at each flow step jointly evaluate Random8, Attention8, and locked Instability8 reference; M_common=min(M_R,M_A,M_I,M_cap); apply alpha_s=M_common/M_s','magnitude_cap':CAP,'trust_region_kappa':.25,'action_dimensions':7,'extrapolation_forbidden':True,'group_count':8,'replacement':'v8 position-conditioned visual mean','toward_scale':1.0,'dynamic_selection_each_replan':True},'shared':{'frozen_eval_checkpoint':True,'independent_reset_per_arm':True,'canonical_first_observation':'vanilla reset observation copied to all arms for first replan','flow_steps':10,'chunk_size':50,'execute_per_chunk':10,'max_control_steps':280},'phase0_artifact':str(phase),'phase0_protocol_sha256':sha(phase/'protocol.lock.yaml'),'seed_rules':{'reset':'170000000 + task*1000 + init','noise_base':'202609000000 + task*100000 + init*1000; add replan','selection':'20270900 + task*10000 + init*100 + replan'},'analysis':{'task_cluster_bootstrap':True,'bootstrap_replicates':10000,'exact_mcnemar':True,'no_selector_training':True,'no_summary_before_complete':True},'checkpoint':{'revision':'31d453f7edd78c839a8bbc39744a292686daf0de','config_sha256':sha(ck/'config.json'),'weights_sha256':sha(ck/'model.safetensors')},'calibration_mean_sha256':sha(means),'code_sha256':sha(w/'research/coreact_dynamic_core/dynamic_guidance.py'),'environment':{'host':socket.gethostname(),'platform':platform.platform()}}
 (out/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False)); rows=[]
 for task in TASKS:
  for init in range(50):
   pair=f'task{task:02d}__init{init:02d}'
   for arm in ARMS: rows.append({'pair_id':pair,'episode_id':f'{pair}__{arm}','task_id':task,'init_state_id':init,'arm':arm,'reset_seed':170000000+task*1000+init,'noise_seed_base':202609000000+task*100000+init*1000,'selection_seed_base':20270900+task*10000+init*100})
 (out/'episode_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)); (out/'decision.json').write_text(json.dumps({'decision':'PROTOCOL_LOCKED_NO_ROLLOUT','planned_pairs':len(TASKS)*50,'planned_episodes':len(rows)},indent=2)+'\n'); print(out)
if __name__=='__main__': main()

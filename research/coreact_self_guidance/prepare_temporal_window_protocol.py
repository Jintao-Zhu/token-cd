#!/usr/bin/env python3
"""Lock the preregistered temporal-window mapping artifact and manifest."""
from __future__ import annotations
import argparse, hashlib, json, platform, socket
from pathlib import Path
import yaml

TASKS=(0,4,8); INITS=tuple(range(5)); PROGRESS=(0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80)
ARMS=("V_vanilla","N0_shift_only","W20_extrapolation")
NOISE=(2026081001,2026081002,2026081003,2026081004,2026081005)
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--parent',type=Path,required=True); a=p.parse_args()
 ws=a.workspace.resolve(); out=a.output.resolve(); parent=a.parent.resolve()
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True); (out/'episodes').mkdir(); (out/'geometry').mkdir(); (out/'invalid_units').mkdir(); (out/'status').mkdir()
 parent_protocol=yaml.safe_load((parent/'protocol.lock.yaml').read_text()); lock=json.loads((parent/'rollout_code.lock.json').read_text())
 protocol={'experiment_name':'coreact_temporal_window_mapping_v1','stage':'development_mechanism_mapping','benchmark':'LIBERO','suite':'libero_spatial','tasks':list(TASKS),'negative_control_task':0,'init_state_ids':list(INITS),'progress_points':list(PROGRESS),'progress_rule':'actual vanilla trajectory length L; resolved step=max(1,floor(L*progress)); fixed before intervention outcomes','arms':list(ARMS),'matched_noise_seeds':list(NOISE),'locked_parameters':{'shift':0.2,'trust_region_kappa':0.25,'flow_steps':10,'chunk_size':50,'executed_actions_per_chunk':10,'maximum_control_steps':280,'real_action_dim':7},'replay':'each arm independently reset and exact vanilla action-prefix replay; branch-point equality required','primary_question':'is W2 benefit localized to a narrow temporal window?','statistics':{'episode_pair_unit':'snapshot x noise','snapshot_cluster_bootstrap':True,'bootstrap_replicates':2000,'no_selector_training':True,'no_parameter_changes_after_outcomes':True},'checkpoint':parent_protocol['checkpoint'],'calibration_mean_sha256':parent_protocol.get('calibration_mean_sha256'),'parent_artifact':str(parent),'parent_protocol_sha256':sha(parent/'protocol.lock.yaml'),'code_lock':lock.get('files',{}),'environment':{'host':socket.gethostname(),'platform':platform.platform()}}
 (out/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
 rows=[]
 for t in TASKS:
  for init in INITS:
   for pi,progress in enumerate(PROGRESS):
    sid=f'task{t:02d}__init{init:02d}__bin{pi:02d}'
    for ns in NOISE:
     for arm in ARMS: rows.append({'snapshot_id':sid,'task_id':t,'init_state_id':init,'progress':progress,'phase':'temporal_window','noise_seed':ns,'arm':arm,'episode_id':f'{sid}__noise{ns}__{arm}'})
 (out/'episode_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 (out/'decision.json').write_text(json.dumps({'decision':'PROTOCOL_LOCKED_NO_ROLLOUT','planned_snapshots':len(TASKS)*len(INITS)*len(PROGRESS),'planned_causal_units':len(TASKS)*len(INITS)*len(PROGRESS)*len(NOISE),'planned_episodes':len(rows),'arms':ARMS},indent=2)+'\n')
 print(json.dumps({'artifact':str(out),'snapshots':len(TASKS)*len(INITS)*len(PROGRESS),'causal_units':len(TASKS)*len(INITS)*len(PROGRESS)*len(NOISE),'episodes':len(rows)},indent=2))
if __name__=='__main__': main()

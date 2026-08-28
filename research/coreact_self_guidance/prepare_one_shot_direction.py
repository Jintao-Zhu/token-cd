from __future__ import annotations
import argparse, hashlib, json, platform, socket
from datetime import datetime
from pathlib import Path
import yaml

TASKS=tuple(range(10)); INIT_IDS=tuple(range(5)); PROGRESS_BINS=(0.25,0.65)
ARMS=("V_vanilla","T_attention8_one_shot","A_attention8_one_shot")
NOISE_SEEDS=(2026080901,2026080902,2026080903,2026080904,2026080905)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--reference-artifact',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 ws=a.workspace.resolve(); ref=a.reference_artifact.resolve(); out=a.output.resolve()
 if out.exists(): raise FileExistsError(out)
 if json.loads((ref/'decision.json').read_text())['decision'] != 'REFERENCE_SNAPSHOT_GATE_PASS_READY_FOR_MATCHED_CAUSAL_ROLLOUT': raise RuntimeError('reference gate not passed')
 out.mkdir(parents=True); (out/'episodes').mkdir(); (out/'geometry').mkdir(); (out/'invalid_units').mkdir(); (out/'status').mkdir(); (out/'logs').mkdir()
 protocol={'experiment_name':'coreact_one_shot_matched_direction_v1','created_at':datetime.now().astimezone().isoformat(),'stage':'development_one_shot_causal_direction','source_reference_artifact':str(ref),'source_reference_protocol_sha256':sha(ref/'reference_gate.lock.yaml'),'tasks':list(TASKS),'snapshots':100,'progress_bins':list(PROGRESS_BINS),'init_state_ids':list(INIT_IDS),'noise_seeds':list(NOISE_SEEDS),'arms':list(ARMS),'planned_causal_units':100*5,'planned_episodes':100*5*3,'one_shot':True,'treatment':'only first replan; all later replans vanilla','guidance':{'selector':'attention_top8','replacement':'visual_position_mean','direction':'opposite signs only','toward_formula':'clean - 0.5*clipped(clean-masked)','away_formula':'clean + 0.5*clipped(clean-masked)','group_count':8,'scale':0.5,'kappa':0.25,'flow_steps':10,'action_dim':7,'chunk_size':50,'execute_per_chunk':10,'max_control_steps':280},'statistics':{'state_cluster_bootstrap':True,'no_predictor':True,'no_outcome_summary_before_complete':True},'environment':{'host':socket.gethostname(),'platform':platform.platform()}}
 (out/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False),encoding='utf-8')
 rows=[]
 for t in TASKS:
  for init in INIT_IDS:
   for bi,progress in enumerate(PROGRESS_BINS):
    sid=f'task{t:02d}__init{init:02d}__bin{bi:02d}'
    for ns in NOISE_SEEDS:
     for arm in ARMS:
      rows.append({'snapshot_id':sid,'task_id':t,'init_state_id':init,'progress_bin':progress,'phase':'progress_bin_only','noise_seed':ns,'arm':arm,'episode_id':f'{sid}__noise{ns}__{arm}'})
 with (out/'episode_manifest.jsonl').open('x') as f:
  f.write(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 (out/'decision.json').write_text(json.dumps({'decision':'PROTOCOL_LOCKED_NO_ROLLOUT','planned_causal_units':500,'planned_episodes':1500},indent=2)+'\n')
 print(out)
if __name__=='__main__': main()

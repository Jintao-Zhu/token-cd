#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime
from pathlib import Path
import yaml
ARMS=('vanilla','random8_common','attention8_common','instability8_common')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--phase0',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();w=a.workspace.resolve();q=a.phase0.resolve();am=json.loads((q/'phase0_gate_amendment.json').read_text())
 if am['amended_decision']!='COMMON_MIN_PHASE0_PASS_READY_FOR_ROLLOUT' or not am['gate_pass']:raise RuntimeError('phase0 gate')
 o=(a.output or w/'artifacts'/f'coreact_common_min_selector_task4_rollout_v1_{datetime.now():%Y%m%d_%H%M%S}').resolve()
 if o.exists():raise FileExistsError(o)
 for d in ('episodes','invalid_pairs','status','logs'):(o/d).mkdir(parents=True,exist_ok=True)
 base=yaml.safe_load((q/'protocol.lock.yaml').read_text());base['experiment_name']='coreact_common_min_selector_task4_rollout_v1';base['phase0_artifact']=str(q);base['phase0_protocol_sha256']=sha(q/'protocol.lock.yaml');base['phase0_summary_sha256']=sha(q/'phase0_summary.json');base['phase0_gate_amendment_sha256']=sha(q/'phase0_gate_amendment.json');base['code_sha256']=sha(w/'research/coreact_dynamic_core/dynamic_guidance.py');base['runner_sha256']=sha(w/'research/coreact_dynamic_core/run_common_min_rollout.py')
 (o/'protocol.lock.yaml').write_text(yaml.safe_dump(base,sort_keys=False));rows=[]
 for init in range(50):
  for arm in ARMS:rows.append({'pair_id':f'task04__init{init:02d}','episode_id':f'task04__init{init:02d}__{arm}','task_id':4,'init_state_id':init,'arm':arm,'reset_seed':170004000+init,'noise_seed_base':202608640000+init*1000,'selection_seed_base':20270814+40000+init*100})
 (o/'episode_manifest.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));(o/'decision.json').write_text(json.dumps({'decision':'PROTOCOL_LOCKED_ROLLOUT_NOT_STARTED','planned_pairs':50,'planned_episodes':200},indent=2)+'\n');print(o)
if __name__=='__main__':main()

from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime
from pathlib import Path

ARMS=("lambda_0","lambda_001"); DOSES={"lambda_0":0.0,"lambda_001":0.01}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();o.mkdir(parents=True,exist_ok=False)
 for n in ('episodes','invalid_pairs','logs'): (o/n).mkdir()
 ck=w/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model'; lock=w/'artifacts/coreact_slg_self_weak_low_noise_v1_20260816_234129/selected_self_weak.lock.json'
 protocol={'experiment':'Exploratory W1 ultra-low-dose lambda=.01 test','created_at':datetime.now().astimezone().isoformat(),'exploratory_posthoc':True,'strong_checkpoint':str(ck),'weak':'W1 skip final action-expert block','window':[8,9],'lambda':DOSES,'trust_region_kappa':.25,'suite':'libero_spatial','tasks':list(range(10)),'init_states':'0..24 per task (same Selection states)','pairs':250,'episodes':500,'horizon':520,'chunk':50,'execute':10,'offline_lock_sha256':sha(lock),'no_confirmation_claim':True}
 (o/'protocol.lock.json').write_text(json.dumps(protocol,indent=2)+'\n')
 rows=[]
 for t in range(10):
  for i in range(25):
   pair=f'task{t:02d}__init{i:02d}'; reset=880_000_000+t*1000+i*10; noise=890_000_000+t*1000+i*10
   for arm in ARMS: rows.append({'episode_id':f'{pair}__{arm}','pair_id':pair,'arm':arm,'task_id':t,'init_state_id':i,'reset_seed':reset,'flow_noise_seed_base':noise,'lambda':DOSES[arm]})
 with (o/'episode_manifest.jsonl').open('x') as f: f.write('\n'.join(json.dumps(x,sort_keys=True) for x in rows)+'\n')
 print(str(o))
if __name__=='__main__':main()

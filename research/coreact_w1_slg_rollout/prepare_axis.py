from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime
from pathlib import Path

ARMS=("lambda_n010","lambda_n005","lambda_0","lambda_005","lambda_010")
DOSES={"lambda_n010":-.10,"lambda_n005":-.05,"lambda_0":0.,"lambda_005":.05,"lambda_010":.10}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();o.mkdir(parents=True,exist_ok=False)
 for n in ('episodes','invalid_pairs','logs'): (o/n).mkdir()
 ck=w/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model';lock=w/'artifacts/coreact_slg_self_weak_low_noise_v1_20260816_234129/selected_self_weak.lock.json'
 protocol={'experiment':'W1 Residual-Axis Closed-Loop Local-Optimum Test','created_at':datetime.now().astimezone().isoformat(),'scientific_role':'local curvature test, not lambda optimization','strong_checkpoint':str(ck),'weak':'W1 skip final action-expert block','axis':'vS + lambda*(vS-vW1)','arms':DOSES,'low_noise_steps':[8,9],'trust_region_kappa':.25,'scheduler':'10 steps tau 1.0 to .1','rollout':{'suite':'libero_spatial','tasks':list(range(10)),'init_state_ids':list(range(50)),'pairs':500,'episodes':2500,'horizon':520,'chunk':50,'execute':10},'seeds':{'reset':'900000000+task*1000+init*10','noise':'910000000+task*1000+init*10+replan'},'primary':'LocalPeak_005=Success(0)-mean(Success(-.05),Success(+.05))','offline_lock_sha256':sha(lock),'prohibited':['new weak','new window','task lambda','dynamic lambda','CFG','additional lambda']}
 (o/'protocol.lock.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[]
 for t in range(10):
  for i in range(50):
   pair=f'task{t:02d}__init{i:02d}';reset=900_000_000+t*1000+i*10;noise=910_000_000+t*1000+i*10
   for arm in ARMS:rows.append({'episode_id':f'{pair}__{arm}','pair_id':pair,'arm':arm,'task_id':t,'init_state_id':i,'reset_seed':reset,'flow_noise_seed_base':noise,'lambda':DOSES[arm]})
 with (o/'episode_manifest.jsonl').open('x') as f:f.write('\n'.join(json.dumps(x,sort_keys=True) for x in rows)+'\n')
 (o/'manifest.sha256').write_text(sha(o/'episode_manifest.jsonl')+'  episode_manifest.jsonl\n');(o/'status.json').write_text(json.dumps({'status':'PREPARED_DRY_RUN_PENDING','episodes_complete':0,'episodes_planned':2500},indent=2)+'\n');print(o)
if __name__=='__main__':main()

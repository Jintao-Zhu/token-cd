#!/usr/bin/env python3
import argparse,json,hashlib,datetime
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();art=a.artifact.resolve();cal=json.loads((art/'calibration.json').read_text());
 lock={'experiment':'guidance_placement_existence_gate_v1','suite':'libero_spatial','tasks':list(range(10)),'init_state_ids':list(range(20)),'arms':['vanilla','full','early','mid','late','near','far'],'episodes':1400,'flow_steps':10,'chunk_size':50,'executed_prefix':10,'branch':'ACG action-token self-attention diagonal-only','guidance_scale':0.5,'trust_region_kappa':0.25,'placement_multiplier_cap':2.0,'locked_multipliers':cal['locked_multipliers'],'calibration_artifact':'calibration.json','outcome_blind_calibration':True,'primary_endpoint':'closed_loop_success_rate','secondary_endpoints':['action_total_variation','chunk_discontinuity','correction_rms','latency'],'no_outcome_tuning':True}
 (art/'protocol.lock.json').write_text(json.dumps(lock,indent=2)+'\n'); rows=[]
 for t in range(10):
  for i in range(20):
   for arm in lock['arms']:rows.append({'episode_id':f'task{t:02d}__init{i:02d}__{arm}','task_id':t,'init_state_id':i,'arm':arm,'reset_seed':920000000+t*1000+i,'noise_seed_base':930000000+t*100000+i*1000})
 with (art/'episode_manifest.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
 print(json.dumps({'artifact':str(art),'episodes':len(rows),'created':datetime.datetime.now().isoformat()}))
if __name__=='__main__':main()

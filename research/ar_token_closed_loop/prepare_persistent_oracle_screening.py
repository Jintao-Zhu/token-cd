from __future__ import annotations
import argparse,hashlib,json,platform,socket
from datetime import datetime
from pathlib import Path
import yaml
TASKS=list(range(10));INITS=list(range(10));ARMS=['V','A4','A8','A16','R8']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--phase0-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();src=a.phase0_source.resolve();out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 for d in ('episodes','invalid_pairs','logs','status'):(out/d).mkdir(parents=True)
 protocol={'experiment_name':'ar_persistent_token_candidate_oracle_screening_v1','created_at':datetime.now().astimezone().isoformat(),'stage':'preregistered_cross_task_screening','tasks':TASKS,'init_state_ids':INITS,'arms':ARMS,'planned_pairs':100,'planned_episodes':500,'phase0_source_artifact':str(src),'phase0_protocol_sha256':sha(src/'protocol.lock.yaml'),'persistent_policy':{'every_control_step_recompute_selector_and_mean_replace':True,'openvla_7_action_tokens_equal_one_control_action':True,'max_control_steps':220,'settle_steps':10},'primary':{'attention_oracle':'max(V,A4,A8,A16)','uplift':'attention_oracle - V','vanilla_failure_rescue':'P(attention_oracle=1|V=0)'},'random_control':'R8','go':'clear oracle uplift around 8-10pp with rescues across multiple tasks','no_go':'oracle uplift around 1-2pp or rescues concentrated in Task4','no_outcome_reading_before_500':True,'checkpoint_revision':'962318cec55ac10993ff0f5f43eda9a270b4c873','environment':{'host':socket.gethostname(),'platform':platform.platform()}}
 (out/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False));rows=[]
 for t in TASKS:
  for i in INITS:
   pair=f'task{t:02d}__init{i:02d}'
   for arm in ARMS:rows.append({'pair_id':pair,'episode_id':f'{pair}__{arm}','task_id':t,'init_state_id':i,'arm':arm})
 (out/'episode_manifest.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));(out/'decision.json').write_text(json.dumps({'decision':'SCREENING_PROTOCOL_LOCKED_READY','planned_episodes':500},indent=2)+'\n');print(out)
if __name__=='__main__':main()

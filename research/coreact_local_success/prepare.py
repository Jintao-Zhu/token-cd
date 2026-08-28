from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();w,o=a.workspace.resolve(),a.artifact.resolve();o.mkdir(parents=True,exist_ok=False)
 for d in ('snapshots','audits','logs','u0','invalid_units','neighbors','neighbor_parts'): (o/d).mkdir()
 ck=w/'artifacts/coreact_trained_weak_local_finetune_v1_20260812_121522/training_run/trajectory/checkpoints/015000/pretrained_model';protocol={'experiment':'Local Success Direction / Action Utility Existence Test','created_at':datetime.now().astimezone().isoformat(),'phase':'SNAPSHOT_GATE_THEN_U0','strong_checkpoint':str(ck),'strong_model_sha256':sha(ck/'model.safetensors'),'snapshots':{'tasks':list(range(10)),'init_state_ids':list(range(40,45)),'progress':[.30,.65],'count':100,'selection':'no outcome filtering','horizon':520},'u0':{'continuation_seeds':5,'arms':['Strong','W1_plus','W1_minus','Nearest_manifold','Random_smooth'],'episodes':2500,'one_shot':True},'u1_conditional':True,'forbidden':['lambda tuning','W1 changes','window search','CFG','contact/uncertainty gate','success predictor']};(o/'protocol.lock.json').write_text(json.dumps(protocol,indent=2)+'\n');(o/'status.json').write_text(json.dumps({'status':'SNAPSHOT_CAPTURE_RUNNING','snapshots_complete':0,'snapshots_planned':100},indent=2)+'\n');print(o)
if __name__=='__main__':main()

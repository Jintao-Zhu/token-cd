from __future__ import annotations
import argparse, hashlib, json, platform
from datetime import datetime
from pathlib import Path
import torch, yaml
from libero.libero import benchmark
from research.coreact_closed_loop.prepare_pilot import sha256_file
from research.coreact_closed_loop.runtime import CHECKPOINT_REVISION, checkpoint_path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); a=p.parse_args(); w=a.workspace.resolve(); task=benchmark.get_benchmark_dict()['libero_spatial']().get_task(4); ck=checkpoint_path(w); art=w/'artifacts'/f"coreact_spatial_task4_direct_delete_every_replan_v1_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"; art.mkdir(parents=True,exist_ok=False); (art/'episodes').mkdir()
 protocol={'experiment':'coreact_spatial_task4_direct_delete_every_replan_v1','task_id':4,'suite':'libero_spatial','task':task.language,'init_state_ids':list(range(50)),'conditions':['vanilla','top8_mask_only','away','toward'],'episodes':200,'backbone':{'repo':'lerobot/smolvla_libero','revision':CHECKPOINT_REVISION,'frozen_eval':True},'flow_steps':10,'chunk_size':50,'executed_actions_per_chunk':10,'max_control_steps':280,'mask_frequency':'every replan','mask_operation':'direct sequence deletion of 8 eligible post-connector visual prefix tokens','selection':'native prefix late-half action-to-context attention top8, reranked each replan','protected':['language','state','special','padding'],'away':'clean + 0.5 * trust_region_clip(clean-deleted), kappa=0.25','toward':'clean - 0.5 * trust_region_clip(clean-deleted), kappa=0.25','same_state_noise_preprocessing':True,'bootstrap_replicates':2000,'development_replication':True}
 hashes={'checkpoint_config':sha256_file(ck/'config.json'),'calibration_mean':sha256_file(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt'),'code':{x:str(hashlib.sha256((w/x).read_bytes()).hexdigest()) for x in ('research/coreact_closed_loop/guidance.py','research/coreact_closed_loop/runtime.py','research/coreact_task4_replication/run.py','research/coreact_direct_delete.py','research/coreact_direct_delete_prepare.py')}}; protocol['hashes']=hashes
 (art/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False)); (art/'task_manifest.json').write_text(json.dumps({'suite':'libero_spatial','task_id':4,'language':task.language,'bddl_file':task.bddl_file},indent=2)+'\n')
 with (art/'episode_manifest.jsonl').open('x') as f:
  for i in range(50):
   base=104000000+4*100000+i*100
   for c in protocol['conditions']: f.write(json.dumps({'episode_id':f'direct_delete__task04__init{i:02d}__{c}','pair_id':f'direct_delete__task04__init{i:02d}','suite':'libero_spatial','task_id':4,'language':task.language,'init_state_id':i,'condition':c,'reset_seed':base+1,'action_noise_seed':base+2,'selection_seed':base+3},sort_keys=True)+'\n')
 (art/'environment.json').write_text(json.dumps({'host':platform.node(),'torch':torch.__version__,'cuda':torch.version.cuda,'checkpoint_revision':CHECKPOINT_REVISION},indent=2)+'\n'); print(art)
if __name__=='__main__': main()

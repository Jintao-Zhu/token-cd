from __future__ import annotations
import argparse, hashlib, json, platform
from datetime import datetime
from pathlib import Path
import torch, yaml
from libero.libero import benchmark
from research.coreact_closed_loop.prepare_pilot import sha256_file
from research.coreact_closed_loop.runtime import CHECKPOINT_REVISION, checkpoint_path

TASKS=(0,1,2,3,5); CONDITIONS=('vanilla','top8_mask_only','away','toward')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); a=p.parse_args(); w=a.workspace.resolve(); suite=benchmark.get_benchmark_dict()['libero_spatial'](); ck=checkpoint_path(w); art=w/'artifacts'/f"coreact_spatial_tasks_0_1_2_3_5_direct_delete8_every_replan_v1_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"; art.mkdir(parents=True,exist_ok=False); (art/'episodes').mkdir(); (art/'logs').mkdir()
 protocol={'experiment':'coreact_spatial_tasks_0_1_2_3_5_direct_delete8_every_replan_v1','suite':'libero_spatial','task_ids':list(TASKS),'init_state_ids':list(range(50)),'conditions':list(CONDITIONS),'episodes_per_task_condition':50,'episodes':1000,'group_count':8,'mask_frequency':'every replan','mask_operation':'direct deletion of 8 eligible post-connector visual prefix tokens','selection':'native prefix late-half action-to-context attention top8, reranked every replan','flow_steps':10,'chunk_size':50,'executed_actions_per_chunk':10,'max_control_steps':280,'away':'clean + 0.5 * trust_region_clip(clean-deleted), kappa=0.25','toward':'clean - 0.5 * trust_region_clip(clean-deleted), kappa=0.25','protected':['language','state','special','padding'],'same_noise_preprocessing_reset':True,'development_replication':True,'prohibited':['embedding replacement','partial outcome tuning']}
 protocol['hashes']={'checkpoint_config':sha256_file(ck/'config.json'),'calibration_mean':sha256_file(w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt'),'code':{x:hashlib.sha256((w/x).read_bytes()).hexdigest() for x in ('research/coreact_closed_loop/guidance.py','research/coreact_closed_loop/runtime.py','research/coreact_direct_delete.py','research/coreact_direct_delete_run.py','research/coreact_direct_delete_prepare_multi8.py')}}
 (art/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False)); (art/'environment.json').write_text(json.dumps({'host':platform.node(),'torch':torch.__version__,'cuda':torch.version.cuda,'checkpoint_revision':CHECKPOINT_REVISION},indent=2)+'\n')
 with (art/'task_manifest.json').open('w') as f: json.dump({'suite':'libero_spatial','tasks':{str(t):{'language':suite.get_task(t).language,'bddl_file':suite.get_task(t).bddl_file} for t in TASKS}},f,indent=2,sort_keys=True); f.write('\n')
 with (art/'episode_manifest.jsonl').open('x') as f:
  for task in TASKS:
   language=suite.get_task(task).language
   for init in range(50):
    base=500000000+task*100000+init*100
    for c in CONDITIONS: f.write(json.dumps({'episode_id':f'multi8__task{task:02d}__init{init:02d}__{c}','pair_id':f'multi8__task{task:02d}__init{init:02d}','suite':'libero_spatial','task_id':task,'language':language,'init_state_id':init,'condition':c,'reset_seed':base+1,'action_noise_seed':base+2,'selection_seed':base+3},sort_keys=True)+'\n')
 print(art)
if __name__=='__main__': main()

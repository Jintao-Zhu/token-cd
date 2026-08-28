from __future__ import annotations
import argparse, hashlib, json, platform, socket, sys
from datetime import datetime
from pathlib import Path
import yaml

TASKS=tuple(range(10)); INIT_IDS=tuple(range(5)); PROGRESS_BINS=(0.25,0.65); ARMS=("V_vanilla","N0_shift_only","W05_interpolation","W20_extrapolation"); NOISE_SEEDS=(2026080901,2026080902,2026080903,2026080904,2026080905)

def sha(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); ws=a.workspace.resolve(); out=a.output.resolve()
 if out.exists(): raise FileExistsError(out)
 sys.path.insert(0,str(ws/'LIBERO')); sys.path.insert(0,str(ws/'lerobot/src')); sys.path.insert(0,str(ws))
 (out/'snapshots').mkdir(parents=True); (out/'episodes').mkdir(); (out/'status').mkdir()
 ck=ws/'task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de'; mean=ws/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt'
 from libero.libero import benchmark
 suite=benchmark.get_benchmark_dict()['libero_spatial'](); tasks=[]
 for t in TASKS:
  task=suite.get_task(t); tasks.append({'task_id':t,'language':task.language,'bddl_file':task.bddl_file})
 protocol={'experiment_name':'coreact_state_level_matched_causal_calibration_v1','created_at':datetime.now().astimezone().isoformat(),'stage':'protocol_locked_before_rollout','benchmark':'LIBERO','suite':'libero_spatial','tasks':TASKS,'snapshot_rule':{'init_state_ids':INIT_IDS,'progress_bins':PROGRESS_BINS,'selection':'fixed trajectory-progress bins before any arm outcome','phase_labels':'progress_bin_only','control_horizon':280},'arms':ARMS,'locked_parameters':{'shift':0.2,'trust_region_kappa':0.25,'flow_steps':10,'chunk_size':50,'executed_actions_per_chunk':10,'action_dim':7},'matched_noise_seeds':NOISE_SEEDS,'planned_snapshots':len(TASKS)*len(INIT_IDS)*len(PROGRESS_BINS),'planned_rollouts':len(TASKS)*len(INIT_IDS)*len(PROGRESS_BINS)*len(NOISE_SEEDS)*len(ARMS),'outcome':'success in {0,1}; preserve per-seed D=W-V and snapshot mean utility','selector_features':'pre-action only: x_tau, v_clean, v_shift, d_raw, applied correction, norms, cosines, parallel/orthogonal ratios, temporal and action-dimension summaries','task_split':'leave-one-task-out; all seeds of a snapshot stay together','no_ref_top8':True,'checkpoint':{'repo':'lerobot/smolvla_libero','revision':'31d453f7edd78c839a8bbc39744a292686daf0de','config_sha256':sha(ck/'config.json'),'weights_sha256':sha(ck/'model.safetensors')},'calibration_mean_sha256':sha(mean),'environment':{'host':socket.gethostname(),'platform':platform.platform()},'formal_rollout_forbidden_until':'snapshot reference collection and branch integrity gates pass'}
 (out/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False),encoding='utf-8'); (out/'task_manifest.json').write_text(json.dumps(tasks,indent=2,sort_keys=True)+'\n')
 rows=[]
 for t in TASKS:
  for init in INIT_IDS:
   for bi,progress in enumerate(PROGRESS_BINS):
    sid=f'task{t:02d}__init{init:02d}__bin{bi:02d}'; base=120000000+t*100000+init*1000+bi*100
    for ns in NOISE_SEEDS:
     for arm in ARMS: rows.append({'snapshot_id':sid,'task_id':t,'init_state_id':init,'progress_bin':progress,'phase':'progress_bin_only','noise_seed':ns,'arm':arm,'reset_seed':base+1,'episode_id':f'{sid}__noise{ns}__{arm}'})
 with (out/'episode_manifest.jsonl').open('x') as f:
  for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
 (out/'decision.json').write_text(json.dumps({'decision':'PROTOCOL_LOCKED_NO_ROLLOUT','planned_snapshots':100,'planned_rollouts':2000},indent=2)+'\n')
 print(json.dumps({'artifact':str(out),'snapshots':100,'rollouts':2000},indent=2))

if __name__=='__main__': main()

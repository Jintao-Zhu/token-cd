#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math
from datetime import datetime
from pathlib import Path
import h5py,yaml
REPO='yifengzhu-hf/LIBERO-datasets';REVISION='f13aa24a3da8c43c7225569f28c562979fa0e35a';PHASES=(('early',.2),('middle',.5),('late',.8))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();w=a.workspace.resolve();o=(a.output or w/'artifacts'/f'coreact_expert_aligned_direction_audit_v1_{datetime.now():%Y%m%d_%H%M%S}').resolve()
 if o.exists():raise FileExistsError(o)
 for d in ('qualification','units','status','logs'):(o/d).mkdir(parents=True,exist_ok=True)
 import sys;sys.path.insert(0,str(w/'LIBERO'));from libero.libero import benchmark;suite=benchmark.get_benchmark_dict()['libero_spatial']();root=w/'LIBERO/libero/datasets/libero_spatial';sources=[];states=[];units=[]
 for task in range(10):
  language=suite.get_task(task).language;path=root/(language.replace(' ','_')+'_demo.hdf5')
  if not path.exists():raise FileNotFoundError(path)
  with h5py.File(path,'r') as f:
   demos=sorted(f['data'],key=lambda x:int(x.split('_')[-1]));
   if len(demos)!=50:raise RuntimeError(f'{path}: expected 50 demos')
   info=f['data'].attrs['problem_info'];info=info.decode() if isinstance(info,bytes) else info;stored_language=json.loads(info)['language_instruction']
   if stored_language!=language:raise RuntimeError(f'language mismatch task {task}')
   lengths=[]
   for ordinal,demo in enumerate(demos):
    g=f['data'][demo];n=len(g['actions']);
    if len(g['states'])!=n or n<2:raise RuntimeError(f'invalid episode {task}/{demo}')
    phase,progress=PHASES[ordinal%3];frame=math.floor((n-1)*progress);lengths.append(n);state={'state_id':f'task{task:02d}__demo{ordinal:02d}','task_id':task,'language':language,'demo_id':demo,'demo_ordinal':ordinal,'episode_length':n,'phase':phase,'target_progress':progress,'resolved_frame':frame,'demo_path':str(path)};states.append(state)
    for noise_ordinal in range(3):units.append({**state,'unit_id':f"{state['state_id']}__noise{noise_ordinal}",'noise_ordinal':noise_ordinal,'noise_seed':202608150000+task*10000+ordinal*10+noise_ordinal,'random_selection_seed':20260815+task*10000+ordinal})
  sources.append({'task_id':task,'language':language,'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'demo_count':50,'min_length':min(lengths),'max_length':max(lengths)})
 ck=w/'task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de';means=w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt';protocol={'experiment_name':'coreact_expert_aligned_direction_audit_v1','stage':'offline_development_direction_audit_no_rollout','suite':'libero_spatial','tasks':list(range(10)),'states_per_task':50,'state_sampling':'one state from each of 50 official expert demos; demo ordinal modulo 3 assigns early=0.2, middle=0.5, late=0.8; frame=floor((episode_length-1)*target_progress)','noise_seeds_per_state':3,'flow_taus':[round(1-.1*i,1) for i in range(10)],'matched_units':1500,'selectors':['attention8','random8'],'attention_selector':'native clean prefix, tau=1, late-half action-to-context attention, top 8 eligible visual tokens, recomputed per noise seed','random_selector':'uniform 8 eligible visual tokens, fixed per state across noise seeds','replacement':'v8 camera/position-conditioned visual mean','flow_target':{'x_tau':'tau*noise+(1-tau)*normalized_expert_action','v_star':'noise-normalized_expert_action','source':'checkpoint training implementation modeling_smolvla.py lines 787-789'},'guidance':{'toward':'clean + 0.5*trust_region_clip(counterfactual-clean)','away':'clean - 0.5*trust_region_clip(counterfactual-clean)','trust_region_kappa':.25,'real_action_dimensions':7},'primary_metrics':['cos(counterfactual-clean,expert-clean)','delta_toward_mse'],'aggregation':{'unit':'equal mean over 10 flow steps','state':'three-noise mean and sign agreement','task':'50 demonstration states, not 150 noise units treated as independent','phase':'early/middle/late state-level aggregation'},'no_summary_before_units_complete':1500,'data':{'repo':REPO,'revision':REVISION,'sources':sources},'checkpoint':{'repo':'lerobot/smolvla_libero','revision':'31d453f7edd78c839a8bbc39744a292686daf0de','config_sha256':sha(ck/'config.json'),'weights_sha256':sha(ck/'model.safetensors')},'calibration_mean_sha256':sha(means),'code_sha256':sha(w/'research/coreact_expert_direction/direction_audit.py')}
 (o/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False,allow_unicode=False));(o/'state_manifest.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in states));(o/'unit_manifest.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in units));(o/'source_audit.json').write_text(json.dumps({'repo':REPO,'revision':REVISION,'files':sources,'total_bytes':sum(x['bytes'] for x in sources),'all_hdf5_complete':True},indent=2,sort_keys=True)+'\n');(o/'decision.json').write_text(json.dumps({'decision':'QUALIFICATION_PENDING_NO_CAPTURE','states':500,'units':1500},indent=2)+'\n');print(o)
if __name__=='__main__':main()

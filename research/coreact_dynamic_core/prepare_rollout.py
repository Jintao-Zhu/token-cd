#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import yaml
ARMS=('vanilla','random8','attention8','instability8','combined8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase0',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();o=a.output.resolve();q=a.phase0.resolve()
 if o.exists():raise FileExistsError(o)
 if json.loads((q/'decision.json').read_text())['decision']!='PHASE0_DYNAMIC_SELECTOR_PASS_READY_FOR_ROLLOUT':raise RuntimeError('phase0 gate')
 for d in ('episodes','invalid_pairs','status','logs'): (o/d).mkdir(parents=True,exist_ok=True)
 ck=w/'task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de';means=w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt'
 protocol={'experiment_name':'coreact_dynamic_core_token_selection_rollout_v2','stage':'mechanism_development_not_confirmation','tasks':[4,7],'init_state_ids':list(range(50)),'arms':list(ARMS),'planned_pairs':100,'planned_episodes':500,'shared':{'same_init_reset_observation_preprocessing_noise_per_pair':True,'initial_observation_rule':'each arm independently resets; physics/robot fingerprints must match; cache vanilla real reset observation and use that byte-identical observation for every arm first replan; later observations are arm-native','renderer_mismatch_policy':'raw camera mismatch may be recorded but never enters policy input; any physics/robot mismatch invalidates whole pair','flow_steps':10,'chunk_size':50,'executed_actions_per_chunk':10,'max_control_steps':280,'group_count':8,'replacement':'v8 position-conditioned mean','operator':'toward scale=0.5 kappa=0.25 real action dims only','dynamic_selection':'recomputed from latest two-camera observation at every replan'},'selectors':{'random8':'deterministic modality-valid random 8','attention8':'native tau=1 late-half action-to-context attention top 8','instability8':'10-step adjacent JS contribution EMA beta=0.9 top 8','combined8':'sum of attention and instability percentile ranks top 8'},'seed_rules':{'reset':'150000000 + task*1000 + init','noise_base':'202608130000 + task*100000 + init*1000; add replan index','random_selection':'20260813 + task*10000 + init*100 + replan'},'analysis':{'no_summary_before_500':True,'paired_init_bootstrap_replicates':2000,'exact_mcnemar':True,'primary_ordering':'instability/combined versus attention versus random','random_magnitude_caveat':'Phase0 median postclip RMS 0.595x attention; no operator retuning','development_only':True},'repair_provenance':{'source_failed_artifact':'artifacts/coreact_dynamic_core_token_selection_rollout_v1_20260810_140753','source_failure':'task04 init15 raw camera2 reset render mismatch; no init15 outcomes written','old_outcomes_reused':False},'phase0_artifact':str(q),'phase0_decision_sha256':sha(q/'decision.json'),'checkpoint':{'repo':'lerobot/smolvla_libero','revision':'31d453f7edd78c839a8bbc39744a292686daf0de','config_sha256':sha(ck/'config.json'),'weights_sha256':sha(ck/'model.safetensors')},'calibration_mean_sha256':sha(means),'code_sha256':sha(w/'research/coreact_dynamic_core/dynamic_guidance.py')}
 (o/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False));rows=[]
 for t in (4,7):
  for init in range(50):
   for arm in ARMS:rows.append({'pair_id':f'task{t:02d}__init{init:02d}','episode_id':f'task{t:02d}__init{init:02d}__{arm}','task_id':t,'init_state_id':init,'arm':arm,'reset_seed':150000000+t*1000+init,'noise_seed_base':202608130000+t*100000+init*1000,'selection_seed_base':20260813+t*10000+init*100})
 (o/'episode_manifest.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));(o/'decision.json').write_text(json.dumps({'decision':'PROTOCOL_LOCKED_NO_ROLLOUT','planned_pairs':100,'planned_episodes':500},indent=2)+'\n');print(o)
if __name__=='__main__':main()

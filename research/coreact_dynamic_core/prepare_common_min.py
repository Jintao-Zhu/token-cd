#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime
from pathlib import Path
import yaml

ARMS=('vanilla','random8_common','attention8_common','instability8_common');CAP=3.0661711077317473
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();w=a.workspace.resolve();o=(a.output or w/'artifacts'/f'coreact_common_min_selector_task4_v2_{datetime.now():%Y%m%d_%H%M%S}').resolve()
 if o.exists():raise FileExistsError(o)
 for d in ('phase0_states','episodes','invalid_pairs','status','logs'):(o/d).mkdir(parents=True,exist_ok=True)
 ck=w/'task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de';means=w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt';failed=w/'artifacts/coreact_norm_matched_selector_task4_v1_20260810_164644'
 protocol={'experiment_name':'coreact_common_min_selector_task4_v2','stage':'mechanism_development_not_confirmation','task':4,'task_language':'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate','init_state_ids':list(range(50)),'unique_states':50,'arms':list(ARMS),'planned_pairs':50,'planned_episodes':200,'phase0_states':50,'operator':{'definition':'at each flow step jointly evaluate Random8, Attention8, Instability8; trust-region clip each raw clean-minus-counterfactual difference; M_common=min(M_R,M_A,M_I,M_cap); apply alpha_s=M_common/(M_s+eps)','magnitude_cap':CAP,'trust_region_kappa':.25,'action_dimensions':7,'extrapolation_forbidden':True,'joint_integration':'three selector-specific flow states are advanced jointly; common magnitude is recomputed every flow step'},'selectors':{'random8_common':'deterministic random 8 eligible visual tokens','attention8_common':'native tau=1 late-half action-to-context attention top 8','instability8_common':'10-step adjacent JS-contribution EMA beta=0.9 top 8'},'shared':{'frozen_eval_checkpoint':True,'same_init_state_reset_observation_preprocessing_noise_per_pair':True,'canonical_first_observation':'independent arm resets; vanilla real reset observation is used byte-identically by all arms at first replan; later observations are arm-native','flow_steps':10,'chunk_size':50,'execute_first_actions':10,'max_control_steps':280,'replacement':'v8 camera/position-conditioned visual mean','dynamic_selection_each_replan':True},'phase0_gate':{'exact8_protected_finite_deterministic_clean_parity':True,'all_alpha_le_1':True,'zero_extrapolation':True,'per_step_applied_norm_pairwise_relative_error_max':1e-5,'selector_rms_ratio_range':[.999,1.001],'median_common_over_attention_min':.8},'seed_rules':{'reset':'170004000+init','noise_base':'202608640000+init*1000; add replan','selection':'20270814+40000+init*100+replan'},'analysis':{'no_outcome_summary_before_200':True,'primary_pairs':['instability8_common-random8_common','attention8_common-random8_common','instability8_common-attention8_common'],'paired_bootstrap_replicates':10000,'exact_mcnemar':True},'supersedes_phase0':str(failed),'checkpoint':{'repo':'lerobot/smolvla_libero','revision':'31d453f7edd78c839a8bbc39744a292686daf0de','config_sha256':sha(ck/'config.json'),'weights_sha256':sha(ck/'model.safetensors')},'calibration_mean_sha256':sha(means),'code_sha256':sha(w/'research/coreact_dynamic_core/dynamic_guidance.py')}
 (o/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False));rows=[]
 for init in range(50):
  for arm in ARMS:rows.append({'pair_id':f'task04__init{init:02d}','episode_id':f'task04__init{init:02d}__{arm}','task_id':4,'init_state_id':init,'arm':arm,'reset_seed':170004000+init,'noise_seed_base':202608640000+init*1000,'selection_seed_base':20270814+40000+init*100})
 (o/'episode_manifest.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));(o/'decision.json').write_text(json.dumps({'decision':'COMMON_MIN_PHASE0_PENDING_NO_ROLLOUT','planned_pairs':50,'planned_episodes':200},indent=2)+'\n');print(o)
if __name__=='__main__':main()

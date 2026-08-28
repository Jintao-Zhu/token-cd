#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import yaml
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();o=a.output.resolve()
 if o.exists():raise FileExistsError(o)
 for d in ('states','status'): (o/d).mkdir(parents=True,exist_ok=True)
 ck=w/'task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de';means=w/'artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt'
 protocol={'experiment_name':'coreact_dynamic_core_token_selection_phase0_v1','stage':'outcome_blind_selector_qualification_no_rollout','tasks':[4,7],'init_state_ids':[0,1,2,3,4],'states':10,'selectors':['random','attention','instability','combined'],'eligible_tokens':'128 valid post-connector visual tokens only','group_count':8,'attention_score':'locked native-prefix late-half action-to-context attention at tau=1','instability_score':{'source':'late-half action-to-visual probability at each of 10 clean flow steps','visual_normalization':'renormalize over eligible visual tokens','transition':'per-token Jensen-Shannon contribution between adjacent steps','ema_beta':0.9},'combined_score':'sum of within-visual ascending percentile ranks of attention and instability; top 8','replacement':'v8 position-conditioned camera/token visual mean','operator':'locked toward: clean - 0.5*trust_region_clip(clean-masked), kappa=0.25, real action dims only','shared':'same real observation, preprocessing, Gaussian noise, frozen eval checkpoint','gate':{'all_selectors_exactly_8_and_protected_untouched':True,'deterministic_repeat_one_state':True,'clean_flow_hash_identical_across_selectors':True,'median_attn_instability_jaccard_max':0.75,'median_postclip_rms_ratio_to_attention_range':[0.5,2.0],'no_nonfinite':True},'formal_rollout_forbidden_until':'PHASE0_DYNAMIC_SELECTOR_PASS','checkpoint':{'repo':'lerobot/smolvla_libero','revision':'31d453f7edd78c839a8bbc39744a292686daf0de','config_sha256':sha(ck/'config.json'),'weights_sha256':sha(ck/'model.safetensors')},'calibration_mean_sha256':sha(means),'code_sha256':sha(w/'research/coreact_dynamic_core/dynamic_guidance.py')}
 (o/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False));(o/'decision.json').write_text(json.dumps({'decision':'PROTOCOL_LOCKED_PHASE0_NOT_RUN','rollout_started':False},indent=2)+'\n');print(o)
if __name__=='__main__':main()

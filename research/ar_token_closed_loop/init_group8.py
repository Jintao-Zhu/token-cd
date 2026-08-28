from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from datetime import datetime
import yaml
from .common import file_sha256, read_jsonl, write_json

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); a=p.parse_args()
 w, out=a.workspace.resolve(), a.artifact.resolve(); src=w/'artifacts/ar_token_closed_loop_magnitude_calibration_v1_20260809_011155'
 if out.exists(): raise FileExistsError(out)
 for d in ('snapshots','groups','episodes','logs'): (out/d).mkdir(parents=True)
 protocol={'experiment':'ar_token_closed_loop_causal_magnitude_calibration_k8_v1','created_at':datetime.now().astimezone().isoformat(),'source_single_token_artifact':str(src),'checkpoint_revision':'962318cec55ac10993ff0f5f43eda9a270b4c873','tasks':[2,9,3],'snapshots':75,'conditions':['vanilla','mask_top8_effect','mask_bottom8_effect','mask_random8'],'candidate_universe':256,'group_size':8,'selection':'single-token teacher-forced JS ranking; top/bottom 8; deterministic random 8 excluding both','replacement':'position_conditioned_visual_mean','intervention':'first control action only; later actions vanilla','rollouts':300,'bootstrap':{'clusters':['task','snapshot'],'replicates':2000,'seed':20260809},'decision_criteria':'same magnitude criteria as single-token calibration'}
 (out/'protocol.lock.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False),encoding='utf-8')
 plans=read_jsonl(src/'snapshot_plan.jsonl')
 with (out/'snapshot_plan.jsonl').open('w') as f:
  for x in plans: f.write(json.dumps(x,sort_keys=True)+'\n')
 for x in plans: shutil.copytree(src/'snapshots'/x['snapshot_id'],out/'snapshots'/x['snapshot_id'])
 write_json(out/'source_provenance.json',{'source_artifact':str(src),'source_protocol_sha256':file_sha256(src/'protocol.lock.yaml'),'source_environment_sha256':file_sha256(src/'environment.lock.json'),'source_snapshot_audit_sha256':file_sha256(src/'snapshot_audit.json'),'source_candidate_audit_sha256':file_sha256(src/'candidate_audit.json'),'source_mean_sha256':file_sha256(w/'artifacts/ar_token_counterfactual_qualification_v1_20260808_231044/position_conditioned_visual_mean.pt')})
 print(out)
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Evaluate target-free action consistency features task-heldout."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from research.coreact_revision.evaluate_region_sign_probe_v3 import ATTENTION,REGION,aggregate,evaluate

CONS=["clean_cross_condition_dispersion","masked_cross_condition_dispersion","masked_minus_clean_dispersion","clean_mask_consensus_distance","intervention_direction_consistency","mean_intervention_norm"]
VALUE=["late_half_value_weighted_attention","late_half_value_weighted_attention_std","raw_last_value_weighted_attention"]
def load(p):
 d={}
 for line in Path(p).read_text().splitlines():
  x=json.loads(line);d[(x["task_id"],x["demo_id"],x["frame_id"],x["group_id"])]=x
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument("--raw-effects",type=Path,required=True);p.add_argument("--consistency",type=Path,required=True);p.add_argument("--value-scores",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=aggregate(a.raw_effects);c=load(a.consistency);v=load(a.value_scores)
 for r in rows:
  k=(r["task_id"],r["demo_id"],r["frame_id"],r["group_id"]);r.update({x:c[k][x] for x in CONS});r.update({x:v[k][x] for x in VALUE})
 methods={"consistency_only":CONS,"region_plus_consistency":REGION+CONS,"all_target_free":ATTENTION+VALUE+REGION+CONS}
 result={"stage":"teacher-forced-orbit target-free-feature development; whole-task heldout","methods":{n:evaluate(rows,f,"task_id") for n,f in methods.items()},"limit":"Although Q_rel is not a feature, x_tau is constructed around demonstration actions in this offline experiment; deployment transfer remains untested."};a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()

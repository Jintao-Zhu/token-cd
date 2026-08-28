#!/usr/bin/env python3
"""Evaluate target-dependent signed gradient diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,precision_score,recall_score,balanced_accuracy_score
from research.coreact_revision.evaluate_region_sign_probe_v3 import REGION,aggregate,evaluate

GRAD=["gradient_dot_replacement_delta","gradient_l2","gradient_abs_dot"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--raw-effects",type=Path,required=True);p.add_argument("--gradients",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=aggregate(a.raw_effects);g={}
 for line in a.gradients.read_text().splitlines():
  x=json.loads(line);g[(x["task_id"],x["demo_id"],x["frame_id"],x["group_id"])]=x
 for r in rows:r.update({k:g[(r["task_id"],r["demo_id"],r["frame_id"],r["group_id"])][k] for k in GRAD})
 y=np.asarray([r["label_nuisance"] for r in rows]); pred=np.asarray([r["gradient_dot_replacement_delta"]<0 for r in rows]); score=np.asarray([-r["gradient_dot_replacement_delta"] for r in rows])
 direct={"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"average_precision":float(average_precision_score(y,score)),"selected":int(pred.sum())}
 result={"stage":"target-dependent diagnostic not online selector","groups":len(rows),"direct_first_order_sign":direct,"task_heldout_gradient_classifier":evaluate(rows,GRAD,"task_id"),"task_heldout_region_gradient_classifier":evaluate(rows,REGION+GRAD,"task_id"),"interpretation_limit":"The gradient uses demonstration teacher-forced error and is therefore label-adjacent; it cannot be deployed without a target action."}
 a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()

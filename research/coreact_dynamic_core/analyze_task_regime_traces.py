from __future__ import annotations
import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

def summarize_episode(record):
 traces=record.get('replan_traces',[]); norms=[]; ratios=[]; alphas=[]; clips=[]; overlaps=[]; selected=[]; cameras=[]
 for tr in traces:
  steps=tr.get('step_traces',[])
  if not steps: continue
  norms.extend(float(x['applied_guidance_norm']) for x in steps); ratios.extend(float(x['common_over_attention']) for x in steps); alphas.extend(float(x['alpha']) for x in steps); clips.extend(bool(x['clipped']) for x in steps)
  selected.append(set(tr.get('selected_indices',[])))
  toks=tr.get('selected_tokens',[]); cameras.append(sum(t.get('camera_id')=='camera1' for t in toks)/max(len(toks),1))
  if len(selected)>1: overlaps.append(len(selected[-1]&selected[-2])/max(len(selected[-1]|selected[-2]),1))
 return {'success':int(record['success']),'replans':len(traces),'mean_applied_norm':float(np.mean(norms)) if norms else 0.,'rms_applied_norm':float(np.sqrt(np.mean(np.square(norms)))) if norms else 0.,'mean_common_over_attention':float(np.mean(ratios)) if ratios else 0.,'mean_alpha':float(np.mean(alphas)) if alphas else 0.,'clip_fraction':float(np.mean(clips)) if clips else 0.,'mean_adjacent_jaccard':float(np.mean(overlaps)) if overlaps else 0.,'mean_camera1_fraction':float(np.mean(cameras)) if cameras else 0.}
def load_task4(path):
 return [json.loads(p.read_text()) for p in (path/'episodes').glob('*__attention8_common.json')]
def load_heldout(path,task):
 return [json.loads(p.read_text()) for p in (path/'episodes').glob('*__attention8_common.json') if json.loads(p.read_text())['task_id']==task]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--task4-artifact',type=Path,required=True); p.add_argument('--heldout-artifact',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 groups={4:load_task4(a.task4_artifact),3:load_heldout(a.heldout_artifact,3),9:load_heldout(a.heldout_artifact,9)}; rows=[]
 for task,records in groups.items():
  for r in records: rows.append({'task_id':task,**summarize_episode(r)})
 summary={}
 for task in groups:
  xs=[r for r in rows if r['task_id']==task]; summary[str(task)]={'episodes':len(xs),'vanilla_success_rate':float(np.mean([r['success'] for r in xs])),'success':{k:float(np.mean([r[k] for r in xs if r['success']==1])) for k in xs[0] if k not in ('task_id','success')},'failure':{k:float(np.mean([r[k] for r in xs if r['success']==0])) for k in xs[0] if k not in ('task_id','success')},'all':{k:float(np.mean([r[k] for r in xs])) for k in xs[0] if k not in ('task_id','success')}}
 out={'decision':'OFFLINE_TRACE_REGIME_ANALYSIS_COMPLETE','source_artifacts':{'task4':str(a.task4_artifact),'heldout':str(a.heldout_artifact)},'groups':summary,'rows':rows,'unavailable_features':['clean-correction cosine','parallel/orthogonal decomposition','translation/rotation/gripper correction split: velocity vectors were not saved in this rollout artifact']}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 lines=['# Repeated Guidance Task-Regime Trace Analysis','','This is outcome-linked offline analysis; no new rollout was run.','','## Task-level summary']
 for task,s in summary.items(): lines += [f"### Task {task}",f"- episodes: {s['episodes']}; Vanilla success: {s['vanilla_success_rate']:.1%}",f"- all: mean applied norm {s['all']['mean_applied_norm']:.3f}, RMS {s['all']['rms_applied_norm']:.3f}, mean alpha {s['all']['mean_alpha']:.3f}, clipping {s['all']['clip_fraction']:.1%}, adjacent token Jaccard {s['all']['mean_adjacent_jaccard']:.3f}, camera1 fraction {s['all']['mean_camera1_fraction']:.3f}",f"- Vanilla-success: {s['success']}",f"- Vanilla-failure: {s['failure']}"]
 lines += ['','## Interpretation','- Saved traces support norms, clipping, common-magnitude alpha, token overlap, and camera concentration comparisons.','- Velocity cosine and action-dimension decomposition were not saved, so those two hypotheses cannot be tested retrospectively.','- This analysis is descriptive and does not train a selector or claim causal prediction.']
 (a.output/'report.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

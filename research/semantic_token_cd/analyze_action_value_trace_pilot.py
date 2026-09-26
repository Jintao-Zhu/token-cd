import json
from pathlib import Path
import numpy as np
root=Path('/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero_action_value_trace_pilot_v2')
def summarize(ep_path):
 d=json.loads(ep_path.read_text()); z=np.load(ep_path.parent/'step_arrays.npz')
 pos=z['positive_action_logits']; neg=z['negative_action_logits']; gui=z['guided_action_logits']
 # first six continuous dimensions; seventh gripper remains clean in canonical code
 rows=[]
 trace=d['trace']
 for t in range(len(trace)):
  for q in range(6):
   zp=pos[t,q]; zn=neg[t,q]; zg=gui[t,q]
   order=np.argsort(zp)[::-1]; a1,a2=map(int,order[:2]); margin=float(zp[a1]-zp[a2]); g=float((zp[a1]-zn[a1])-(zp[a2]-zn[a2])); mg=float(zg[a1]-zg[a2])
   rows.append((q,margin,g,mg, int(np.argmax(zg)!=a1),trace[t]['task_phase'],trace[t]['matched_m']))
 arr=np.array([r[:5] for r in rows],dtype=float)
 phases={}
 for ph in sorted(set(r[5] for r in rows)):
  ss=[r for r in rows if r[5]==ph]
  phases[ph]={'n_dim_steps':len(ss),'clean_margin_median':float(np.median([r[1] for r in ss])),'guidance_G_median':float(np.median([r[2] for r in ss])),'guided_margin_median':float(np.median([r[3] for r in ss])),'winner_flip_rate':float(np.mean([r[4] for r in ss]))}
 return {'case_id':d['case_id'],'arm':d['arm'],'success':d['success'],'steps':d['steps'],'first_stable_grasp_step':d.get('first_stable_grasp_step'),'overall':{'n_dim_steps':len(rows),'clean_margin_median':float(np.median(arr[:,1])),'guidance_G_median':float(np.median(arr[:,2])),'guided_margin_median':float(np.median(arr[:,3])),'winner_flip_rate':float(np.mean(arr[:,4])),'guidance_G_negative_fraction':float(np.mean(arr[:,2]<0))},'phases':phases}
reports=[]
for p in sorted((root/'episodes').glob('*/*/episode.json')):
 if (p.parent.parent/'vanilla'/'episode.json').exists() and (p.parent.parent/'matched'/'episode.json').exists(): reports.append(summarize(p))
# unique arm summaries
out={'protocol_id':'LIBERO90_TRACE_FAITHFUL_ACTION_VALUE_PILOT_V2','complete_pairs':2,'pair_outcomes':[json.loads(p.read_text()) for p in sorted((root/'pairs').glob('*.json'))],'episode_summaries':reports,'method':'per-step/per-action-dimension clean top1/top2 margin; G=(z+-z-)[clean winner]-(z+-z-)[runner-up]; guided margin from stored z*; these are descriptive within observed trajectories, not independent samples'}
(root/'ACTION_VALUE_TRACE_PILOT_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

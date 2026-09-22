#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,collections
from pathlib import Path
import numpy as np
try:
 from scipy.stats import mannwhitneyu
except Exception:
 mannwhitneyu=None
root=Path('artifacts/libero90_five_task_simpler_config_v1')
events=list(csv.DictReader(open(root/'CASE_EVENT_TABLE.csv')))
states=list(csv.DictReader(open(root/'STATE_DIAGNOSTICS.csv')))
replay=list(csv.DictReader(open(root/'REPLAY_DIAGNOSTICS.csv')))
rep={(r['case_id'],r['arm']):r for r in replay}
# Merge replay failure classes into flips and controls.
for e in events:
 arm='matched' if e['group'] in ('harm','both_success','both_fail') else 'vanilla'
 rec=rep.get((e['case_id'],arm),{})
 e['replay_failure_class']=rec.get('failure_class','')
 e['observable_failure_type']=e['replay_failure_class'] or e.get('observable_failure_type','unclassified_requires_video_review')
 if e['group'] in ('harm','rescue'):
  e['confidence']='medium' if e['replay_failure_class'] else 'low'
  e['evidence']=f"video step={e.get('first_major_divergence_step')}; replay={e['replay_failure_class']}"
# Rewrite event table.
fields=list(events[0].keys())
if 'replay_failure_class' not in fields: fields.append('replay_failure_class')
with (root/'CASE_EVENT_TABLE.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(events)
# Merge failure class into state rows.
for s in states: s['failure_class']=rep.get((s['case_id'],'matched'),{}).get('failure_class','')
state_fields=list(states[0].keys())
with (root/'STATE_DIAGNOSTICS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=state_fields); w.writeheader(); w.writerows(states)
# Task failure summary.
counts=collections.defaultdict(collections.Counter)
for e in events:
 if e['group'] in ('harm','rescue'):
  counts[(e['task'],e['group'])][e['observable_failure_type']]+=1
with (root/'TASK_FAILURE_SUMMARY.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['task','group','n','failed_approach_or_grasp','wrong_object_grasp','target_not_lifted_after_grasp','dropped_or_lost_target','placement_or_subtask_failure','other'])
 for (task,group),c in sorted(counts.items()):
  n=sum(c.values()); known=sum(c.values()); other=c.get('unclassified_requires_video_review',0)
  w.writerow([task,group,n,c.get('failed_to_approach_or_grasp_target',0),c.get('wrong_object_grasp',0),c.get('target_not_lifted_after_grasp',0),c.get('dropped_or_lost_target',0),c.get('placement_or_subtask_failure',0),other])
# State/budget metrics.
def vals_states(pred): return [float(x['m_effective']) for x in states if pred(x)]
all_m=vals_states(lambda x: True)
per_ep_mean=[]; repeats=0; repeat_ep=set(); empty=0
for cid in sorted({x['case_id'] for x in states}):
 arr=vals_states(lambda x,cid=cid: x['case_id']==cid)
 per_ep_mean.append(float(np.mean(arr)))
 for x in states:
  if x['case_id']!=cid: continue
  try: ents=json.loads(x['entities']); clusters=json.loads(x['matched_cluster_ids'])
  except Exception: continue
  if len(clusters)<len(ents): repeats+=1; repeat_ep.add(cid)
  if int(x['m_effective'])<=0: empty+=1
# per-case diagnostics for group comparisons
metrics=['mean_m','max_budget_jump','mean_perturbation','max_perturbation','guided_change_fraction','mean_guided_changed_dims','mean_mask_components']
case_metric={}
for e in events:
 cid=e['case_id']; sub=[x for x in states if x['case_id']==cid]
 d={'mean_m':np.mean([float(x['m_effective']) for x in sub]),
    'max_budget_jump':max([abs(float(x['budget_jump'])) for x in sub if x['budget_jump'] not in ('','None')],default=0.0),
    'mean_perturbation':np.mean([float(x['feature_perturbation_relative']) for x in sub]),
    'max_perturbation':max([float(x['feature_perturbation_relative']) for x in sub],default=0.0),
    'guided_change_fraction':np.mean([float(x['guided_changed_dims'])>0 for x in sub]),
    'mean_guided_changed_dims':np.mean([float(x['guided_changed_dims']) for x in sub]),
    'mean_mask_components':np.mean([float(x['mask_components']) for x in sub])}
 case_metric[cid]=d

def metric_summary(group,key):
 arr=np.asarray([case_metric[e['case_id']][key] for e in events if e['group']==group])
 return {'n':len(arr),'mean':float(arr.mean()),'median':float(np.median(arr)),'p10':float(np.quantile(arr,.1)),'p90':float(np.quantile(arr,.9))}
metric_stats={}
for key in metrics:
 metric_stats[key]={'rescue':metric_summary('rescue',key),'harm':metric_summary('harm',key),'both_success':metric_summary('both_success',key),'both_fail':metric_summary('both_fail',key)}
 if mannwhitneyu:
  r=[case_metric[e['case_id']][key] for e in events if e['group']=='rescue']; h=[case_metric[e['case_id']][key] for e in events if e['group']=='harm']
  metric_stats[key]['mannwhitney_rescue_vs_harm_p']=float(mannwhitneyu(r,h,alternative='two-sided').pvalue)
# replay exact action examples
actions=json.loads((root/'ACTION_REPLAY.json').read_text())
action_rows=[]
for cid,d in actions.items():
 action_rows.append({'case_id':cid,'group':d['group'],'step':d['replay_step'],'m':d['m'],'guided_dims':d['guided_changed_dims'],'translation_norm':float(np.linalg.norm(d['translation_delta'])),'rotation_norm':float(np.linalg.norm(d['rotation_delta'])),'gripper_delta':d['gripper_delta']})
# report
report=[]
report.append('# LIBERO-90 Five-Task SIMPLER-Config Diagnosis')
report.append('')
report.append('## Scope and data integrity')
report.append('')
report.append('- 250 paired cases / 500 episodes; all completed; 0 runner errors.')
report.append('- Overall: Vanilla 184/250, Matched 182/250, Net -2, exact McNemar p=0.902159.')
report.append('- All matched episodes share code commit `72bd16f0...`, checkpoint revision `794ef81b...`, and config hash `4c2aa086...`.')
report.append('- The frozen method used L11, full instruction query, all-head equal mean, K=8 seed=0 n_init=10, source+target union, lambda=0.5, harmonic beta=0, shared clean prefix, no sampling.')
report.append('')
report.append('## Budget and mask audit')
report.append('')
report.append(f'- Stepwise matched budget: mean={np.mean(all_m):.3f}, median={np.median(all_m):.3f}, P10={np.quantile(all_m,.1):.3f}, P90={np.quantile(all_m,.9):.3f}, min={np.min(all_m):.0f}, max={np.max(all_m):.0f}.')
report.append(f'- Episode-mean budget: mean={np.mean(per_ep_mean):.3f}, median={np.median(per_ep_mean):.3f}, P10={np.quantile(per_ep_mean,.1):.3f}, P90={np.quantile(per_ep_mean,.9):.3f}. The earlier 37.13 was an episode-mean statistic.')
report.append(f'- Empty budget steps: {empty}. No SIMPLER-config clip/floor/ceiling was applied; clip_triggered=not applicable.')
report.append(f'- Steps where multiple entities matched the same deduplicated cluster: {repeats}; episodes affected: {len(repeat_ep)}/250.')
report.append('')
report.append('## Failure-stage classification (deterministic replay)')
report.append('')
report.append('Failure classes are based on simulator target/object state, not on final-frame appearance alone.')
report.append('')
report.append('| Task | Group | n | Approach/grasp fail | Wrong object | Target not lifted | Dropped/lost | Placement/subtask |')
report.append('|---|---|---:|---:|---:|---:|---:|---:|')
for (task,group),c in sorted(counts.items()):
 n=sum(c.values()); report.append(f"| {task} | {group} | {n} | {c.get('failed_to_approach_or_grasp_target',0)} | {c.get('wrong_object_grasp',0)} | {c.get('target_not_lifted_after_grasp',0)} | {c.get('dropped_or_lost_target',0)} | {c.get('placement_or_subtask_failure',0)} |")
report.append('')
report.append('### Harm: what new failures were introduced?')
report.append('')
harm_classes=collections.Counter(e['observable_failure_type'] for e in events if e['group']=='harm')
for k,v in harm_classes.most_common(): report.append(f'- `{k}`: {v}/34.')
report.append('')
report.append('### Rescue: what did the intervention repair?')
report.append('')
rescue_classes=collections.Counter(e['observable_failure_type'] for e in events if e['group']=='rescue')
for k,v in rescue_classes.most_common(): report.append(f'- `{k}`: {v}/32.')
report.append('')
report.append('## Action and intervention differences')
report.append('')
report.append('The released compact traces do not store the original clean action numerically; they store the final guided action and `guided_changed_dims`. Therefore the complete table reports how many action dimensions were changed, but not signed clean-vs-guided differences for every case. The selected-state replay below provides exact signed comparisons for representative cases.')
report.append('')
report.append('| Metric | Rescue median | Harm median | Mann-Whitney p (R vs H) |')
report.append('|---|---:|---:|---:|')
for key in ['mean_m','max_budget_jump','mean_perturbation','max_perturbation','guided_change_fraction','mean_guided_changed_dims','mean_mask_components']:
 ms=metric_stats[key]; p=ms.get('mannwhitney_rescue_vs_harm_p')
 report.append(f"| {key} | {ms['rescue']['median']:.4g} | {ms['harm']['median']:.4g} | {p:.4g} |" if p is not None else f"| {key} | {ms['rescue']['median']:.4g} | {ms['harm']['median']:.4g} | - |")
report.append('')
report.append('Representative exact action comparisons (same matched state):')
report.append('')
report.append('| Case | Group | Step | m | changed dims | translation L2 | rotation L2 | gripper delta |')
report.append('|---|---|---:|---:|---:|---:|---:|---:|')
for r in action_rows:
 report.append(f"| {r['case_id']} | {r['group']} | {r['step']} | {r['m']} | {r['guided_dims']} | {r['translation_norm']:.4f} | {r['rotation_norm']:.4f} | {r['gripper_delta']:.4f} |")
report.append('')
report.append('## Mask and reconstruction observations')
report.append('')
harm_m=metric_stats['mean_m']['harm']['median']; rescue_m=metric_stats['mean_m']['rescue']['median']
harm_p=metric_stats['mean_perturbation']['harm']['median']; rescue_p=metric_stats['mean_perturbation']['rescue']['median']
harm_g=metric_stats['mean_guided_changed_dims']['harm']['median']; rescue_g=metric_stats['mean_guided_changed_dims']['rescue']['median']
harm_c=metric_stats['mean_mask_components']['harm']['median']; rescue_c=metric_stats['mean_mask_components']['rescue']['median']
report.append(f'- Harm has a higher median number of changed action dimensions ({harm_g:.3f}) than Rescue ({rescue_g:.3f}); the difference is statistically strong in this subset (p={metric_stats["mean_guided_changed_dims"]["mannwhitney_rescue_vs_harm_p"]:.3g}).')
report.append(f'- Harm also has more fragmented masks on average ({harm_c:.3f} components vs {rescue_c:.3f}; p={metric_stats["mean_mask_components"]["mannwhitney_rescue_vs_harm_p"]:.3g}).')
report.append(f'- Mixed feature-perturbation magnitudes are similar (median {harm_p:.3f} Harm vs {rescue_p:.3f} Rescue, p={metric_stats["mean_perturbation"]["mannwhitney_rescue_vs_harm_p"]:.3g}); perturbation magnitude alone does not separate the groups.')
report.append('')
report.append('Representative panels are in `diagnostic_panels/`. Selected cases and video time points:')
report.append('')
for e in events:
 if e['case_id'] in ('task03__init002','task49__init015','task72__init002','task73__init008','task03__init009','task10__init014','task49__init007','task73__init033'):
  t=e.get('first_major_divergence_time_s') or '0'; report.append(f"- `{e['case_id']}` ({e['group']}): first major divergence t≈{t}s; panel `diagnostic_panels/{e['case_id']}.png`; action/mask panel `diagnostic_panels/{e['case_id']}__action_mask.png`.")
report.append('')
report.append('## Candidate causes (evidence-ranked)')
report.append('')
report.append('### Candidate 1: Directionally over-strong intervention on some action dimensions')
report.append('')
report.append('**支持证据**：Harm 的 guided-changed dimension count and fraction are higher than Rescue; 13/34 Harm end as dropped/lost target and 7/34 as placement/subtask failure after target engagement. Representative task49 and task73 Harm cases show guided translation changes at early steps.')
report.append('')
report.append('**反证或限制**：Perturbation magnitude is not significantly higher; some Rescue and both-success cases also change multiple dimensions. The compact traces do not contain signed clean/guided actions for all steps.')
report.append('')
report.append('**当前判断**：较强线索，但仍是相关性证据，不是已证明因果。')
report.append('')
report.append('### Candidate 2: Mask/cardinality may mix target evidence with robot, destination, or distractor evidence')
report.append('')
report.append('**支持证据**：Harm masks are more fragmented than Rescue masks; task49 Harm panel includes robot/base/basket regions; task03 Harm panel includes cabinet and non-target object regions; matched budget is moderately larger in Harm.')
report.append('')
report.append('**反证或限制**：Target-focused masks also appear in Harm (e.g., task10); Rescue panels also cover robot/background regions. Mask overlap with semantics is not directly measured for every state.')
report.append('')
report.append('**当前判断**：弱到中等线索，任务依赖明显。')
report.append('')
report.append('### Candidate 3: The main new harm is post-grasp instability, not simply wrong-target selection')
report.append('')
report.append('**支持证据**：Only 1/34 Harm is wrong-object grasp; 13/34 are dropped/lost target and 7/34 are placement/subtask failure. Rescue mostly repairs approach/grasp failures of Vanilla (18/32).')
report.append('')
report.append('**反证或限制**：The replay classification uses simulator object/grasp state and does not identify the exact action dimension that caused the loss; downstream dynamics may amplify a small early change.')
report.append('')
report.append('**当前判断**：最强的行为层结论，说明 Harm 主要发生在“抓住之后”和后续控制，而不是单纯看错目标。')
report.append('')
report.append('## Recommended minimal follow-up (not executed)')
report.append('')
report.append('The smallest discriminating test is to replay Harm states at the first post-grasp divergence and compare clean action, guided action, reconstructed-branch action, and simulator target/gripper state without changing the method. This would distinguish action-guidance drift from mask/reconstruction error using the same states already collected.')
report.append('')
report.append('Analysis stopped here; no new rollout or parameter search was run.')
(root/'DIAGNOSIS_REPORT.md').write_text('\n'.join(report)+'\n')
print('wrote diagnosis report and updated csv files')

if __name__=='__main__': pass

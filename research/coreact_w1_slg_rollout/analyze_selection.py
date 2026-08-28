from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ARMS=("lambda_0","lambda_005","lambda_010","lambda_025","lambda_050")
DOSES={"lambda_0":0.0,"lambda_005":0.05,"lambda_010":0.1,"lambda_025":0.25,"lambda_050":0.5}

def stats(rows,left,right,rng):
    x=np.asarray([int(r[left])-int(r[right]) for r in rows],dtype=np.int8); b=[]
    for _ in range(10000): b.append(x[rng.integers(0,len(x),len(x))].mean()*100)
    rescue=int((x==1).sum()); harm=int((x==-1).sum()); n=rescue+harm
    return {"delta_pp":float(x.mean()*100),"ci95_pp":[float(np.quantile(b,.025)),float(np.quantile(b,.975))],"rescue":rescue,"harm":harm,"mcnemar_p":float(binomtest(min(rescue,harm),n).pvalue) if n else 1.0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();out=a.artifact.resolve(); paths=sorted(out.joinpath('episodes').glob('*.json')); episodes=[json.loads(x.read_text()) for x in paths]
    selection=[x for x in episodes if x['split']=='selection']
    if len(selection)!=1250: raise RuntimeError(f'selection episodes {len(selection)} != 1250')
    groups={}
    for e in selection: groups.setdefault(e['pair_id'],{})[e['arm']]=e
    if len(groups)!=250 or any(set(g)!=set(ARMS) for g in groups.values()): raise RuntimeError('selection pairing failed')
    rows=[]
    for pair,g in sorted(groups.items()): rows.append({'pair_id':pair,'task':g['lambda_0']['task_id'],**{arm:bool(g[arm]['success']) for arm in ARMS}})
    rng=np.random.default_rng(20260817); comparisons={arm:stats(rows,arm,'lambda_0',rng) for arm in ARMS[1:]}; rates={arm:sum(r[arm] for r in rows)/250 for arm in ARMS}
    task_rows=[]
    for task in range(10):
        s=[r for r in rows if r['task']==task]; task_rows.append({'task':task,**{arm:sum(r[arm] for r in s) for arm in ARMS}})
    candidates=[]
    for arm in ARMS[1:]:
        c=comparisons[arm]; nonworse=sum(r[arm]>=r['lambda_0'] for r in task_rows)
        c['tasks_nonworse']=nonworse; c['passes']=bool(c['delta_pp']>=2 and c['rescue']>c['harm'] and nonworse>=6 and rates[arm]>rates['lambda_0'])
        if c['passes']: candidates.append(arm)
    if candidates:
        candidates.sort(key=lambda arm:(comparisons[arm]['delta_pp'], -DOSES[arm]),reverse=True); selected=candidates[0]; decision='SELECTION_PASS'
    else: selected=None; decision='W1_DOSE_RESPONSE_NO_GO'
    dose_metrics={}
    for arm in ARMS:
        traces=[t for e in selection for t in e['replan_traces'] if e['arm']==arm]
        deltas=[t['normalized_action_delta_l2'] for t in traces]; clips=[c for t in traces for c in t['clip_scales']]
        corrections=[x for t in traces for x in t['velocity_correction_norms']]
        dose_metrics[arm]={'mean_normalized_action_delta_l2':float(np.mean(deltas)), 'mean_xyz_delta':float(np.mean([t['normalized_action_delta_xyz'] for t in traces])), 'mean_rotation_delta':float(np.mean([t['normalized_action_delta_rotation'] for t in traces])), 'mean_gripper_delta':float(np.mean([t['normalized_action_delta_gripper'] for t in traces])), 'clip_rate':float(np.mean(np.asarray(clips)<1.0)) if clips else 0.0, 'mean_clip_scale':float(np.mean(clips)) if clips else 1.0, 'mean_velocity_correction_norm':float(np.mean(corrections)) if corrections else 0.0}
    result={'stage':'selection','decision':decision,'rates':rates,'comparisons':comparisons,'dose_metrics':dose_metrics,'task_rows':task_rows,'candidate_lambdas':candidates,'selected_arm':selected,'selection_pairs':250}
    (out/'selection_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    if selected: (out/'selected_dose.lock.json').write_text(json.dumps({'selected_arm':selected,'lambda':DOSES[selected],'selection_result_sha256':__import__('hashlib').sha256((out/'selection_analysis.json').read_bytes()).hexdigest(),'confirmation_allowed':True},indent=2)+'\n')
    (out/'selection_decision.json').write_text(json.dumps({'decision':decision,'selected_arm':selected},indent=2)+'\n')
    print(json.dumps(result,indent=2));
    if not selected: raise SystemExit(2)
if __name__=='__main__':main()

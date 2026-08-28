import json,glob,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import binomtest

def main():
 art=Path('artifacts/coreact_native_action_support_ratio_r0_stage_a_v1_20260813_180803'); rows=[json.load(open(p)) for p in glob.glob(str(art/'episodes/*.json'))]; groups=defaultdict(list)
 for x in rows:groups[(x['snapshot_id'],x['candidate'])].append(x)
 integrity={'episodes':len(rows),'snapshots':len({x['snapshot_id'] for x in rows}),'snapshot_candidate_groups':len(groups),'all_finite':all(x['all_actions_finite'] for x in rows),'six_per_group':all(len(v)==6 for v in groups.values()),'roles_per_group':all(Counter(x['seed_role'] for x in v)=={'selection':3,'evaluation':3} for v in groups.values()),'branch_hash_match':all(len({x['branch_sim_state_sha256'] for x in v})==1 for v in groups.values()),'candidate_noise_hash_match':all(len({x['candidate_noise_sha256'] for x in v})==1 for v in groups.values()),'candidate_chunk_hash_match':all(len({x['candidate_chunk_first10_sha256'] for x in v})==1 for v in groups.values())};integrity['pass']=integrity['episodes']==1440 and integrity['snapshots']==40 and integrity['snapshot_candidate_groups']==240 and all(v for k,v in integrity.items() if isinstance(v,bool))
 bysnap=defaultdict(list)
 for x in rows:bysnap[x['snapshot_id']].append(x)
 evals=[]; choices=[]
 for sid,v in sorted(bysnap.items()):
  sel={c:sum(x['success'] for x in v if x['candidate']==c and x['seed_role']=='selection') for c in range(6)};mx=max(sel.values());chosen=0 if sel[0]==mx else min(c for c,z in sel.items() if z==mx); choices.append(chosen)
  for seed in range(3,6):
   vv=next(x for x in v if x['candidate']==0 and x['continuation_seed']==seed);oo=next(x for x in v if x['candidate']==chosen and x['continuation_seed']==seed);evals.append({'snapshot_id':sid,'task_id':int(sid[4:6]),'seed':seed,'V':int(vv['success']),'O':int(oo['success']),'chosen':chosen})
 vr=np.mean([x['V'] for x in evals]);orr=np.mean([x['O'] for x in evals]);delta=(orr-vr)*100;res=sum(x['O'] and not x['V'] for x in evals);harm=sum(x['V'] and not x['O'] for x in evals);disc=res+harm
 task=[]
 for t in range(10):
  z=[x for x in evals if x['task_id']==t];task.append({'task_id':t,'vanilla':np.mean([x['V'] for x in z]),'oracle':np.mean([x['O'] for x in z]),'delta_pp':100*np.mean([x['O']-x['V'] for x in z])})
 arr=np.array([[x['O']-x['V'] for x in evals if x['snapshot_id']==sid] for sid in sorted(bysnap)]);rng=np.random.default_rng(20260813);boot=np.array([arr[rng.integers(0,40,40)].mean()*100 for _ in range(100000)]);ci=np.quantile(boot,[.025,.975]).tolist();nonneg=sum(x['delta_pp']>=0 for x in task)
 if delta>=8 and ci[0]>0 and nonneg>=7 and res>harm:decision='NATIVE_SUPPORT_HEADROOM_STRONG_GO'
 elif delta>3:decision='NATIVE_SUPPORT_HEADROOM_PROMISING'
 else:decision='NATIVE_SUPPORT_HEADROOM_NO_GO'
 out={'integrity':integrity,'decision':decision,'vanilla_rate':float(vr),'heldout_oracle_rate':float(orr),'oracle_minus_vanilla_pp':float(delta),'snapshot_bootstrap_95_ci_pp':ci,'rescue':int(res),'harm':int(harm),'mcnemar_exact_p':float(binomtest(min(res,harm),disc).pvalue) if disc else 1.0,'nonnegative_tasks':int(nonneg),'choice_counts':{str(k):int(v) for k,v in Counter(choices).items()},'per_task':[{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in x.items()} for x in task]}
 (art/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');(art/'report.md').write_text(f"# Native Action Support R0 Stage A\n\nIntegrity: {'PASS' if integrity['pass'] else 'FAIL'} (1440/1440).\n\n- Vanilla: {vr:.1%}\n- Held-out Oracle: {orr:.1%}\n- Delta: {delta:+.1f}pp\n- Snapshot bootstrap 95% CI: [{ci[0]:+.1f}, {ci[1]:+.1f}]pp\n- Rescue/harm: {res}/{harm}\n- Non-negative tasks: {nonneg}/10\n\nDecision: `{decision}`\n")
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()

import json,glob
from collections import defaultdict
from pathlib import Path
def main():
 art=Path('artifacts/coreact_guidance_placement_existence_gate_v1_20260813_150411');arms=['vanilla','full','early','mid','late','near','far'];d=defaultdict(dict)
 for p in glob.glob(str(art/'episodes/*.json')):
  x=json.load(open(p));d[(x['task_id'],x['init_state_id'])][x['arm']]=x
 assert len(d)==200 and all(set(v)==set(arms) for v in d.values())
 rates={a:sum(v[a]['success'] for v in d.values())/200 for a in arms}; rows=[]
 for t in range(10): rows.append({'task_id':t,**{a:sum(v[a]['success'] for (tt,_),v in d.items() if tt==t)/20 for a in arms}})
 paired={}
 for a in arms[1:]:
  rescue=sum(v[a]['success'] and not v['vanilla']['success'] for v in d.values());harm=sum(v['vanilla']['success'] and not v[a]['success'] for v in d.values());paired[a]={'rescue':rescue,'harm':harm,'difference_pp':(rescue-harm)/2}
 best=max(arms[1:],key=rates.get);best_gain=(rates[best]-rates['vanilla'])*100;full_gain=(rates['full']-rates['vanilla'])*100;task_nonneg=sum(r[best]>=r['vanilla'] for r in rows)
 decision='GUIDANCE_PLACEMENT_EXISTENCE_NO_GO'
 if best_gain>=5 and best_gain-full_gain>=3 and task_nonneg>=7: decision='GUIDANCE_PLACEMENT_EXISTENCE_STRONG_GO'
 elif best_gain>=3 and task_nonneg>=7: decision='GUIDANCE_PLACEMENT_EXISTENCE_PROMISING'
 out={'episodes':len(d)*7,'rates':rates,'per_task':rows,'paired_vs_vanilla':paired,'best_arm':best,'best_minus_vanilla_pp':best_gain,'best_minus_full_pp':best_gain-full_gain,'tasks_nonnegative_vs_vanilla':task_nonneg,'decision':decision}
 (art/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');
 lines=['# Guidance Placement Existence Gate','','Integrity: PASS, 200 paired units, 1400/1400 episodes.','', '## Success rates']+[f'- {a}: {rates[a]:.1%} ({round(rates[a]*200)}/200)' for a in arms]+['',f'Best arm: `{best}`; versus Vanilla: `{best_gain:+.1f}pp`; versus Full: `{best_gain-full_gain:+.1f}pp`.',f'Non-negative tasks for best arm: `{task_nonneg}/10`.','', '## Paired rescue/harm']+[f'- {a}: rescue {v["rescue"]}, harm {v["harm"]}, net {v["difference_pp"]:+.1f}pp' for a,v in paired.items()]+['','## Decision',decision,'','The result is a development existence screen, not an independent confirmation.']
 (art/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

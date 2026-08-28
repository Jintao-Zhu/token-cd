#!/usr/bin/env python3
"""Analyze structured group/duration causal effects after complete collection."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from research.coreact_causal_dataset.common import paired_effect
from research.coreact_closed_loop.run_pilot import read_jsonl,write_json
def h(p):
 d=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): d.update(b)
 return d.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();o=a.artifact.resolve(); manifest=read_jsonl(o/'rollout_manifest.jsonl'); results={};missing=[];failed=[]
 for s in manifest:
  d=o/'episodes'/s['episode_id'];
  if (d/'episode.json').exists(): results[s['episode_id']]=json.loads((d/'episode.json').read_text())
  elif (d/'failure.json').exists(): failed.append(s['episode_id'])
  else: missing.append(s['episode_id'])
 complete={'expected':len(manifest),'complete':len(results),'missing':len(missing),'failed':len(failed),'duplicates':len(manifest)-len({s['episode_id'] for s in manifest}),'paired_identity_failures':[]}
 bysnap=defaultdict(list)
 for s in manifest:bysnap[s['snapshot_id']].append(s)
 for sid,specs in bysnap.items():
  clean={s['replicate']:results.get(s['episode_id']) for s in specs if s['condition']=='clean'}
  for s in (x for x in specs if x['condition']=='masked' and x['episode_id'] in results):
   c=clean[s['replicate']];m=results[s['episode_id']]
   if c is None:continue
   n=min(len(c['noise_sha256_by_replan']),len(m['noise_sha256_by_replan']))
   if c['initial_sim_state_sha256']!=m['initial_sim_state_sha256'] or c['initial_prefix_sha256']!=m['initial_prefix_sha256'] or c['noise_sha256_by_replan'][:n]!=m['noise_sha256_by_replan'][:n]:complete['paired_identity_failures'].append(s['episode_id'])
 write_json(o/'completeness.json',complete)
 if len(results)!=len(manifest) or missing or failed or complete['duplicates'] or complete['paired_identity_failures']:
  write_json(o/'decision.json',{'status':'INCONCLUSIVE_DATA_OR_RESOURCES','completeness':complete});raise SystemExit(1)
 rows=[]
 for sid,specs in sorted(bysnap.items()):
  clean={s['replicate']:results[s['episode_id']] for s in specs if s['condition']=='clean'}; groups=defaultdict(list)
  for s in specs:
   if s['condition']=='masked':groups[(s['group_id'],s['duration'])].append(s)
  for (gid,dur),gs in groups.items():
   gs.sort(key=lambda x:x['replicate']);cs=[int(clean[x['replicate']]['success']) for x in gs];ms=[int(results[x['episode_id']]['success']) for x in gs];eff=paired_effect(cs,ms);f=gs[0]
   rows.append({'snapshot_id':sid,'task_id':f['task_id'],'phase':f['assigned_phase'],'group_id':gid,'duration':dur,'token_count':len(f['token_indices']),'clean_successes':sum(cs),'masked_successes':sum(ms),**eff})
 with (o/'structured_effects.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 strat={}
 for gid in sorted({r['group_id'] for r in rows}):
  for dur in (1,3):
   sub=[r for r in rows if r['group_id']==gid and r['duration']==dur];strong=[r for r in sub if abs(r['tau'])>0.4]
   strat[f'{gid}_d{dur}']={'groups':len(sub),'tau_zero_fraction':sum(r['tau']==0 for r in sub)/len(sub),'strong_fraction':len(strong)/len(sub),'anchors':sum(r['strong_anchor'] for r in sub),'nuisances':sum(r['strong_nuisance'] for r in sub),'tau_distribution':dict(Counter(r['tau'] for r in sub))}
 overall=[r for r in rows if abs(r['tau'])>0.4];go=len(overall)/len(rows)>=.15 and any(r['strong_anchor'] for r in rows) and any(r['strong_nuisance'] for r in rows);status='STRUCTURED_CAUSAL_UNITS_QUALIFIED' if go else 'STRUCTURED_CAUSAL_UNITS_NO_GO';summary={'rows':len(rows),'strong_fraction':len(overall)/len(rows),'strong_anchors':sum(r['strong_anchor'] for r in rows),'strong_nuisances':sum(r['strong_nuisance'] for r in rows),'stratified':strat,'by_phase':{ph:{'groups':sum(r['phase']==ph for r in rows),'strong':sum(r['phase']==ph and abs(r['tau'])>.4 for r in rows)} for ph in sorted({r['phase'] for r in rows})}}
 write_json(o/'summary.json',summary);write_json(o/'decision.json',{'status':status,'gate_pass':go,'classifier_trained':False})
 report=f"""# Structured Causal Unit Qualification\n\n- Complete rollouts: {len(results)}/{len(manifest)}; missing/failed: 0/0.\n- Structured state/group/duration labels: {len(rows)}.\n- Overall strong effects: {len(overall)}/{len(rows)} ({len(overall)/len(rows):.1%}).\n- Strong anchors: {summary['strong_anchors']}; strong nuisances: {summary['strong_nuisances']}.\n\n## Stratified results\n\n```json\n{json.dumps(strat,indent=2,sort_keys=True)}\n```\n\n## Decision\n\n`{status}`\n\nNo attention ranking or learned classifier was used.\n""";(o/'report.md').write_text(report)
 audit={str(x.relative_to(o)):h(x) for x in sorted(o.rglob('*')) if x.is_file() and x.name!='sha256_audit.json'};write_json(o/'sha256_audit.json',audit);print(json.dumps({'status':status,**summary},indent=2,sort_keys=True))
if __name__=='__main__':main()

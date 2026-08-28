#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();w,art=a.workspace.resolve(),a.artifact.resolve();lock=json.loads((art/'selected_self_weak.lock.json').read_text())
 if lock['candidate']!='W1_skip_last_1' or lock['low_steps']!=[8,9] or lock['high_steps']!=[0,1]:raise RuntimeError('unexpected frozen selection lock')
 source=w/'artifacts/coreact_capacity_weak_v1_20260815_202000/state_manifest.jsonl';rows=[json.loads(x) for x in source.read_text().splitlines() if json.loads(x)['split']=='confirmation']
 if len(rows)!=250 or any(r['demo_ordinal']%2!=1 for r in rows):raise RuntimeError('fresh confirmation manifest invalid')
 selection_ids={json.loads(x)['state_id'] for x in (art/'state_manifest.jsonl').read_text().splitlines()}
 if selection_ids&{r['state_id'] for r in rows}:raise RuntimeError('selection/confirmation overlap')
 root=art/'confirmation_manifold';(root/'neighbors').mkdir(parents=True,exist_ok=True);(root/'state_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 protocol={'stage':'FROZEN_CONFIRMATION','selected_lock_sha256':sha(art/'selected_self_weak.lock.json'),'candidate':'W1_skip_last_1','low_steps':[8,9],'high_placebo_steps':[0,1],'lambda':.5,'trust_region_kappa':.25,'states':250,'tasks':10,'states_per_task':25,'source_manifest':str(source),'source_manifest_sha256':sha(source),'selection_overlap':0,'neighbors':{'k':16,'same_task':True,'anchor_trajectory_excluded':True,'one_state_per_neighbor_trajectory':True,'physical_features_and_standardization':'identical build_manifold_neighbors implementation'}};(art/'confirmation_protocol.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n');(art/'status/current.json').write_text(json.dumps({'stage':'CONFIRMATION_NEIGHBORS_PENDING'},indent=2)+'\n');print(root)
if __name__=='__main__':main()

from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--artifact',type=Path,required=True); p.add_argument('--reference-artifact',type=Path,required=True); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shard-count',type=int,default=1); a=p.parse_args()
 ws=a.workspace.resolve(); art=a.artifact.resolve(); ref=a.reference_artifact.resolve()
 if not (art/'status'/'dry_run.pass.json').exists(): raise RuntimeError('dry-run gate has not passed')
 rows=[json.loads(x) for x in (art/'episode_manifest.jsonl').read_text().splitlines()]; pairs=[]; seen=set()
 for row in rows:
  key=(row['snapshot_id'],row['noise_seed'])
  if key not in seen: seen.add(key); pairs.append(key)
 log=art/'logs'/'formal_progress.jsonl'
 pairs=[pair for index,pair in enumerate(pairs) if index % a.shard_count == a.shard_index]
 for index,(snapshot,seed) in enumerate(pairs,1):
  output=art/'status'/f'unit_{snapshot}_{seed}.json'
  if output.exists(): continue
  cmd=[str(ws/'task1/.conda-envs/flow-vla/bin/python'),str(ws/'research/coreact_self_guidance/run_one_shot_direction.py'),'--workspace',str(ws),'--artifact',str(art),'--reference-artifact',str(ref),'--snapshot',snapshot,'--noise-seed',str(seed)]
  subprocess.run(cmd,cwd=ws,check=True)
  with log.open('a') as stream: stream.write(json.dumps({'completed':index,'planned':len(pairs),'snapshot':snapshot,'noise_seed':seed})+'\n')
 if a.shard_count == 1:
  (art/'status'/'rollout.complete').write_text(f'{len(pairs)}/{len(pairs)} causal units; {len(pairs)*3}/{len(pairs)*3} episodes\n')
if __name__=='__main__': main()

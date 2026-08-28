from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
ARMS=('Strong','W1_plus','W1_minus','Nearest_manifold','Random_smooth')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();o=a.artifact.resolve();gate=json.loads((o/'snapshot_gate.json').read_text());
 if gate['decision']!='LOCAL_SUCCESS_SNAPSHOT_GATE_PASS':raise RuntimeError('snapshot gate')
 snaps=sorted(o.joinpath('snapshots').glob('*.pt'))
 if len(snaps)!=100 or len(list(o.joinpath('neighbors').glob('*.npz')))!=100:raise RuntimeError('snapshot/neighbor count')
 rows=[]
 for ordinal,path in enumerate(snaps):
  import torch;meta=torch.load(path,weights_only=False,map_location='cpu')['metadata']
  for seed in range(5):
   unit=f"{meta['snapshot_id']}__seed{seed}";noise=940_000_000+ordinal*100+seed
   for arm in ARMS:rows.append({'episode_id':f'{unit}__{arm}','unit_id':unit,'snapshot_id':meta['snapshot_id'],'task_id':meta['task_id'],'init_state_id':meta['init_state_id'],'progress':meta['target_progress'],'continuation_seed':seed,'noise_seed':noise,'random_direction_seed':950_000_000+ordinal,'arm':arm})
 with (o/'u0_manifest.jsonl').open('x') as f:f.write('\n'.join(json.dumps(x,sort_keys=True) for x in rows)+'\n')
 (o/'u0_manifest.sha256').write_text(sha(o/'u0_manifest.jsonl')+'  u0_manifest.jsonl\n');(o/'status.json').write_text(json.dumps({'status':'U0_READY','causal_units':500,'episodes_planned':2500},indent=2)+'\n');print(len(rows))
if __name__=='__main__':main()

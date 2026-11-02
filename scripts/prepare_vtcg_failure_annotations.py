"""Generate blinded pair manifest after formal membership audit.
No automated failure category is presented as human annotation.

With --materialize, each selected clip is COPIED to
failure_annotation/blinded/<pair_id>_<A|B>.mp4 so the reviewer never sees the arm
name or the original path; the analyst-only key keeps the token -> arm mapping.
"""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True)
 ap.add_argument('--materialize',action='store_true',help='copy clips into failure_annotation/blinded/')
 a=ap.parse_args()
 audit=json.loads((a.artifact/'FORMAL_MEMBERSHIP_AUDIT.json').read_text())
 if not audit['passed']:raise RuntimeError('formal membership must pass')
 c=json.loads((a.artifact/'CONFIG_LOCK.json').read_text());rng=np.random.default_rng(98211);out=a.artifact/'failure_annotation';out.mkdir(exist_ok=True)
 keyfile=out/'analyst_key.json';csvfile=out/'blinded_annotations.csv'
 if keyfile.exists() or csvfile.exists():raise RuntimeError('refuse overwrite annotation work')
 blinded=None
 if a.materialize:
  blinded=out/'blinded'
  if blinded.exists():raise RuntimeError(f'refuse to write into existing {blinded}')
  blinded.mkdir(parents=True)
 keys={};annotations=[]
 for task in c['tasks']:
  strata={x:[] for x in ['rescue','harm','both_success','both_failure']}
  for seed in c['formal_seeds']:
   ds=[]
   for arm in ['vanilla','attn_matched']:
    p=a.artifact/'episodes/formal'/task/arm/f'episode_{seed:03d}_summary.json';ds.append(json.loads(p.read_text()))
   v,m=ds
   kind='rescue' if m['success'] and not v['success'] else 'harm' if v['success'] and not m['success'] else 'both_success' if v['success'] else 'both_failure'
   strata[kind].append((seed,ds))
  selected=[]
  for kind,items in strata.items():
   if kind.startswith('both_') and len(items)>20:
    items=[items[int(i)] for i in rng.choice(len(items),20,replace=False)]
   selected.extend((kind,seed,ds) for seed,ds in items)
  double=set(rng.choice(len(selected),size=int(np.ceil(.2*len(selected))),replace=False).tolist())
  for ix,(kind,seed,ds) in enumerate(selected):
   token=hashlib.sha256(f'vtcg-review-{task}-{seed}'.encode()).hexdigest()[:12]
   flip=bool(rng.integers(2));order=[1,0] if flip else [0,1]
   keys[token]={'task':task,'seed':seed,'stratum':kind,'double_annotate':ix in double,'A_arm':['vanilla','attn_matched'][order[0]],'B_arm':['vanilla','attn_matched'][order[1]],'videos':{label:ds[j]['video'] for label,j in zip(['A','B'],order)}}
   blinded_names={}
   for label,j in zip(['A','B'],order):
    source=Path(ds[j]['video'])
    if not source.is_file():raise RuntimeError(f'missing clip {source}')
    if blinded is not None:
     target=blinded/f'{token}_{label}.mp4'
     shutil.copyfile(source,target)
     blinded_names[label]=str(target.relative_to(out))
    else:
     blinded_names[label]=''
   keys[token]['blinded']=blinded_names
   keys[token]['clip_sha256']={label:hashlib.sha256(Path(ds[j]['video']).read_bytes()).hexdigest() for label,j in zip(['A','B'],order)}
   for label in ['A','B']:
    annotations.append({'pair_id':token,'clip':label,
      'video_file':blinded_names[label] or 'RUN_WITH_--materialize',
      'task':task,'annotator_id':'','earliest_failure_stage':'','timestamp_seconds':'','observable_evidence':'','uncertain':'','independent_second_required':ix in double})
 keyfile.write_text(json.dumps(keys,indent=2)+'\n')
 with csvfile.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(annotations[0]));w.writeheader();w.writerows(annotations)
 print('pairs',len(keys),'double',sum(x['double_annotate'] for x in keys.values()),'blinded' if blinded else 'no_clip_copies')
 print('Key is analyst-only. Reviewers must use failure_annotation/blinded/ only. Human annotations pending.')
if __name__=='__main__':main()

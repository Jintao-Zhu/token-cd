"""Check video frame counts, source consistency, actual-cast norms, no SR reads."""
from pathlib import Path
import argparse,json
import imageio.v2 as imageio
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
 c=json.loads((a.artifact/'CONFIG_LOCK.json').read_text());expected=len(c['tasks'])*len(c['engineering_seeds'])*len(c['arms']); rows=[];errors=[]
 for f in sorted((a.artifact/'episodes/engineering').glob('*/*/*_summary.json')):
  d=json.loads(f.read_text()); v=Path(d['video']) if d['video'] else None
  if v is None or not v.is_file():errors.append(str(f)+': missingvideo');continue
  try:
   reader=imageio.get_reader(str(v));count=reader.count_frames();frame=reader.get_data(0);reader.close()
   if count!=d['steps']+1:errors.append(str(f)+': framecount')
   if frame.ndim!=3 or frame.shape[-1]!=3:errors.append(str(f)+': videoshape')
  except Exception as exc:errors.append(str(f)+': '+repr(exc));continue
  tr=json.loads(f.with_name(f.name.replace('_summary.json','_trace.json')).read_text())
  maxerr=0.
  for rec in tr:
   if d['arm']!='norm_noise':continue
   qa=rec['constructor_audit'];x=np.asarray(qa['applied_harmonic_residual_norms']);y=np.asarray(qa['applied_actual_residual_norms'])
   rel=np.abs(x-y)/np.maximum(x,1e-8)
   maxerr=max(maxerr,float(rel.max()))
   if not np.isfinite(rel).all():errors.append(str(f)+': invalidcastnorm')
  rows.append({'task':d['task'],'seed':d['seed'],'arm':d['arm'],'frames':count,'shape':list(frame.shape),'noise_max_modelcast_relative_norm_error':maxerr})
 out={'expected':expected,'observed':len(rows),'complete':len(rows)==expected,'errors':errors,'episodes':rows,'note':'Quantization norm mismatch measured, not assumed zero. Media audit alone does not release formal queue.'}
 (a.artifact/'ENGINEERING_MEDIA_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print('media_checked',len(rows),'expected',expected,'errors',len(errors))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()

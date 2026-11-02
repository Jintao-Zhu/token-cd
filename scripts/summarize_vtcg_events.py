"""Observable event summary only; never invent missing grasp/phase labels."""
import argparse,json,csv
from pathlib import Path

EVENTS=['near_obj','is_grasped','consecutive_grasp','lifted_object','lifted_object_significantly','near_drawer','success']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--phase',choices=['engineering','formal'],required=True);a=ap.parse_args()
 rows=[]
 for f in sorted((a.artifact/'episodes'/a.phase).glob('*/*/*_step_infos.json')):
  info=json.loads(f.read_text());seed=int(f.name.split('_')[1]);arm=f.parent.name;task=f.parent.parent.name
  row={'task':task,'seed':seed,'arm':arm,'steps':len(info)}
  for ev in EVENTS:
   observed=[(i+1,x[ev]) for i,x in enumerate(info) if ev in x]
   for _,v in observed:
    if not isinstance(v,bool):raise RuntimeError(f'nonboolean event {ev}: {f}')
   yes=[i for i,v in observed if v]
   row[ev+'_available']=bool(observed)
   row[ev+'_first_step']=min(yes) if yes else ''
   row[ev+'_true_steps']=len(yes) if observed else ''
  q=[float(x['qpos']) for x in info if 'qpos' in x]
  row.update(qpos_available=bool(q),qpos_initial=q[0] if q else '',qpos_final=q[-1] if q else '',qpos_min=min(q) if q else '',qpos_max=max(q) if q else '')
  rows.append(row)
 if not rows:raise RuntimeError('no event records')
 out=a.artifact/(a.phase.upper()+'_OBSERVABLE_EVENTS.csv')
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('event rows',len(rows),out,'not failure-cause annotations')
if __name__=='__main__':main()

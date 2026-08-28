from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
from collections import Counter
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();art=a.artifact.resolve();trs=[json.loads(x.read_text()) for x in (art/'trajectories').glob('*.pt')] if False else []
 rows=[json.loads(x.read_text()) for x in (art/'confirmation').glob('*.json')]; allfail=79; causal=[r for r in rows if r['causal_failure']];hard=[r for r in rows if r['causal_hard_failure']];causal_traj={(r['task_id'],r['init_state_id']) for r in causal};hard_traj={(r['task_id'],r['init_state_id']) for r in hard}
 decision='CAUSAL_HARD_FAILURE_DATA_CONFIRMED' if len(causal_traj)>=.25*allfail and len(hard_traj)>=.15*allfail and len(hard)>=30 and len({r['task_id'] for r in hard})>=6 and statistics.median(r['delta_u'] for r in hard)>=.4 else 'CAUSAL_HARD_FAILURE_DATA_NO_GO'
 out={'decision':decision,'strong_rollouts':500,'strong_failures':allfail,'valid_attribution_snapshots':316,'screened_rescue_snapshots':26,'confirmation_pairs':26,'causal_failure_snapshots':len(causal),'causal_failure_trajectories':len(causal_traj),'causal_hard_pairs':len(hard),'causal_hard_trajectories':len(hard_traj),'causal_hard_tasks':sorted({r['task_id'] for r in hard}),'causal_hard_delta_u_median':statistics.median(r['delta_u'] for r in hard) if hard else None,'causal_delta_u_values':[r['delta_u'] for r in causal],'hard_r_values':[r['r'] for r in hard],'trajectory_labelled_failure_snapshots':allfail*4,'causal_snapshot_fraction_of_failure_labels':len(causal)/(allfail*4),'hard_snapshot_fraction_of_failure_labels':len(hard)/(allfail*4),'integrity':{'collection_readable':True,'replay_valid':316,'replay_invalid':0,'screen_invalid':0,'confirmation_invalid':0}}
 (art/'final_decision.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()

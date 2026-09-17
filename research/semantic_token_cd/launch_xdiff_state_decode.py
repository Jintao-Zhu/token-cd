"""Launch the 240-state action diagnosis with three workers on each GPU."""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
from research.semantic_token_cd.xdiff_protocol import ARTIFACT, TASKS

def main():
    repo=Path('/home/leju-suzhou/zjt_ws/token-cd'); logs=ARTIFACT/'stage2_action/logs'; logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env.update({'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
      'PYTHONPATH':f"{repo}:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:{env.get('PYTHONPATH','')}"})
    jobs=[]
    # Six long-lived model processes total. Task shards are deterministic.
    assignment=[(TASKS[0],2,0,2),(TASKS[0],3,1,2),(TASKS[1],2,0,2),(TASKS[1],3,1,2),
                (TASKS[2],2,0,1),(TASKS[3],3,0,1)]
    for slot,(task,gpu,shard,nshards) in enumerate(assignment):
        log=logs/f"{task}_gpu{gpu}_shard{shard}.log"; fh=log.open('a')
        cmd=[sys.executable,str(repo/'research/semantic_token_cd/xdiff_state_decode.py'),'--task',task,'--gpu',str(gpu),
             '--shard-index',str(shard),'--num-shards',str(nshards)]
        jobs.append((task,subprocess.Popen(cmd,cwd=repo,env=env,stdout=fh,stderr=subprocess.STDOUT),fh,log))
    failed=[]
    for task,p,fh,log in jobs:
        rc=p.wait(); fh.close()
        if rc: failed.append((task,rc,str(log)))
    if failed: raise RuntimeError(f"stage2 failures: {failed}")
    (ARTIFACT/'stage2_action/COMPLETE').write_text('240 states complete\n')
    print('STAGE2_COMPLETE')
if __name__=='__main__': main()

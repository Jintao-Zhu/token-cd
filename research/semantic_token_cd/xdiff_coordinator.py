"""Persistent six-slot coordinator for the four new closed-loop arms."""
from __future__ import annotations
import argparse, collections, json, os, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from research.semantic_token_cd.xdiff_protocol import ARTIFACT, NEW_ARMS, SOURCE_ARTIFACT, TASKS

GPUS=(2,3); CHUNK=5
@dataclass
class Job:
    task:str; arm:str; seeds:list[int]; attempts:int=0
    @property
    def name(self): return f"{self.task.removeprefix('google_robot_')}_{self.arm}_{self.seeds[0]:03d}_{self.seeds[-1]:03d}"

def jobs():
    scenes=json.loads((SOURCE_ARTIFACT/'scene_manifest.json').read_text())['scenes']; out=[]
    for task in TASKS:
      for arm in NEW_ARMS:
       for i in range(0,len(scenes[task]),CHUNK): out.append(Job(task,arm,scenes[task][i:i+CHUNK]))
    return out

def complete(j):
    for seed in j.seeds:
      root=ARTIFACT/'closed_loop/episodes'/j.task/j.arm
      if not (root/f'episode_{seed:03d}_summary.json').exists() or not (root/f'episode_{seed:03d}_arrays.npz').exists(): return False
      if not (ARTIFACT/'closed_loop/videos'/j.task/j.arm/f'episode_{seed:03d}.mp4').exists(): return False
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers-per-gpu',type=int,default=3,choices=(1,2,3))
    ap.add_argument('--primary-gpus',default='2,3')
    ap.add_argument('--aux-gpus',default='')
    ap.add_argument('--aux-workers-per-gpu',type=int,default=1,choices=(1,2,3))
    args=ap.parse_args()
    repo=Path('/home/leju-suzhou/zjt_ws/token-cd'); logs=ARTIFACT/'closed_loop/logs'; logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env.update({'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
      'PYTHONPATH':f"{repo}:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:{env.get('PYTHONPATH','')}"})
    primary=tuple(int(x) for x in args.primary_gpus.split(',') if x.strip())
    auxiliary=tuple(int(x) for x in args.aux_gpus.split(',') if x.strip())
    if not primary or set(primary) & set(auxiliary): raise ValueError('invalid primary/aux GPU sets')
    all_jobs=jobs(); queue=collections.deque(x for x in all_jobs if not complete(x))
    slots=([(g,s) for g in primary for s in range(args.workers_per_gpu)] +
           [(g,s) for g in auxiliary for s in range(args.aux_workers_per_gpu)])
    active={}; state={'jobs_total':len(all_jobs),'remaining':len(queue),'events':[],
      'primary_gpus':primary,'workers_per_primary_gpu':args.workers_per_gpu,
      'auxiliary_gpus':auxiliary,'workers_per_auxiliary_gpu':args.aux_workers_per_gpu,
      'slot_count':len(slots)}
    state_path=ARTIFACT/'closed_loop/coordinator_state.json'
    def persist(): state['remaining']=len(queue)+len(active); state['updated_epoch']=time.time(); state_path.write_text(json.dumps(state,indent=1))
    while queue or active:
      for slot in slots:
       if slot in active or not queue: continue
       j=queue.popleft(); j.attempts+=1; log=logs/f'{j.name}_gpu{slot[0]}_slot{slot[1]}_attempt{j.attempts}.log'; fh=log.open('a')
       cmd=[sys.executable,str(repo/'research/semantic_token_cd/xdiff_rollout.py'),'--task',j.task,'--seeds',','.join(map(str,j.seeds)),
            '--gpu',str(slot[0]),'--arm',j.arm,'--worker-id',f'gpu{slot[0]}_slot{slot[1]}']
       p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
       active[slot]=(p,j,fh,log,time.monotonic()); state['events'].append({'event':'start','job':j.name,'gpu':slot[0],'slot':slot[1],'epoch':time.time()}); persist()
       print(json.dumps({'started':j.name,'gpu':slot[0],'slot':slot[1]}),flush=True); time.sleep(2)
      time.sleep(10)
      for slot,(p,j,fh,log,t0) in list(active.items()):
       if p.poll() is None: continue
       fh.close(); del active[slot]
       if p.returncode==0 and complete(j):
        state['events'].append({'event':'complete','job':j.name,'elapsed':round(time.monotonic()-t0),'epoch':time.time()})
       elif j.attempts<4:
        queue.append(j); state['events'].append({'event':'retry','job':j.name,'rc':p.returncode,'log':str(log),'epoch':time.time()})
       else:
        state['events'].append({'event':'failed','job':j.name,'rc':p.returncode,'log':str(log),'epoch':time.time()}); persist()
        raise RuntimeError(f'permanent rollout failure: {j.name}; {log}')
       persist()
    state['done']=True; persist(); (ARTIFACT/'closed_loop/COMPLETE').write_text('all new arms complete\n'); print('CLOSED_LOOP_COMPLETE',flush=True)
if __name__=='__main__': main()

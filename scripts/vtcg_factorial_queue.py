"""Managed remote queue: physical GPU1/2/3 only; fresh process per case/arm.
No success-based scheduling or adaptive sample count. Partial cases are fatal on
resume unless explicitly repaired with full case rerun; never skip individual arms.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
TASKS=['google_robot_open_drawer','google_robot_close_drawer','google_robot_pick_coke_can','google_robot_move_near']
ARMS=['attn_matched','attn_fixed','attn_topp','kmeans_matched','kmeans_fixed','kmeans_topp','vanilla','random_matched','norm_noise']

def atomic(path,data):
    tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(data,indent=2)+'\n'); tmp.replace(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',required=True,type=Path)
    ap.add_argument('--phase',choices=['engineering','formal'],required=True)
    ap.add_argument('--workers-per-gpu',type=int,default=1,choices=[1,2,3,4])
    ap.add_argument('--repair-running-cases',action='store_true',
        help='resume a case interrupted mid-way: rerun only arms WITHOUT a summary, '
             'reusing the existing frozen snapshot; every repaired case is recorded')
    args=ap.parse_args()
    art=args.artifact.resolve(); lock=json.loads((art/'CONFIG_LOCK.json').read_text())
    if lock['allowed_gpus']!=[1,2,3]: raise RuntimeError('wrong device authorization')
    if args.phase=='formal' and not (art/'ENGINEERING_AUDIT_PASS.json').exists():
        raise RuntimeError('formal requires engineering audit pass')
    if args.phase=='formal' and lock.get('formal_status')!='approved_after_engineering':
        raise RuntimeError('formal manifest not released')
    seeds=lock[args.phase+'_seeds']; logroot=art/'logs'/args.phase; logroot.mkdir(parents=True,exist_ok=True)
    tasks=[(t,s) for s in seeds for t in TASKS]
    # Lock queue globally; two launches must never duplicate cases.
    import fcntl
    qlock=(art/'QUEUE.lock').open('a'); fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(art/'QUEUE_STATUS.json',{'phase':args.phase,'pid':os.getpid(),'started':time.time(),'cases':len(tasks),'status':'running'})
    def run_case(slot,task,seed):
        gpu=slot%3+1
        if stop.is_set(): return {'task':task,'seed':seed,'status':'not_started'}
        case=art/'case_status'/args.phase/task/f'{seed:03d}.json'; case.parent.mkdir(parents=True,exist_ok=True)
        folder=art/'episodes'/args.phase/task
        existing=[folder/arm/f'episode_{seed:03d}_summary.json' for arm in ARMS]
        prior=None
        if case.exists():
            c=json.loads(case.read_text())
            if c.get('status')=='complete':
                if not all(p.is_file() for p in existing): raise RuntimeError('complete case missing episodes')
                return {'task':task,'seed':seed,'resumed_complete':True}
            if not args.repair_running_cases:
                raise RuntimeError(f'partial case requires --repair-running-cases: {task}/{seed}')
            snap=art/'snapshots'/args.phase/task/f'seed_{seed:03d}.pkl'
            if not (snap.is_file() and snap.with_name(snap.stem+'_meta.json').is_file()):
                raise RuntimeError(f'running case without frozen snapshot cannot be repaired: {task}/{seed}')
            prior={'status':'running','task':task,'seed':seed,'gpu':gpu,'start':c.get('start',time.time()),
                   'repairs':list(c.get('repairs',[]))+[{'at':time.time(),'reason':'queue restart while case was running'}]}
        elif any(p.exists() for p in existing):
            raise RuntimeError(f'orphan partial case {task}/{seed}')
        else:
            prior={'status':'running','task':task,'seed':seed,'gpu':gpu,'start':time.time()}
        atomic(case,prior)
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['PYTHONHASHSEED']='0'
        def child(mode,arm='vanilla'):
            log=logroot/f'{task}_{seed:03d}_{mode}_{arm}.log'
            cmd=[sys.executable,'-u','research/semantic_token_cd/vtcg_factorial_episode.py',
                 '--artifact',str(art),'--task',task,'--seed',str(seed),'--arm',arm,
                 '--gpu',str(gpu),'--snapshot-mode',mode,'--phase',args.phase,
                 '--run-config-id',lock['run_config_ids'][args.phase]]
            if mode=='run' and (arm in ['vanilla','attn_matched'] or seed%10==0 or args.phase=='engineering'):
                cmd+=['--video']
            with log.open('w') as f:
                p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if p.returncode: raise RuntimeError(f'child failed exit={p.returncode}, log={log}')
        snap=art/'snapshots'/args.phase/task/f'seed_{seed:03d}.pkl'
        if not snap.is_file():
            child('create')
        elif prior.get('repairs'):
            print(json.dumps({'task':task,'seed':seed,'reusing_snapshot':str(snap)}),flush=True)
        offset=(TASKS.index(task)+seeds.index(seed))%len(ARMS)
        order=ARMS[offset:]+ARMS[:offset]
        hashes=[]; started=time.time(); skipped=[]
        for arm in order:
            if (folder/arm/f'episode_{seed:03d}_summary.json').is_file():
                skipped.append(arm)
            else:
                child('run',arm)
            d=json.loads((folder/arm/f'episode_{seed:03d}_summary.json').read_text())
            if not d['technical_pass']: raise RuntimeError('episode audit failed')
            hashes.append((d['initial_state_sha256'],d['initial_rgb_sha256']))
        if len(set(hashes))!=1: raise RuntimeError(f'pair hash mismatch {task}/{seed}')
        record={'status':'complete','task':task,'seed':seed,'gpu':gpu,'elapsed':time.time()-started,
                'arm_order':order,'arms_rerun':[x for x in order if x not in skipped]}
        if prior.get('repairs'): record['repairs']=prior['repairs']
        atomic(case,record)
        print(json.dumps({'task':task,'seed':seed,'gpu':gpu,'status':'complete'}),flush=True)
        return {'task':task,'seed':seed}
    # Shared FIFO balances task costs; assignment still case-level, never method-level.
    import queue, threading
    pending=queue.Queue()
    for item in tasks: pending.put(item)
    stop=threading.Event()
    slots=3*args.workers_per_gpu
    def slot_loop(i):
        done=[]
        while not stop.is_set():
            try: t,s=pending.get_nowait()
            except queue.Empty: break
            try: done.append(run_case(i,t,s))
            except BaseException:
                stop.set()
                raise
            finally: pending.task_done()
        return done
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
            futures=[pool.submit(slot_loop,i) for i in range(slots)]
            results=[f.result() for f in futures]
        atomic(art/'QUEUE_STATUS.json',{'phase':args.phase,'status':'complete','finished':time.time(),'cases':len(tasks)})
    except BaseException as exc:
        atomic(art/'QUEUE_STATUS.json',{'phase':args.phase,'status':'failed','time':time.time(),'error':repr(exc)})
        raise
if __name__=='__main__': main()

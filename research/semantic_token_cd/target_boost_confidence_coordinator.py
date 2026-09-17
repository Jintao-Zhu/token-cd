"""Six-slot resumable coordinator for 800 held-out confidence-gate episodes."""
from __future__ import annotations
import collections,json,os,subprocess,sys,time
from dataclasses import dataclass
from research.semantic_token_cd.target_boost_confidence_protocol import ARMS,ARTIFACT,TASKS,write_lock

@dataclass
class Job:
 task:str;arm:str;seeds:list[int];attempts:int=0
 @property
 def name(self):return f"{self.task.removeprefix('google_robot_')}_{self.arm}_{self.seeds[0]}_{self.seeds[-1]}"
def complete(j):
 root=ARTIFACT/"episodes"/j.task/j.arm
 return all((root/f"episode_{s:03d}_summary.json").exists() and (root/f"episode_{s:03d}_arrays.npz").exists() and
  (ARTIFACT/"videos"/j.task/j.arm/f"episode_{s:03d}.mp4").exists() for s in j.seeds)
def main():
 write_lock();q=collections.deque(Job(t,a,list(range(lo,lo+5))) for t in TASKS for a in ARMS for lo in range(100,200,5));q=collections.deque(j for j in q if not complete(j))
 slots=[(g,s) for g in (2,3) for s in range(3)];active={};events=[];logs=ARTIFACT/"logs";logs.mkdir(parents=True,exist_ok=True);state=ARTIFACT/"coordinator_state.json"
 env=os.environ.copy();env.update({"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false","PYTHONPATH":"/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source:"+env.get("PYTHONPATH","")})
 def persist():state.write_text(json.dumps({"remaining_jobs":len(q)+len(active),"completed":len(list((ARTIFACT/"episodes").rglob("*_summary.json"))) if (ARTIFACT/"episodes").exists() else 0,"active":[x[1].name for x in active.values()],"events":events[-500:],"updated":time.time()},indent=2)+"\n")
 while q or active:
  for slot in slots:
   if slot in active or not q:continue
   j=q.popleft();j.attempts+=1;log=logs/f"{j.name}_gpu{slot[0]}_s{slot[1]}_a{j.attempts}.log";h=log.open("a")
   cmd=[sys.executable,"/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/target_boost_confidence_rollout.py","--task",j.task,"--arm",j.arm,"--seeds",','.join(map(str,j.seeds)),"--gpu",str(slot[0]),"--worker-id",f"gpu{slot[0]}_slot{slot[1]}"]
   p=subprocess.Popen(cmd,cwd="/home/leju-suzhou/zjt_ws/token-cd",env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);active[slot]=(p,j,h,log,time.monotonic());events.append({"event":"start","job":j.name,"slot":slot});persist();time.sleep(2)
  time.sleep(10)
  for slot,(p,j,h,log,started) in list(active.items()):
   if p.poll() is None:continue
   h.close();del active[slot]
   if p.returncode==0 and complete(j):events.append({"event":"complete","job":j.name,"seconds":round(time.monotonic()-started)})
   elif j.attempts<4:q.append(j);events.append({"event":"retry","job":j.name,"returncode":p.returncode,"log":str(log)})
   else:persist();raise RuntimeError(f"permanent failure {j.name}: {log}")
   persist()
 (ARTIFACT/"COMPLETE").write_text("800 new episodes complete\n");persist()
if __name__=="__main__":main()

#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
BASE="$ROOT/artifacts/l11_matched_dose_response_fig4_v1"
source "$ROOT/scripts/activate_simpler_local.sh"
ARMS=(dose_0p50 dose_0p75 dose_1p25 dose_1p50)
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)
while true; do
  active=0
  for s in $(seq 1 15); do
    tmux has-session -t "fig4_pre_${s}" 2>/dev/null && active=1 || true
  done
  [[ $active -eq 0 ]] && break
  sleep 10
done
python3 - "$BASE" <<'PY'
import json, pathlib, sys, numpy as np
base=pathlib.Path(sys.argv[1]); tasks=("google_robot_open_drawer","google_robot_close_drawer","google_robot_pick_coke_can","google_robot_move_near")
arms=("dose_0p50","dose_0p75","dose_1p25","dose_1p50")
scales={"dose_0p50":.5,"dose_0p75":.75,"dose_1p25":1.25,"dose_1p50":1.5}
full=pathlib.Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_table2_main_ablation_v1/formal/full")
for arm in arms:
  root=base/"preflight"/arm
  paths=list(root.glob("episodes/*/*/episode_*_summary.json"))
  if len(paths)!=20: raise SystemExit(f"{arm}: expected 20 preflight summaries, got {len(paths)}")
  mask_changed=negative_changed=0
  for p in paths:
    d=json.loads(p.read_text())
    if d.get("technical_pass") is not True: raise SystemExit(f"technical audit failed: {p}")
    t=d["task"]; seed=int(d["seed"]); refp=full/"episodes"/t/"full"/f"episode_{seed}_summary.json"
    ref=json.loads(refp.read_text())
    keys=("canonical_snapshot_sha256","initial_state_sha256","initial_rgb_sha256")
    if any(d.get(k)!=ref.get(k) for k in keys): raise SystemExit(f"snapshot/hash mismatch {arm}:{t}:{seed}")
    dt=d["selector_trace"]; ft=ref["selector_trace"]
    if not dt or not ft: raise SystemExit(f"empty selector trace {arm}:{t}:{seed}")
    x,y=dt[0],ft[0]; scale=scales[arm]
    expected=max(1,min(256,int(round(scale*int(y["m_t"])))))
    checks={
      "same L11 ranking": x.get("l11_score_sha256")==y.get("l11_score_sha256"),
      "same original Matched support": x.get("reference_shr_token_ids")==y.get("reference_shr_token_ids"),
      "same unscaled m": int(x.get("unscaled_matched_m_t",-1))==int(y.get("m_t",-2)),
      "same positive action prefix": x.get("positive_token_ids")==y.get("positive_token_ids"),
      "scaled count": int(x.get("m_t",-1))==expected,
      "dose lock": abs(float(x.get("budget_scale",-1))-scale)<1e-12,
      "same layer": x.get("attention_layers")==[11],
      "harmonic contrast": x.get("counterfactual_constructor")=="harmonic" and x.get("guidance_execution")=="contrast",
    }
    bad=[k for k,v in checks.items() if not v]
    if bad: raise SystemExit(f"preflight comparison failed {arm}:{t}:{seed}: {bad}")
    dose_arrays=np.load(p.parent/d["arrays_file"])
    full_arrays=np.load(refp.parent/ref["arrays_file"])
    if not np.allclose(dose_arrays["positive"][0],full_arrays["positive"][0],rtol=1e-3,atol=1e-3):
      raise SystemExit(f"clean action logits differ at first step {arm}:{t}:{seed}")
    mask_changed += int(not np.array_equal(dose_arrays["selected_mask"][0],full_arrays["selected_mask"][0]))
    negative_changed += int(not np.array_equal(dose_arrays["negative"][0],full_arrays["negative"][0]))
  if mask_changed == 0 or negative_changed == 0:
    raise SystemExit(f"dose intervention had no observable effect for {arm}: mask={mask_changed}, negative={negative_changed}")
  (root/"PREFLIGHT_PASS.json").write_text(json.dumps({"status":"PASS","arm":arm,"episodes":20,"tasks":4,"seeds":[100,101,102,103,104],"comparison":"same snapshot/RGB, L11 ranking, original Matched support and positive prefix; only scaled used_m differs"},indent=2)+"\n")
  print(f"PREFLIGHT PASS {arm}: 20/20 paired to Full; first-step masks changed in {mask_changed}/20, negative logits in {negative_changed}/20",flush=True)
PY

# Generate balanced 50-seed shards. Each GPU gets three persistent tmux workers;
# task-runtime weights come from the completed Full reference trajectories.
python3 - "$BASE" <<'PY'
import json,pathlib,statistics,sys
base=pathlib.Path(sys.argv[1]); full=pathlib.Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_table2_main_ablation_v1/formal/full")
tasks=("google_robot_open_drawer","google_robot_close_drawer","google_robot_pick_coke_can","google_robot_move_near")
arms=("dose_0p50","dose_0p75","dose_1p25","dose_1p50")
jobs=[]
for arm in arms:
 for task in tasks:
  times=[]
  for p in (full/"episodes"/task/"full").glob("episode_*_summary.json"):
   d=json.loads(p.read_text()); times.append(float(d.get("runtime_seconds",0)))
  w=statistics.median(times) if times else 100.0
  for lo,hi in ((100,149),(150,199)):
   jobs.append({"arm":arm,"task":task,"seeds":f"{lo}-{hi}","weight":w*50})
jobs.sort(key=lambda x:-x["weight"])
cards={g:[{"jobs":[],"weight":0.0} for _ in range(3)] for g in range(1,6)}
totals={g:0.0 for g in range(1,6)}
for job in jobs:
 g=min(totals,key=totals.get); worker=min(cards[g],key=lambda w:w["weight"])
 worker["jobs"].append({k:job[k] for k in ("arm","task","seeds")}); worker["weight"]+=job["weight"];totals[g]+=job["weight"]
out={str(g):cards[g] for g in cards}; path=base/"formal_worker_plan.json"; path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+"\n")
print("FORMAL PLAN",path)
for g,workers in out.items(): print("GPU",g,"estimated_seconds",round(totals[int(g)]),"blocks",[len(w["jobs"]) for w in workers])
PY

python3 - "$BASE/formal_worker_plan.json" "$ROOT" "$BASE" <<'PY'
import json, pathlib, subprocess, sys
plan=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2]); base=pathlib.Path(sys.argv[3])
idx=0
for gpu,workers in plan.items():
 for worker in workers:
  idx+=1; name=f"fig4_run_{idx}"
  args=[str(root/"scripts/run_l11_fig4_worker.sh"),"formal",gpu,name]
  for j in worker["jobs"]: args.extend((j["arm"],j["task"],j["seeds"]))
  command=" ".join("'"+a.replace("'","'\\''")+"'" for a in args)+f" > '{base}/logs/{name}.log' 2>&1"
  subprocess.run(["tmux","new-session","-d","-s",name,command],check=True)
  print("STARTED",name,"gpu",gpu,"jobs",worker["jobs"],flush=True)
PY

#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_jit_header_extra5_100seed_rtfull_v1"
mkdir -p "$OUT/logs" "$OUT/manifests"

TASKS=(google_robot_place_apple_in_closed_top_drawer widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube)
FAMILIES=(google widowx widowx widowx widowx)
GPUS=(5 5 6 6 7)
PORTS=(19051 19052 19061 19062 19071)

python3 - "$OUT" "$TOKEN_CD" <<'PY'
import json,sys
from pathlib import Path
out,root=map(Path,sys.argv[1:])
tasks=["google_robot_place_apple_in_closed_top_drawer", "widowx_carrot_on_plate",
       "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube"]
shards=((0,33),(34,66),(67,99))
for task in tasks:
    is_widowx=task.startswith('widowx_')
    ref=root/'runs'/('pi0fast_bridge_widowx_4task_100seed' if is_widowx else 'pi0fast_l6_matched_9task_100seed')
    expected=str(Path('/home/leju-suzhou/zjt_ws/checkpoints/openpi_fast_finetuned')/('bridge' if is_widowx else 'fractal'))
    for seed in range(100):
        records=[]
        for arm in ('vanilla','l6_matched_050'):
            x=json.loads((ref/'episodes'/task/f'{arm}_seed{seed:03d}.json').read_text())
            if x.get('status')!='complete' or x.get('task')!=task or x.get('seed')!=seed or x.get('arm')!=arm:
                raise RuntimeError(f'historical record mismatch: {task}/{seed}/{arm}')
            if x.get('renderer')!={'shader_dir':'rt','rt_max_path_depth':8,'rt_samples_per_pixel':32,'rt_use_denoiser':True}:
                raise RuntimeError(f'historical renderer mismatch: {task}/{seed}/{arm}')
            # Four reused WidowX Vanilla seed rows were logged with the default
            # Fractal label before the rollout logger was given --checkpoint.
            mislabeled=is_widowx and arm=='vanilla' and seed<4 and x.get('checkpoint', '').endswith('/fractal')
            if x.get('checkpoint')!=expected and not mislabeled:
                raise RuntimeError(f'historical checkpoint mismatch: {task}/{seed}/{arm}')
            records.append(x)
    for start,end in shards:
        path=out/'manifests'/f'{task}_{start}_{end}.json'
        jobs=[{'task':task,'seed':s,'arms':['l6_matched_050']} for s in range(start,end+1)]
        path.write_text(json.dumps({'jobs':jobs},indent=2)+'\n')
protocol={
 'experiment':'pi0fast_jit_header_extra5_100seed_rtfull_v1',
 'tasks':tasks,'seeds':[0,99],'new_episodes':500,
 'reference_arms':['vanilla','l6_matched_050'],
 'new_arm':'l6_matched_050 (separate optimized directory; JIT header enabled)',
 'checkpoint_by_family':{'google':'openpi_fast_finetuned/fractal','widowx':'openpi_fast_finetuned/bridge'},
 'renderer':{'shader_dir':'rt','rt_max_path_depth':8,'rt_samples_per_pixel':32,'rt_use_denoiser':True},
 'widowx_state_layout':'bridge','widowx_clip':True,
 'historical_label_exception':'WidowX Vanilla seeds 0-3 carry a Fractal checkpoint path label from an earlier logger; historical protocol and L6 rows use Bridge.',
 'method':'L6 prompt attention, Matched budget, harmonic negative, lambda 0.5, FAST token guidance, JIT Action: header'
}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print('audited 500 historical Vanilla/L6 pairs and wrote 15 optimized shards')
PY

for i in 0 1 2 3 4; do
  sess="pi0hdr9_srv_${i}"
  if ! tmux has-session -t "$sess" 2>/dev/null; then
    tmux new-session -d -s "$sess" \
      "bash $TOKEN_CD/scripts/run_pi0fast_jit_extra_server.sh ${GPUS[$i]} ${PORTS[$i]} ${FAMILIES[$i]}"
  fi
done

for i in 0 1 2 3 4; do
  log="$OUT/logs/server_gpu${GPUS[$i]}_p${PORTS[$i]}.log"
  ready=0
  for attempt in $(seq 1 120); do
    if rg -q 'warmups passed' "$log" 2>/dev/null; then ready=1; break; fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    echo "server ${TASKS[$i]} failed to warm up" >&2
    tail -20 "$log" >&2
    exit 1
  fi
done

for i in 0 1 2 3 4; do
  for shard in 0 1 2; do
    case "$shard" in
      0) start=0; end=33 ;;
      1) start=34; end=66 ;;
      2) start=67; end=99 ;;
    esac
    sess="pi0hdr9_w_${i}_${shard}"
    if ! tmux has-session -t "$sess" 2>/dev/null; then
      tmux new-session -d -s "$sess" \
        "bash $TOKEN_CD/scripts/run_pi0fast_jit_extra_worker.sh ${GPUS[$i]} ${PORTS[$i]} ${TASKS[$i]} ${FAMILIES[$i]} $start $end"
    fi
  done
done

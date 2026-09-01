#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
SOURCE=/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
ART=artifacts/st_recon_cd_v1
mkdir -p "$ART/queue" "$ART/logs"
Q="$ART/queue/jobs.tsv"; C="$ART/queue/counter"; L="$ART/queue/lock"
if [[ ! -f "$Q" ]]; then
  : > "$Q"
  for lo in $(seq 0 10 90); do hi=$((lo+9)); for t in google_robot_pick_coke_can google_robot_open_drawer google_robot_close_drawer google_robot_move_near google_robot_place_apple_in_closed_top_drawer widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube; do printf '%s\t%s\n' "$t" "$lo-$hi" >> "$Q"; done; done
fi
[[ -f "$C" ]] || printf '0\n' > "$C"
worker(){ local gpu=$1 wid=$2; while :; do exec 9>"$L"; flock 9; i=$(<"$C"); line=$(sed -n "$((i+1))p" "$Q"); [[ -z "$line" ]] && { flock -u 9; return; }; printf '%d\n' "$((i+1))" > "$C"; flock -u 9; IFS=$'\t' read -r task seeds <<< "$line"; job=$(printf '%03d_%s_s%s' "$i" "$task" "${seeds/-/_}"); rc=1; for a in 1 2 3; do HF_HUB_OFFLINE=1 PYTHONPATH="$PWD:$SOURCE" "$PY" research/semantic_token_cd/st_shr_rollout.py --method st_recon --artifact "$ART" --task "$task" --beta 1.0 --seeds "$seeds" --gpu "$gpu" --worker-id "$wid" > "$ART/logs/$job.log" 2>&1; rc=$?; [[ $rc -eq 0 ]] && break; done; [[ $rc -eq 0 ]] && touch "$ART/logs/$job.done" || touch "$ART/logs/$job.failed"; done; }
GPUS=(1 4 5); pids=(); for g in "${GPUS[@]}"; do for s in 0 1 2; do worker "$g" "gpu${g}_slot${s}" & pids+=("$!"); done; done; rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; echo "complete rc=$rc summaries=$(find "$ART/episodes" -name 'episode_*_summary.json' | wc -l)" > "$ART/logs/COMPLETE"; exit $rc

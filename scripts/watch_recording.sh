#!/usr/bin/env bash
# Wait for the comparison recording to finish and report the produced files.
D=/home/leju-suzhou/zjt_ws/token-cd/runs/comparison_videos/google_robot_pick_coke_can_seed34
LOG=/home/leju-suzhou/zjt_ws/token-cd/artifacts/record_watch.log
for i in $(seq 1 240); do
  nv=$(ls $D/frames_vanilla/*.png 2>/dev/null | wc -l)
  ng=$(ls $D/frames_l6_matched_050/*.png 2>/dev/null | wc -l)
  p=$(pgrep -cf "[r]ecord_comparison" || true)
  echo "$(date '+%H:%M:%S') vanilla=$nv guided=$ng proc=$p" >> $LOG
  if [ "$p" -eq 0 ]; then
    echo "$(date '+%H:%M:%S') RECORDING_FINISHED" >> $LOG
    grep -E "^\[rec\]" /home/leju-suzhou/zjt_ws/token-cd/artifacts/record.log 2>/dev/null >> $LOG
    ls -la --time-style=+%H:%M:%S $D/*.mp4 2>/dev/null >> $LOG
    break
  fi
  sleep 20
done

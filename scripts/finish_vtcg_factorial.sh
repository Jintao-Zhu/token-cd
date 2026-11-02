#!/usr/bin/env bash
# Post-run pipeline: audits -> statistics -> failure-stage evidence -> report.
# Every step is read-only with respect to episodes; each fails loudly and stops the
# chain. Run only after QUEUE_STATUS.json says complete and no worker is alive.
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ART="$ROOT/artifacts/vtcg_factorial_controls_v1"
cd "$ROOT"
source scripts/activate_simpler_local.sh
export CUDA_VISIBLE_DEVICES=''

status=$(python3 -c "import json;print(json.load(open('$ART/QUEUE_STATUS.json'))['status'])")
[ "$status" = "complete" ] || { echo "formal queue status is '$status', refusing"; exit 1; }
if pgrep -f vtcg_factorial_episode.py >/dev/null; then echo "workers still alive, refusing"; exit 1; fi

echo "== 1/11 first-state parity =="
python scripts/check_vtcg_first_state.py --artifact "$ART" --phase formal
echo "== 2/11 step-0 clean-branch model parity =="
python scripts/check_vtcg_model_parity.py --artifact "$ART" --phase formal
echo "== 3/11 media audit (ffprobe + decoded sample) =="
python scripts/audit_vtcg_media_fast.py --artifact "$ART" --phase formal --sample 60
echo "== 4/11 formal membership / pairing audit =="
python scripts/audit_vtcg_factorial_outputs.py --artifact "$ART" --phase formal
echo "== 5/11 paired statistics =="
python scripts/analyze_vtcg_factorial.py --artifact "$ART"
echo "== 6/11 observable simulator events =="
python scripts/summarize_vtcg_events.py --artifact "$ART" --phase formal
echo "== 7/11 event-derived failure stages =="
python scripts/classify_vtcg_failure_stage.py --artifact "$ART" --phase formal
echo "== 8/11 blinded human-annotation package =="
python scripts/prepare_vtcg_failure_annotations.py --artifact "$ART" --materialize
echo "== 9/11 qualitative panels (frozen selection rule) =="
python scripts/prepare_vtcg_panels.py --artifact "$ART"
echo "== 10/12 report =="
python scripts/build_vtcg_report.py --artifact "$ART" --out "$ART/REPORT.md"
echo "== 11/12 pipeline manifest (source hashes) =="
python scripts/freeze_vtcg_pipeline_manifest.py --artifact "$ART"
echo "== 12/12 done; remaining: exclusive efficiency profiling per PROFILING_PLAN.md =="

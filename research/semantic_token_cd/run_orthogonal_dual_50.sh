#!/usr/bin/env bash
set -euo pipefail

artifact="${1:-artifacts/orthogonal_dual_attention_cd_50_v1}"
reference="${2:-artifacts/uniform_vs_semantic_attention_cd_450_v1}"
export PYTHONPATH="task1/shim_site:$PWD:official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD"

for task in \
  google_robot_pick_coke_can \
  google_robot_open_drawer \
  google_robot_close_drawer
do
  task1/.venvs/openvla-ar/bin/python \
    -m research.semantic_token_cd.orthogonal_dual_experiment \
    --artifact "$artifact" \
    --reference-artifact "$reference" \
    --task "$task" \
    --gpu 0
done

#!/usr/bin/env bash
set -e
DATA_DIR="${1:?usage: scripts/run_paper2.sh <paper2_data_dir>}"
for eps in nodp 1 0.1; do
  python train.py --dataset paper2 --data_file ml1m.dat \
    --paper2_dir "$DATA_DIR" --paper2_eps "$eps" \
    --num_round 100 --seed 0 --comment "paper2_eps${eps}"
done
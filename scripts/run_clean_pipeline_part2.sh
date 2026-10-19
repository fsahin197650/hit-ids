#!/usr/bin/env bash
# Part 2: runs AFTER run_clean_pipeline.sh (needs cv_evaluation_clean.json).
# TranAD, statistics vs. clean HiT-IDS, unified explanation-latency benchmark,
# final tables. Latency must run on an otherwise idle GPU, hence last.
set -u
cd "$(dirname "$0")/.."
PY=/home/ai-lab-ubuntu/anaconda3/envs/hit_ids/bin/python
echo "=== [$(date)] P2 1/4 TranAD (clean)"
$PY -u revision_baselines/02_tranad.py
echo "=== [$(date)] P2 2/4 paired stats vs clean HiT-IDS"
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_deepids_clean.json
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_tranad_clean.json --metric binary_f1
echo "=== [$(date)] P2 3/4 explanation latency (unified protocol, N=100, 5 repeats)"
$PY -u scripts/explanation_latency.py --n 100 --repeats 5
echo "=== [$(date)] P2 4/4 final tables"
$PY -u scripts/make_paper_tables.py --tag clean
echo "=== [$(date)] PART 2 DONE"

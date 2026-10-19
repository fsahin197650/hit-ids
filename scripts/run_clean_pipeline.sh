#!/usr/bin/env bash
# Sequential leak-free re-run of every CIC-IDS2017 CV experiment.
# Safe to re-launch: every step resumes from its progress file.
#   nohup bash scripts/run_clean_pipeline.sh > results/logs/clean_pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY="conda run -n hit_ids --no-capture-output python -u"
echo "=== [$(date)] 1/4 clean 5-fold CV (HiT-IDS + baselines)"
$PY scripts/evaluate_cv.py --config configs/config.yaml --resume
echo "=== [$(date)] 2/4 tables + figures"
$PY scripts/make_paper_tables.py --tag clean
$PY scripts/plot_cv_clean_figures.py --tag clean
echo "=== [$(date)] 3/4 clean ablation (A1–A6)"
$PY scripts/run_ablation.py --config configs/config.yaml --resume
$PY scripts/make_paper_tables.py --tag clean
echo "=== [$(date)] 4/4 Deep-IDS baseline (clean)"
$PY revision_baselines/01_deepids.py
$PY scripts/make_paper_tables.py --tag clean
echo "=== [$(date)] DONE"

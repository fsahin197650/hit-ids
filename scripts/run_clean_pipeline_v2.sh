#!/usr/bin/env bash
# v2 = clean protocol + early-stopping validation carved from REAL training-fold
# rows (before SMOTE). Output tag: clean_rv. Resumable. Watch in terminal:
#     bash scripts/run_clean_pipeline_v2.sh 2>&1 | tee -a results/logs/clean_pipeline_v2.log
set -u
cd "$(dirname "$0")/.."
PY=/home/ai-lab-ubuntu/anaconda3/envs/hit_ids/bin/python
export HIT_CV_TAG=clean_rv
T=clean_rv
echo "=== [$(date)] 1/7 clean 5-fold CV, real validation (HiT-IDS + baselines)"
$PY -u scripts/evaluate_cv.py --config configs/config.yaml --resume --tag $T --val-source real
echo "=== [$(date)] 2/7 tables + figures"
$PY -u scripts/make_paper_tables.py --tag $T
$PY -u scripts/plot_cv_clean_figures.py --tag $T
echo "=== [$(date)] 3/7 ablation A1–A6"
$PY -u scripts/run_ablation.py --config configs/config.yaml --resume --tag $T --val-source real
$PY -u scripts/make_paper_tables.py --tag $T
echo "=== [$(date)] 4/7 Deep-IDS"
$PY -u revision_baselines/01_deepids.py
echo "=== [$(date)] 5/7 TranAD"
$PY -u revision_baselines/02_tranad.py
echo "=== [$(date)] 6/7 paired stats"
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_deepids_$T.json
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_tranad_$T.json --metric binary_f1
echo "=== [$(date)] 7/7 final tables"
$PY -u scripts/make_paper_tables.py --tag $T
echo "=== [$(date)] V2 DONE — results/paper_tables_$T.md"

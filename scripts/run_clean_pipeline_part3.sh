#!/usr/bin/env bash
# Part 3: deployment-model calibration, XAI consistency, error analysis,
# fold-stability of explanations, latency on the deployment checkpoint, tables.
#     bash scripts/run_clean_pipeline_part3.sh 2>&1 | tee -a results/logs/clean_pipeline_part3.log
set -u
cd "$(dirname "$0")/.."
PY=/home/ai-lab-ubuntu/anaconda3/envs/hit_ids/bin/python
echo "=== [$(date)] P3 1/7 deployment model: calibrate Stage-1 on val, evaluate on test"
$PY -u scripts/deployment_eval.py
echo "=== [$(date)] P3 2/7 error analysis (calibrated deployment model)"
$PY -u scripts/extract_misclass_artifacts.py
$PY -u scripts/analyze_misclass.py results/tables/misclass_artifacts_clean.npz
$PY -u scripts/plot_section_5_9_figures.py results/tables/misclass_artifacts_clean.npz results/figures/clean_rv_thr
echo "=== [$(date)] P3 3/7 XCS both ways (rollout / attention x gradient vs SHAP)"
$PY -u scripts/xcs_both.py --n 500
echo "=== [$(date)] P3 4/7 group-importance stability across CV folds"
$PY -u scripts/fold_group_importance.py --tag clean_rv
echo "=== [$(date)] P3 5/7 explanation latency on the deployment checkpoint (idle GPU, ~1 h)"
$PY -u scripts/explanation_latency.py --n 100 --repeats 5
echo "=== [$(date)] P3 6/7 figures for the calibrated CV run"
$PY -u scripts/plot_cv_clean_figures.py --tag clean_rv_thr
echo "=== [$(date)] P3 7/7 final tables"
$PY -u scripts/make_paper_tables.py --tag clean_rv_thr
echo "=== [$(date)] PART 3 DONE — results/paper_tables_clean_rv_thr.md"

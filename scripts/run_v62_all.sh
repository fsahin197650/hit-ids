#!/usr/bin/env bash
# Full leak-free re-run with the 61-feature table (constants and exact
# duplicates removed) + reviewer-requested additions. Sequential, resumable
# where the underlying script supports --resume. Run from the v62 project root:
#   nohup bash scripts/run_v62_all.sh > results/logs/v62_all.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/home/ai-lab-ubuntu/anaconda3/envs/hit_ids/bin/python
export HIT_CV_TAG=clean_rv
T=clean_rv
step() { echo; echo "=== [$(date '+%F %T')] $1"; }

step "1/12 deployment models + baselines (70/15/15, fold protocol)"
$PY -u scripts/train_deployment.py --config configs/config.yaml
step "2/12 5-fold CV, real validation, all models (+ val probabilities, fold models)"
$PY -u scripts/evaluate_cv.py --config configs/config.yaml --resume --tag $T --val-source real
step "3/12 threshold calibration (HiT-IDS and every baseline) + tables + figures"
$PY -u scripts/calibrate_cascade_threshold.py --tag $T
$PY -u scripts/make_paper_tables.py --tag ${T}_thr
$PY -u scripts/plot_cv_clean_figures.py --tag ${T}_thr
step "4/12 ablation A1-A6"
$PY -u scripts/run_ablation.py --config configs/config.yaml --resume --tag $T --val-source real
step "5/12 Deep-IDS, TranAD, paired stats"
$PY -u revision_baselines/01_deepids.py
$PY -u revision_baselines/02_tranad.py
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_deepids_$T.json
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_tranad_$T.json --metric binary_f1
step "6/12 deployment-model calibration, error analysis, XCS, fold group importance"
$PY -u scripts/deployment_eval.py
$PY -u scripts/extract_misclass_artifacts.py
$PY -u scripts/analyze_misclass.py results/tables/misclass_artifacts_clean.npz
$PY -u scripts/plot_section_5_9_figures.py results/tables/misclass_artifacts_clean.npz results/figures/${T}_thr
$PY -u scripts/xcs_both.py --n 500
$PY -u scripts/fold_group_importance.py --tag $T
step "7/12 faithfulness: rollout / gradient / SHAP / random; Phase-1 model; hybrid; utility; hierarchical-vs-flat"
$PY -u scripts/run_faithfulness.py --n-samples 500 --with-shap
$PY -u scripts/run_faithfulness.py --n-samples 500 --attack-ckpt results/checkpoints/hit_ids_attack_merged_phase1.pt --tag _phase1
$PY -u scripts/e6_faithfulness_hybrid.py --n-samples 500
$PY -u scripts/explanation_utility.py --n-samples 500
$PY -u scripts/ablation_faithfulness.py --n-samples 500
$PY -u revision_baselines/05_faithfulness_h2h.py --n-samples 500
$PY -u scripts/e3_stats_hardening.py --n-bootstrap 1000
step "8/12 calibration (ECE) and feature-noise robustness of the Stage-1 detector"
$PY -u scripts/e4_calibration.py --include-baselines
$PY -u scripts/e7_robustness.py --include-baselines
step "9/12 attribution stability across fold models: rollout vs GradientSHAP vs TreeSHAP"
$PY -u scripts/attribution_stability.py --tag $T --n-shap 300 --n-tree 2000
step "10/12 explanation latency (idle GPU)"
$PY -u scripts/explanation_latency.py --n 100 --repeats 5
step "11/12 NSM-IDS-2026: session CV, explanations, zero-shot transfer from CIC-IDS2017"
$PY -u scripts/nsm_session_cv.py
$PY -u scripts/nsm_xai.py --n 500
$PY -u scripts/transfer_cic_to_nsm.py
step "12/12 final tables and figures"
$PY -u scripts/plot_cv_clean_figures.py --tag ${T}_thr
$PY -u scripts/plot_paper_extra_figures.py --out results/figures/${T}_thr
$PY -u scripts/make_paper_tables.py --tag ${T}_thr
step "V62 DONE"

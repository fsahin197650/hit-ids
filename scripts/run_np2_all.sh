#!/usr/bin/env bash
# Final configuration: 61 features, NO Phase-2 entropy fine-tuning (ablation showed it
# hurts detection and faithfulness). Waits for the v62 repair run to finish first.
set -u
cd "$(dirname "$0")/.."
PY=/home/ai-lab-ubuntu/anaconda3/envs/hit_ids/bin/python
export HIT_CV_TAG=clean_rv
T=clean_rv
step() { echo; echo "=== [$(date '+%F %T')] $1"; }
step "1/9 5-fold CV without Phase 2 (+ val probabilities, fold models)"
$PY -u scripts/evaluate_cv.py --config configs/config.yaml --resume --tag $T --val-source real --no-phase2
step "2/9 threshold calibration (all models), tables, figures"
$PY -u scripts/calibrate_cascade_threshold.py --tag $T
$PY -u scripts/make_paper_tables.py --tag ${T}_thr
$PY -u scripts/plot_cv_clean_figures.py --tag ${T}_thr
step "3/9 TranAD + paired stats (Deep-IDS results reused from v62)"
$PY -u revision_baselines/02_tranad.py
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_deepids_$T.json
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_tranad_$T.json --metric binary_f1
step "4/9 deployment model (Phase-1 checkpoints): calibration, error analysis, XCS, fold group importance"
$PY -u scripts/deployment_eval.py
$PY -u scripts/extract_misclass_artifacts.py
$PY -u scripts/analyze_misclass.py results/tables/misclass_artifacts_clean.npz
$PY -u scripts/plot_section_5_9_figures.py results/tables/misclass_artifacts_clean.npz results/figures/${T}_thr
$PY -u scripts/xcs_both.py --n 500
$PY -u scripts/fold_group_importance.py --tag $T
step "5/9 faithfulness: main model (+SHAP), Phase-2 variant, hybrid, utility, hierarchical-vs-flat, Deep-IDS head-to-head, Holm"
$PY -u scripts/run_faithfulness.py --n-samples 500 --with-shap
$PY -u scripts/run_faithfulness.py --n-samples 500 --attack-ckpt results/checkpoints/hit_ids_attack_merged_phase2.pt --tag _phase2
$PY -u scripts/e6_faithfulness_hybrid.py --n-samples 500
$PY -u scripts/explanation_utility.py --n-samples 500
$PY -u scripts/ablation_faithfulness.py --n-samples 500
$PY -u revision_baselines/05_faithfulness_h2h.py --n-samples 500
$PY -u scripts/e3_stats_hardening.py --n-bootstrap 1000
step "6/9 calibration (ECE) and feature-noise robustness"
$PY -u scripts/e4_calibration.py --include-baselines
$PY -u scripts/e7_robustness.py --include-baselines
step "7/9 attribution stability across fold models"
$PY -u scripts/attribution_stability.py --tag $T --n-shap 300 --n-tree 2000
step "8/9 NSM-IDS-2026 v1.2 (no Phase 2) + explanation latency"
$PY -u scripts/nsm_session_cv.py --no-phase2
$PY -u scripts/nsm_xai.py --n 500
$PY -u scripts/transfer_cic_to_nsm.py --no-phase2
$PY -u scripts/explanation_latency.py --n 100 --repeats 5
step "9/9 final tables and figures"
$PY -u scripts/plot_cv_clean_figures.py --tag ${T}_thr
$PY -u scripts/plot_paper_extra_figures.py --out results/figures/${T}_thr
$PY -u scripts/make_paper_tables.py --tag ${T}_thr
step "NP2 DONE"

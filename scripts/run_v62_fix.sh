#!/usr/bin/env bash
# Re-run of the two steps that failed on the 61-feature table + NSM v1.2 + final tables.
set -u
cd "$(dirname "$0")/.."
PY=/home/ai-lab-ubuntu/anaconda3/envs/hit_ids/bin/python
export HIT_CV_TAG=clean_rv
T=clean_rv
step() { echo; echo "=== [$(date '+%F %T')] $1"; }
step "A TranAD (binary) + paired stats"
$PY -u revision_baselines/02_tranad.py
$PY -u revision_baselines/03_stats.py --baseline revision_results/results_tranad_$T.json --metric binary_f1
step "B HiT-IDS rollout vs Deep-IDS GradientSHAP head-to-head"
$PY -u revision_baselines/05_faithfulness_h2h.py --n-samples 500
step "C NSM-IDS-2026 v1.2: session CV, explanations, zero-shot transfer"
$PY -u scripts/nsm_session_cv.py
$PY -u scripts/nsm_xai.py --n 500
$PY -u scripts/transfer_cic_to_nsm.py
step "D final tables and figures"
$PY -u scripts/make_paper_tables.py --tag ${T}_thr
step "FIX DONE"

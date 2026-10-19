# HiT-IDS — hierarchical attention explanations for flow-based intrusion detection

Code, protocol and result files for the manuscript *HiT-IDS: hierarchical attention explanations
for flow-based intrusion detection, validated for faithfulness, cost and stability*.

The repository contains everything needed to reproduce the tables and figures of the paper:
the model (`src/`), the leak-free evaluation protocol and every experiment script (`scripts/`,
`revision_baselines/`), the configuration with the 61-feature grouping (`configs/config.yaml`),
and the result files the tables are generated from (`results/tables/`, `revision_results/`,
`manuscript/`). Trained checkpoints and the datasets are not included (see `data/nsm/README.md`).

## Environment
Python 3.10, PyTorch ≥ 2.1 with CUDA, scikit-learn, xgboost, imbalanced-learn, shap, scipy,
pandas, matplotlib, pyyaml (`pip install -r requirements.txt`). All experiments were run on one
NVIDIA RTX 5090.

## Pipeline (what the paper reports)
```bash
# 1. clean feature table from the raw CIC-IDS2017 CSVs (61 features, real rows only; cached)
python -m src.data.cv_data --config configs/config.yaml
# 2. 70/15/15 deployment split and the deployment models + baselines (fold protocol, no Phase 2)
python scripts/preprocess.py --config configs/config.yaml
python scripts/train_deployment.py --config configs/config.yaml --no-phase2
# 3. leak-free 5-fold CV (scaler, SMOTE and validation fitted inside each fold) + threshold calibration for every model
python scripts/evaluate_cv.py --config configs/config.yaml --tag clean_rv --val-source real --no-phase2
python scripts/calibrate_cascade_threshold.py --tag clean_rv
# 4. ablation, Deep-IDS baseline, explanation experiments, stability, latency, NSM-IDS-2026, transfer
bash scripts/run_np2_all.sh        # full sequence; each script can also be run on its own
# 5. tables and figures
python scripts/make_scirep_tables.py && python scripts/plot_scirep_extra.py
```
Key scripts: `evaluate_cv.py` (CV protocol), `calibrate_cascade_threshold.py` (validation-calibrated
thresholds, HiT-IDS and baselines), `run_faithfulness.py` (deletion/insertion/AOPC/comprehensiveness/
sufficiency for rollout, gradient, GradientSHAP, random), `attribution_stability.py` (stability of four
attributions across fold models), `explanation_latency.py` (one timing protocol for eight methods),
`nsm_session_cv.py` / `nsm_xai.py` / `transfer_cic_to_nsm.py` (NSM-IDS-2026 experiments),
`make_scirep_tables.py` (every number in the paper from the result files).

## Datasets
- CIC-IDS2017 — Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/ids-2017.html)
- UNSW-NB15 — UNSW Canberra (https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- NSM-IDS-2026 v1.2 — Zenodo, CC BY 4.0, https://doi.org/10.5281/zenodo.23392183

## A note on the protocol
`data/splits/train.pt` produced by `preprocess.py --apply-smote` is a resampled table and must never
be concatenated with validation/test rows before cross-validation; `src/data/cv_data.py` rebuilds the
real, unscaled table from the raw CSVs for that reason, and every CV script fits the scaler, the
resampler and the early-stopping split inside the training fold only.

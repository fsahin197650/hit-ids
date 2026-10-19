# README — Revision Baseline Experiments

## Overview

This directory contains scripts to run two new baseline experiments for the
HiT-IDS paper revision:

1. **Deep-IDS** (Racha et al., 2024) — LSTM-based IDS baseline
2. **TranAD** (Tuli et al., VLDB 2022) — Unsupervised transformer anomaly detector

All scripts reuse the **identical** 5-fold CV pipeline from the paper's Table 4
(same folds, SMOTE, evaluation) via `pipeline_hooks.py`.

---

## Files

| File | Purpose |
|------|---------|
| `pipeline_hooks.py` | Single wiring point — all scripts import from here |
| `00_repro_check.py` | Sanity gate: reproduce BiLSTM F1 ≈ 0.9966 |
| `01_deepids.py` | Deep-IDS (LSTM-64) 13-class baseline |
| `02_tranad.py` | TranAD binary anomaly detection comparison |
| `03_stats.py` | Paired t-test, Cohen's d, 95% CI vs HiT-IDS |
| `04_latency.py` | Explanation-cost measurement for Table 8 |
| `05_faithfulness_h2h.py` | H2H faithfulness: rollout vs GradientSHAP (Track 1 + Track 2) |
| `prior_results.md` | Literature comparison table (no GPU) |
| `COMMANDS.md` | Runbook with exact commands |

---

## Assumptions About the Existing Repo

These assumptions are documented in `pipeline_hooks.py` header:

| ID | Assumption | Where verified |
|----|-----------|---------------|
| A1 | Data at `data/splits/{train,val,test}.pt` with keys `X`, `y` | `ls data/splits/` |
| A2 | 15-class → 13-class via MERGE_MAP (Web Attack subtypes → single) | `evaluate_cv.py` L50–54 |
| A3 | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | `evaluate_cv.py` L557 |
| A4 | SMOTE `target_count=20_000`, `random_state=42+fold_i` | `evaluate_cv.py` L80, L574–578 |
| A5 | Majority undersample at 200,000 | `evaluate_cv.py` L128 |
| A6 | BiLSTM: `hidden_size=128, num_layers=2, dropout=0.3` | `baselines.py` L158–165 |
| A7 | Training: AdamW, lr=1e-3, CosineAnnealingLR, 30 epochs | `baselines.py` L349–351 |
| A8 | Config: `configs/config.yaml` | All scripts |

---

## What the User Must Fix (if anything)

### In `pipeline_hooks.py`:

1. **`REPO_ROOT`** — auto-detected from the file's location. If you move the
   `revision_baselines/` directory, update this.

2. **`SPLITS_DIR`** — defaults to `data/splits`. If your preprocessed data is
   elsewhere, change this.

3. **`HIT_IDS_PER_FOLD_F1`** — hardcoded from `cv_evaluation.json`. If you re-ran
   the HiT-IDS CV, update these 5 values.

### Dependencies:
```bash
pip install captum  # for 04_latency.py (GradientSHAP)
```

All other dependencies (`torch`, `sklearn`, `imblearn`, `scipy`, `numpy`) should
already be installed if the existing pipeline runs.

---

## Run Order

See `COMMANDS.md` for the full runbook. Quick summary:

```bash
# 0. MUST pass first
python revision_baselines/00_repro_check.py

# 1. Deep-IDS (primary)
python revision_baselines/01_deepids.py

# 2. TranAD (secondary)
python revision_baselines/02_tranad.py      # or: --fallback

# 3. Statistics
python revision_baselines/03_stats.py --baseline revision_results/results_deepids.json

# 4. Latency
python revision_baselines/04_latency.py --models deepids,tranad

# 5. Faithfulness H2H (THE KEY EXPERIMENT)
python revision_baselines/05_faithfulness_h2h.py
```

---

## Design Decisions

1. **Deep-IDS as 13-class supervised**: The paper compares 13-class models in Table 4.
   Deep-IDS is trained under the same protocol for a fair comparison. We use LSTM(64)
   as described in the paper, with the same training budget as BiLSTM (30 epochs,
   AdamW, batch 512).

2. **TranAD as binary only**: TranAD is an unsupervised anomaly detector — forcing it
   into 13-class classification would be unfair and methodologically unsound. We compare
   its binary detection with HiT-IDS's Stage-1 binary detector.

3. **Windowing for TranAD**: CIC-IDS2017 flows are not naturally a time series.
   We use a simple sequential windowing (window_size=10). A window is ATTACK if
   any flow in it is an attack. This is documented and the user can adjust.

4. **Fallback mode**: If TranAD's adaptation to flow data is unreliable, the
   `--fallback` flag emits a paradigm-level differentiation note with published
   numbers instead of forcing questionable results.

5. **Honesty**: If Deep-IDS beats HiT-IDS on macro-F1, the scripts report it as-is.
   The paper's thesis is explainability-at-latency, not topping accuracy.

6. **Faithfulness H2H (05_faithfulness_h2h.py)**: This is the critical experiment.
   It tests whether HiT-IDS's intrinsic explanation is more *faithful* than Deep-IDS's
   post-hoc GradientSHAP, head-to-head. Key methodological point: **each explanation
   is evaluated on its own model** — you do NOT mask features on one model using another
   model's attributions. This is the methodologically correct way to test faithfulness.

   **Modules reused:**
   - `src.xai.faithfulness.ComprehensiveFaithfulness` — the exact 11-metric battery
   - `src.xai.attention_rollout.AttentionRollout` — rollout explainer
   - `src.models.hit_ids.HiTIDS` — HiT-IDS model
   - `revision_baselines.01_deepids.DeepIDSModel` — Deep-IDS model
   - `captum.attr.GradientShap` — SHAP explainer

   **Track 1 (feature-level):** Identical 11 metrics from Section 5.12, both at
   72-feature granularity. Apples-to-apples comparison.

   **Track 2 (group-level):** Aggregate to 4 semantic groups (temporal/volume/
   protocol/behavioral). HiT-IDS uses native inter-group attention; Deep-IDS uses
   sum(|SHAP_attr|) per group. Tests the *structured explanation* claim directly.

   **User must verify in CONFIG block:**
   - `masking_value` = 0.0 (matches Section 5.12)
   - `n_samples` = 500 (matches Section 5.12)
   - `k_values` = [5,10,15,20] (matches Section 5.12)
   - `deepids_num_classes` = 13 (matches 01_deepids.py)

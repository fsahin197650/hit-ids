## Table 1. Datasets after cleaning

| Dataset | Flows | Classes | Features | Evaluation | Notes |
|---|---|---|---|---|---|
| CIC-IDS2017 | 2,520,798 | 13 | 61 | 5-fold stratified; scaler, SMOTE, validation inside the fold | BENIGN 2,095,057; Heartbleed 11, Infiltration 36 |
| NSM-IDS-2026 v1.2 | 97,226 | 10 | 52 | 4 session-grouped folds (3 sessions each); one training session held out for validation | split fragments excluded, exact duplicates removed |
| UNSW-NB15 | 257,673 | 2 (normal/attack) | 68 | 5-fold stratified, fold-internal SMOTE | official train+test concatenated |

## Table 2. CIC-IDS2017, 5-fold cross-validation, every model with a validation-calibrated detection threshold

| Model | Macro-F1, 13 classes | Macro-F1, 11 classes (≥389 test flows/fold) | Macro-F1 at argmax | Detection rate | False-positive rate | Macro ROC-AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.953 ± 0.015 | 0.979 ± 0.002 | 0.940 | 0.9936 | 0.08% | 0.9992 |
| Random Forest | 0.953 ± 0.014 | 0.971 ± 0.005 | 0.951 | 0.9988 | 0.16% | 0.9975 |
| BiLSTM | 0.939 ± 0.012 | 0.959 ± 0.012 | 0.903 | 0.9943 | 0.12% | 0.9993 |
| Flat Transformer | 0.933 ± 0.012 | 0.956 ± 0.005 | 0.910 | 0.9952 | 0.11% | 0.9986 |
| HiT-IDS | 0.895 ± 0.033 | 0.948 ± 0.004 | 0.885 | 0.9918 | 0.12% | 0.9955 |
| Deep-IDS (LSTM) | 0.840 ± 0.025 | — | — | — | — | — |

The two rarest classes (Heartbleed 2–3 and Infiltration 7–8 test flows per fold) dominate the fold-to-fold variance of the 13-class macro-F1; the 11-class column excludes them.

## Table 3. Paired comparison with HiT-IDS over the five folds (13-class macro-F1; two-sided paired t-test, t-based 95% CI)

| Baseline | ΔF1 (HiT-IDS − baseline) | 95% CI | Cohen's d | p |
|---|---|---|---|---|
| XGBoost | -0.058 | [-0.108, -0.009] | -2.01 | 0.030 |
| Random Forest | -0.058 | [-0.122, +0.006] | -2.02 | 0.064 |
| BiLSTM | -0.044 | [-0.080, -0.008] | -1.58 | 0.026 |
| Flat Transformer | -0.038 | [-0.089, +0.014] | -1.34 | 0.114 |
| Deep-IDS (LSTM) | +0.045 | — | 1.58 | 0.083 |

## Table 4. Per-class F1 on CIC-IDS2017 (mean over folds; HiT-IDS also with precision and recall)

| Class | Test flows (5 folds) | HiT-IDS P | HiT-IDS R | HiT-IDS F1 | XGBoost | RF | BiLSTM | Flat Tr. |
|---|---|---|---|---|---|---|---|---|
| BENIGN | 2,095,057 | 0.998 | 0.999 | 0.999 ± 0.000 | 0.999 | 0.999 | 0.999 | 0.999 |
| Bot | 1,948 | 0.626 | 0.588 | 0.570 ± 0.034 | 0.820 | 0.755 | 0.687 | 0.615 |
| DDoS | 128,014 | 0.999 | 0.999 | 0.999 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DoS GoldenEye | 10,286 | 0.990 | 0.975 | 0.982 ± 0.003 | 0.996 | 0.994 | 0.991 | 0.990 |
| DoS Hulk | 172,846 | 0.999 | 0.990 | 0.994 ± 0.001 | 0.998 | 0.997 | 0.996 | 0.996 |
| DoS Slowhttptest | 5,228 | 0.967 | 0.988 | 0.978 ± 0.004 | 0.989 | 0.989 | 0.984 | 0.982 |
| DoS slowloris | 5,385 | 0.989 | 0.990 | 0.989 ± 0.001 | 0.995 | 0.992 | 0.991 | 0.993 |
| FTP-Patator | 5,931 | 0.998 | 0.997 | 0.998 ± 0.000 | 0.999 | 0.998 | 0.998 | 0.999 |
| Heartbleed | 11 | 0.800 | 0.900 | 0.813 ± 0.107 | 0.867 | 0.860 | 0.905 | 0.867 |
| Infiltration | 36 | 0.389 | 0.400 | 0.391 ± 0.352 | 0.761 | 0.851 | 0.757 | 0.739 |
| PortScan | 90,694 | 0.988 | 0.992 | 0.990 ± 0.007 | 0.984 | 0.993 | 0.991 | 0.994 |
| SSH-Patator | 3,219 | 0.946 | 0.987 | 0.966 ± 0.004 | 0.999 | 0.997 | 0.984 | 0.980 |
| Web Attack | 2,143 | 0.955 | 0.975 | 0.965 ± 0.006 | 0.989 | 0.967 | 0.930 | 0.970 |

## Table 5. NSM-IDS-2026 v1.2, session-grouped 4-fold cross-validation (calibrated thresholds)

| Model | Macro-F1, 10 classes | at argmax | Detection rate | False-positive rate | ΔF1 vs HiT-IDS |
|---|---|---|---|---|---|
| XGBoost | 0.990 ± 0.001 | 0.990 | 0.9993 | 0.05% | -0.007 (p = 0.09) |
| Random Forest | 0.989 ± 0.003 | 0.990 | 0.9974 | 0.03% | -0.006 (p = 0.01) |
| BiLSTM | 0.983 ± 0.004 | 0.983 | 0.9986 | 0.08% | +0.000 (p = 0.98) |
| Flat Transformer | 0.984 ± 0.003 | 0.983 | 0.9961 | 0.01% | -0.001 (p = 0.50) |
| HiT-IDS | 0.983 ± 0.005 | 0.983 | 0.9990 | 0.04% | — |

Per-class HiT-IDS F1 (mean over folds): BENIGN 0.999, DoS GoldenEye 0.995, DoS SYN 0.962, DoS Slowhttptest 0.969, DoS Slowloris 0.978, FTP-Patator 0.995, Infiltration 0.967, PortScan 0.990, SSH-Patator 0.977, Web Attack 0.998.

## Table 6. UNSW-NB15, binary, 5-fold cross-validation

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Random Forest | 0.947 ± 0.001 | 0.951 |
| XGBoost | 0.944 ± 0.001 | 0.948 |
| BiLSTM | 0.941 ± 0.001 | 0.945 |
| HiT-IDS | 0.935 ± 0.002 | 0.939 |
| Flat Transformer | 0.934 ± 0.002 | 0.938 |

## Table 7. Faithfulness of four attributions on the same model: CIC-IDS2017 deployment model (498 flows) and NSM-IDS-2026 fold-1 model (500 flows)

| Metric | CIC rollout | CIC gradient | CIC GradientSHAP | CIC random | NSM rollout | NSM gradient | NSM GradientSHAP | NSM random |
|---|---|---|---|---|---|---|---|---|
| Deletion AUC ↓ | 0.339 | 0.552 | 0.380 | 0.564 | 0.335 | 0.523 | 0.366 | 0.633 |
| Insertion AUC ↑ | 0.712 | 0.656 | 0.810 | 0.571 | 0.754 | 0.471 | 0.742 | 0.451 |
| AOPC ↑ | 0.608 | 0.395 | 0.566 | 0.383 | 0.614 | 0.423 | 0.582 | 0.311 |
| Comprehensiveness@5 ↑ | 0.117 | 0.043 | 0.421 | 0.064 | 0.427 | 0.188 | 0.394 | 0.053 |
| Comprehensiveness@10 ↑ | 0.349 | 0.090 | 0.530 | 0.144 | 0.530 | 0.290 | 0.476 | 0.125 |
| Comprehensiveness@15 ↑ | 0.525 | 0.146 | 0.596 | 0.214 | 0.572 | 0.346 | 0.535 | 0.222 |
| Comprehensiveness@20 ↑ | 0.766 | 0.274 | 0.590 | 0.294 | 0.600 | 0.425 | 0.582 | 0.313 |
| Sufficiency@5 ↓ | 0.404 | 0.556 | 0.313 | 0.589 | 0.461 | 0.677 | 0.418 | 0.676 |
| Sufficiency@10 ↓ | 0.276 | 0.524 | 0.234 | 0.578 | 0.234 | 0.606 | 0.269 | 0.601 |
| Sufficiency@15 ↓ | 0.159 | 0.438 | 0.178 | 0.537 | 0.157 | 0.553 | 0.186 | 0.543 |
| Sufficiency@20 ↓ | 0.237 | 0.318 | 0.142 | 0.505 | 0.111 | 0.463 | 0.124 | 0.483 |

Rollout better than random on 11/11 (CIC) and 11/11 (NSM) metrics; better than input gradient on 11/11 and 11/11; better than GradientSHAP on 4/11 and 10/11. Holm-corrected attention-vs-random tests: 11/11 significant on CIC-IDS2017, 11/11 on UNSW-NB15.

## Table 8. Stability of attributions across independently trained fold models (CIC-IDS2017: 5 folds; NSM-IDS-2026: 4 session folds)

| Attribution | CIC: inter-fold ρ (global) | CIC: per-class ρ | CIC: dominant-group agreement | NSM: inter-fold ρ (global) | NSM: per-class ρ | NSM: dominant-group agreement |
|---|---|---|---|---|---|---|
| HiT-IDS rollout | 0.67 | 0.56 | 66% | 0.78 | 0.69 | 78% |
| HiT-IDS GradientSHAP | 0.72 | 0.71 | 81% | 0.74 | 0.68 | 81% |
| XGBoost TreeSHAP | 0.94 | 0.81 | 93% | 0.91 | 0.79 | 100% |
| Random Forest TreeSHAP | 0.98 | 0.94 | 100% | 0.97 | 0.94 | 83% |

## Table 9. Agreement of rollout with GradientSHAP (500 flows, CIC deployment model)

| Attribution | Spearman ρ vs SHAP | top-5 overlap | top-10 overlap |
|---|---|---|---|
| attention rollout | 0.43 ± 0.11 | 0.36 | 0.44 |
| attention × |gradient| | 0.37 ± 0.08 | 0.24 | 0.35 |
| |gradient| alone | 0.27 ± 0.09 | 0.19 | 0.29 |

## Table 10. Explanation latency per flow, one protocol (100 flows, median of 5 repeats, RTX 5090)

| Method | Type | One flow (ms) | Batched (ms/flow) |
|---|---|---|---|
| HiT-IDS prediction only (no explanation) | reference | 1.82 | 0.019 |
| HiT-IDS attention rollout | intrinsic | 3.41 | 0.035 |
| HiT-IDS input x gradient | post-hoc (1 backward) | 7.52 | 0.081 |
| HiT-IDS GradientSHAP | post-hoc | 517.80 | 515.322 |
| Random Forest TreeSHAP | post-hoc | 295.12 | 294.683 |
| XGBoost TreeSHAP | post-hoc | 8.48 | 0.631 |
| BiLSTM GradientSHAP | post-hoc | 1066.27 | 1063.456 |
| Vanilla Transformer GradientSHAP | post-hoc | 348.01 | 346.214 |

## Table 11. Zero-shot transfer CIC-IDS2017 → NSM-IDS-2026 on 48 portable features (attack vs benign on all 97,226 NSM flows)

| Normalisation | Model (trained on CIC-IDS2017) | Binary F1 | Detection rate | FPR | ROC-AUC |
|---|---|---|---|---|---|
| source-scaled | HiT-IDS | 0.642 | 0.489 | 4.42% | 0.744 |
| source-scaled | Random Forest | 0.069 | 0.036 | 0.00% | 0.637 |
| source-scaled | XGBoost | 0.158 | 0.086 | 0.44% | 0.752 |
| target-scaled | HiT-IDS | 0.379 | 0.241 | 3.65% | 0.796 |
| target-scaled | Random Forest | 0.000 | 0.000 | 0.00% | 0.635 |
| target-scaled | XGBoost | 0.001 | 0.000 | 0.00% | 0.728 |

HiT-IDS per-class detection rate on NSM (source-scaled): BENIGN 0.04, DoS GoldenEye 0.78, DoS SYN 1.00, DoS Slowhttptest 0.44, DoS Slowloris 0.59, FTP-Patator 0.01, Infiltration 0.60, PortScan 0.01, SSH-Patator 0.64, Web Attack 0.28. In-domain CIC validation binary F1 of the portable model: 0.991.

## Table 12. Ablation on CIC-IDS2017 (5-fold, argmax rule; reference = design including the entropy fine-tuning stage)

| ID | Variant | Macro-F1 | ΔF1 |
|---|---|---|---|
| — | Reference (with entropy fine-tuning) | 0.859 | — |
| A1 | No Grouping (Flat) | 0.843 ± 0.033 | -0.016 |
| A2 | No Cascade (Direct 13-class) | 0.698 ± 0.005 | -0.161 |
| A3 | No Phase 2 | 0.885 ± 0.025 | +0.026 |
| A4 | ReLU (not GELU) | 0.871 ± 0.014 | +0.012 |
| A5 | No SMOTE | 0.906 ± 0.019 | +0.047 |
| A6 | Single Head (h=1) | 0.840 ± 0.013 | -0.019 |

## Table 13. Deployment model (70/15/15 split) at the argmax rule and at the validation-calibrated threshold

| Rule | Macro-F1 (13) | Macro-F1 (11) | Detection rate | FPR | Bot F1 | Infiltration recall |
|---|---|---|---|---|---|---|
| argmax | 0.919 | 0.953 | 0.9963 | 0.163% | 0.615 | 0.67 (6 flows) |
| calibrated (t* = 0.575) | 0.883 | 0.957 | 0.9949 | 0.118% | 0.626 | 0.17 (6 flows) |

## Table 14. Largest error groups of the deployment model (calibrated) and their group-attribution divergence

| Error group | n | mean conf. | Δ temporal | Δ volume | Δ protocol | Δ behavioral |
|---|---|---|---|---|---|---|
| BENIGN → PortScan (false positive) | 158 | 0.92 | -0.00 | -0.05 | -0.02 | +0.07 |
| DoS Hulk → BENIGN (false negative at Stage 1) | 139 | 0.54 | +0.06 | -0.13 | -0.09 | +0.15 |
| BENIGN → Bot (false positive) | 116 | 0.92 | +0.03 | -0.18 | +0.13 | +0.03 |
| Bot → BENIGN (false negative at Stage 1) | 106 | 0.55 | -0.05 | +0.07 | -0.14 | +0.12 |
| BENIGN → DoS Hulk (false positive) | 39 | 0.91 | +0.02 | -0.20 | +0.25 | -0.06 |
| DoS GoldenEye → BENIGN (false negative at Stage 1) | 32 | 0.58 | +0.13 | -0.25 | +0.01 | +0.11 |

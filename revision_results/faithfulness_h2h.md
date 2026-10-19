# Faithfulness Head-to-Head: HiT-IDS Rollout vs Deep-IDS GradientSHAP

**Eval samples:** 485 attack samples (correctly classified by each model)
**Masking:** zero-masking (value = 0.0)
**GPU:** NVIDIA GeForce RTX 5090
**Runtime:** 94s

---

## Track 1 — Feature-Level Faithfulness (11 Metrics)

Each explanation evaluated on **its own model** (rollout → HiT-IDS, SHAP → Deep-IDS).

| Metric | Better | HiT-IDS Rollout | Deep-IDS SHAP | Δ | p (Wilcoxon) | Winner |
|--------|:------:|:---------------:|:-------------:|:---:|:------------:|:------:|
| deletion_auc | ↓ | 0.3419±0.2165 | 0.1395±0.1610 | +0.2023 | 8.73e-53 | 🔴 SHAP |
| insertion_auc | ↑ | 0.7116±0.1731 | 0.5539±0.2304 | +0.1576 | 1.50e-48 | 🟢 Rollout |
| aopc | ↑ | 0.6053±0.2276 | 0.8503±0.2005 | -0.2450 | 4.13e-58 | 🔴 SHAP |
| comprehensiveness_k5 | ↑ | 0.1107±0.2257 | 0.7479±0.3961 | -0.6371 | 4.50e-71 | 🔴 SHAP |
| comprehensiveness_k10 | ↑ | 0.3394±0.3490 | 0.8193±0.3503 | -0.4799 | 4.12e-55 | 🔴 SHAP |
| comprehensiveness_k15 | ↑ | 0.5204±0.3809 | 0.8621±0.3027 | -0.3417 | 1.77e-43 | 🔴 SHAP |
| comprehensiveness_k20 | ↑ | 0.7665±0.2595 | 0.8812±0.2685 | -0.1147 | 2.11e-33 | 🔴 SHAP |
| sufficiency_k5 | ↓ | 0.3986±0.3399 | 0.8102±0.3422 | -0.4116 | 3.13e-60 | 🟢 Rollout |
| sufficiency_k10 | ↓ | 0.2722±0.3268 | 0.7720±0.3712 | -0.4998 | 2.60e-70 | 🟢 Rollout |
| sufficiency_k15 | ↓ | 0.1518±0.2196 | 0.6767±0.4193 | -0.5248 | 1.02e-69 | 🟢 Rollout |
| sufficiency_k20 | ↓ | 0.2410±0.3615 | 0.5179±0.4549 | -0.2769 | 1.40e-38 | 🟢 Rollout |

**Feature-level tally: Rollout wins 5/11, SHAP wins 6/11, Ties 0/11 (α=0.05)**

---

## Track 2 — Group-Level Faithfulness (Structured Explanation)

HiT-IDS: native inter-group attention. Deep-IDS: GradientSHAP aggregated to 4 groups.

| Metric | Better | HiT-IDS Rollout | Deep-IDS SHAP | Δ | p (Wilcoxon) | Winner |
|--------|:------:|:---------------:|:-------------:|:---:|:------------:|:------:|
| group_comp_top1 | ↑ | 0.2285±0.2527 | 0.7980±0.3488 | -0.5696 | 1.06e-67 | 🔴 SHAP |
| group_suff_top1 | ↓ | 0.1673±0.2882 | 0.7289±0.3388 | -0.5615 | 1.73e-69 | 🟢 Rollout |
| group_comp_top2 | ↑ | 0.5697±0.3082 | 0.9169±0.2271 | -0.3472 | 1.82e-60 | 🔴 SHAP |
| group_suff_top2 | ↓ | 0.0152±0.0848 | 0.3830±0.4339 | -0.3679 | 5.04e-39 | 🟢 Rollout |
| group_comp_top3 | ↑ | 0.6085±0.3276 | 0.7899±0.3117 | -0.1814 | 1.33e-27 | 🔴 SHAP |
| group_suff_top3 | ↓ | 0.0107±0.0829 | 0.1160±0.2612 | -0.1052 | 1.38e-12 | 🟢 Rollout |
| group_comp_top4 | ↑ | 0.5441±0.4246 | 0.6087±0.3939 | -0.0645 | 1.05e-60 | 🔴 SHAP |
| group_suff_top4 | ↓ | 0.0000±0.0000 | 0.0000±0.0000 | +0.0000 | nan | ⚪ Tie |
| group_aopc | ↑ | 0.4877±0.2292 | 0.7784±0.2113 | -0.2907 | 1.54e-66 | 🔴 SHAP |

**Group-level tally: Rollout wins 3/9, SHAP wins 5/9, Ties 1/9 (α=0.05)**

---

## Interpretation

**Case: WEAK** — Rollout does not win either level convincingly.
→ Intrinsic-explainability advantage is not empirically supported vs a real competitor.

**Summary: Rollout wins 5/11 feature-level, 3/9 group-level (α=0.05)**

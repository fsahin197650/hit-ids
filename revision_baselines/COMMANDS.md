# COMMANDS.md — Revision Baseline Runbook

All commands run from the project root:  
`cd /home/ai-lab-ubuntu/Desktop/transformer_based_IDS_XAI`

---

## Prerequisites

```bash
# Ensure dependencies are installed
pip install captum   # needed for 04_latency.py (GradientSHAP)
```

---

## 0) Sanity Gate — MUST PASS before trusting anything

```bash
python revision_baselines/00_repro_check.py
```

| Item | Details |
|------|---------|
| **Writes** | Nothing (prints only) |
| **Runtime** | ~20–40 min (5 folds × BiLSTM training) |
| **GPU** | Required |
| **Pass criterion** | BiLSTM macro-F1 ≈ 0.9966 ± 0.005. If far off → fix `pipeline_hooks.py` and re-run |

> ⚠️ **Do NOT proceed to steps 1–4 if this fails.** The pipeline wiring is broken.

---

## 1) Deep-IDS Baseline (PRIMARY)

```bash
python revision_baselines/01_deepids.py
```

| Item | Details |
|------|---------|
| **Writes** | `revision_results/results_deepids.json` |
| **Runtime** | ~15–30 min (5 folds × LSTM(64) training) |
| **GPU** | Required |
| **Good result** | Macro-F1 roughly in range 0.95–0.997. Should be close to reported 0.9791 but may differ due to our more aggressive SMOTE/harness. |

Optional tuning:
```bash
python revision_baselines/01_deepids.py --epochs 50 --lr 5e-4 --lstm-units 64
```

---

## 2) TranAD Comparison (SECONDARY)

### Normal mode:
```bash
python revision_baselines/02_tranad.py
```

### Fallback mode (if adaptation is unreliable):
```bash
python revision_baselines/02_tranad.py --fallback
```

| Item | Details |
|------|---------|
| **Writes** | `revision_results/results_tranad.json` (normal) or `revision_results/tranad_note.md` (fallback) |
| **Runtime** | ~15–25 min (normal); instant (fallback) |
| **GPU** | Required (normal mode) |
| **Good result** | Binary F1 > 0.5. If binary F1 < 0.3 or results are unstable → use `--fallback` |
| **Note** | TranAD is unsupervised — comparing binary detection only, not 13-class |

Optional tuning:
```bash
python revision_baselines/02_tranad.py --window 10 --epochs 50
```

---

## 3) Paired Statistics vs HiT-IDS

### For Deep-IDS:
```bash
python revision_baselines/03_stats.py --baseline revision_results/results_deepids.json
```

### For TranAD (binary metric):
```bash
python revision_baselines/03_stats.py --baseline revision_results/results_tranad.json --metric binary_f1
```

| Item | Details |
|------|---------|
| **Writes** | `revision_results/stats_deep-ids_lstm.json` / `revision_results/stats_tranad.json` |
| **Runtime** | Instant (no GPU, reads JSON only) |
| **Good result** | Prints p-value, Cohen's d, 95% CI. Numbers should be internally consistent. |

---

## 4) Explanation-Cost Rows (Table 8 Extension)

```bash
python revision_baselines/04_latency.py --models deepids,tranad
```

| Item | Details |
|------|---------|
| **Writes** | `revision_results/table8_new_rows.md` |
| **Runtime** | ~2–5 min |
| **GPU** | Required — run on the **same GPU** as the original Table 8 measurements |
| **Good result** | GradientSHAP >> 3.43 ms (HiT-IDS rollout); TranAD deviation ~ 0.5–5 ms |
| **Dependency** | `pip install captum` for GradientSHAP |

To measure only Deep-IDS:
```bash
python revision_baselines/04_latency.py --models deepids
```

---

## 5) Faithfulness Head-to-Head: HiT-IDS Rollout vs Deep-IDS GradientSHAP

```bash
python revision_baselines/05_faithfulness_h2h.py
```

| Item | Details |
|------|---------|
| **Writes** | `revision_results/faithfulness_h2h.json` + `revision_results/faithfulness_h2h.md` |
| **Runtime** | ~30–60 min (GradientSHAP on 500 samples + perturbation battery × 2 models) |
| **GPU** | Required |
| **Dependencies** | `pip install captum` |
| **Sanity gate** | Rollout should still beat RANDOM on ~11/11 (consistency with Section 5.12). If it doesn't, the battery isn't wired correctly — fix before trusting the SHAP comparison. |
| **Quick test** | `python revision_baselines/05_faithfulness_h2h.py --n-samples 50` |

### How to read the result:

| Case | Feature-level | Group-level | Meaning |
|------|:---:|:---:|---|
| **STRONG** | ≈7+/11 rollout wins | majority wins | Thesis holds: intrinsic > post-hoc |
| **PARTIAL** | mixed | majority wins | Narrow to group-level claim only |
| **WEAK** | losses | losses | Thesis not supported; rethink framing |

---

## Summary of outputs

After running all commands, `revision_results/` should contain:

```
revision_results/
├── results_deepids.json       # Deep-IDS 5-fold results
├── results_tranad.json        # TranAD 5-fold binary results
├── stats_deep-ids_lstm.json   # Paired stats: HiT-IDS vs Deep-IDS
├── stats_tranad.json          # Paired stats: HiT-IDS vs TranAD
├── table8_new_rows.md         # Table 8 extension
├── faithfulness_h2h.json      # H2H faithfulness: per-metric paired stats
├── faithfulness_h2h.md        # H2H faithfulness: ready-to-paste tables
└── tranad_note.md             # (only if --fallback used)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: src.models.baselines` | Check `REPO_ROOT` in `pipeline_hooks.py` |
| `FileNotFoundError: data/splits/train.pt` | Run preprocessing first or fix `SPLITS_DIR` |
| BiLSTM repro F1 far from 0.9966 | Check MERGE_MAP, SMOTE params in `pipeline_hooks.py` |
| TranAD binary F1 < 0.3 | Use `--fallback` mode; the paradigm mismatch may be too large |
| `ModuleNotFoundError: captum` | `pip install captum` |
| CUDA out of memory | Reduce `--batch-size 128` |

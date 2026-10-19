#!/usr/bin/env python3
"""
RUNBOOK 3 — Section 5.9 Figures

Generate Fig. 9 (confusion matrix heatmap) and Fig. 10 (attention divergence
bar chart) from misclass_artifacts.npz.

Usage:
    conda activate hit_ids
    python scripts/plot_section_5_9_figures.py

INPUT:  misclass_artifacts.npz (must exist in project root)
OUTPUT: results/figures/fig9_confusion_matrix.png
        results/figures/fig10_attention_divergence.png

TIME ESTIMATE: ~30 seconds
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys

ART = sys.argv[1] if len(sys.argv) > 1 else "results/tables/misclass_artifacts_clean.npz"
FIG_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/figures/clean_rv_thr"
# ── Load data ────────────────────────────────────────────────────────────
data = np.load(ART, allow_pickle=True)
y_true = data["y_true"]
y_pred = data["y_pred"]
cm = data["confusion_matrix"]
attn_group = data["attn_group"]
class_names = list(data["class_names"])
group_names = list(data["group_names"])

os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIG 9 — Confusion Matrix Heatmap (log-scale)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 10))

# Log-scale for visualization (add 1 to avoid log(0))
cm_log = np.log10(cm + 1)

short_names = [
    "BEN", "Bot", "DDoS", "GEye", "Hulk",
    "SHT", "SLow", "FTP-P", "HBleed",
    "Infil", "PScan", "SSH-P", "Web"
]

sns.heatmap(
    cm_log,
    xticklabels=short_names,
    yticklabels=short_names,
    cmap="YlOrRd",
    annot=cm,
    fmt="d",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Count (log₁₀ scale)"},
    ax=ax,
)
ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
ax.set_title(f"HiT-IDS cascade confusion matrix (deployment model, {len(y_true):,} real test rows)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_confusion_matrix_deployment.png"), dpi=600, bbox_inches="tight")
print("✓ Saved: results/figures/fig9_confusion_matrix.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIG 10 — Attention Divergence per Error Group
# ═══════════════════════════════════════════════════════════════════════════

cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
error_groups = {}
for flat in np.argsort(cm_off.ravel())[::-1][:5]:
    i, j = divmod(int(flat), cm.shape[0])
    if cm_off[i, j] == 0:
        break
    error_groups[f"{short_names[i]}→{short_names[j]}\n(n={int(cm_off[i, j])})"] = {"true": i, "pred": j}

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(error_groups))
width = 0.2
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

for g_idx, gn in enumerate(group_names):
    deltas = []
    for group_label, info in error_groups.items():
        mis_mask = (y_true == info["true"]) & (y_pred == info["pred"])
        cor_mask = (y_true == info["true"]) & (y_pred == info["true"])
        mis_mean = attn_group[mis_mask, g_idx].mean()
        cor_mean = attn_group[cor_mask, g_idx].mean() if cor_mask.sum() > 0 else 0
        deltas.append(mis_mean - cor_mean)
    
    offset = (g_idx - 1.5) * width
    bars = ax.bar(x + offset, deltas, width, label=gn.capitalize(), color=colors[g_idx], edgecolor="white", linewidth=0.5)

ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
ax.set_xlabel("Error Group", fontsize=12, fontweight="bold")
ax.set_ylabel("Attention Divergence (Δ = misclassified − correct)", fontsize=11, fontweight="bold")
ax.set_title("Group-level attention divergence per error group (deployment model)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(list(error_groups.keys()), fontsize=9)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_attention_divergence.png"), dpi=600, bbox_inches="tight")
print("✓ Saved: results/figures/fig10_attention_divergence.png")
plt.close()

print("\n✓ All figures generated!")

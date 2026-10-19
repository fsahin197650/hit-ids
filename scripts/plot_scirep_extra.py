#!/usr/bin/env python3
"""Extra figures for the Scientific Reports version (np2 results).

  faithfulness_4methods.{png,pdf}  rollout / gradient / GradientSHAP / random, CIC + NSM
  stability_cic_nsm.{png,pdf}      4 attribution methods, 3 stability measures, both datasets
  transfer_per_class.{png,pdf}     zero-shot CIC→NSM per-class detection rates
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results/figures/scirep"
os.makedirs(OUT, exist_ok=True)
MET = [("deletion_auc", "Del. AUC ↓"), ("insertion_auc", "Ins. AUC ↑"), ("aopc", "AOPC ↑"),
       ("comprehensiveness_k5", "Comp@5 ↑"), ("comprehensiveness_k10", "Comp@10 ↑"), ("comprehensiveness_k15", "Comp@15 ↑"),
       ("comprehensiveness_k20", "Comp@20 ↑"), ("sufficiency_k5", "Suff@5 ↓"), ("sufficiency_k10", "Suff@10 ↓"),
       ("sufficiency_k15", "Suff@15 ↓"), ("sufficiency_k20", "Suff@20 ↓")]
METHODS = [("attention", "attention rollout", "C0"), ("gradient", "input gradient", "C1"), ("shap", "GradientSHAP", "C2"), ("random", "random", "0.6")]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def faithfulness():
    files = [("CIC-IDS2017 (attack-type model, n = 498)", "results/tables/faithfulness_per_sample.npz"),
             ("NSM-IDS-2026 (fold-1 attack-type model, n = 500)", "results/tables/nsm_faithfulness_per_sample.npz")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(MET)); w = 0.2
    for ax, (title, f) in zip(axes, files):
        d = np.load(f)
        for i, (k, lab, col) in enumerate(METHODS):
            m = np.array([d[f"{k}_{mk}"].mean() for mk, _ in MET])
            ci = np.array([1.96 * d[f"{k}_{mk}"].std(ddof=1) / np.sqrt(len(d[f"{k}_{mk}"])) for mk, _ in MET])
            ax.bar(x + (i - 1.5) * w, m, w, yerr=ci, capsize=2, label=lab, color=col)
        ax.set_title(title, fontsize=10); ax.set_ylabel("metric value"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels([l for _, l in MET], rotation=30, ha="right", fontsize=8)
    save(fig, "faithfulness_4methods")


def stability():
    a = json.load(open("results/tables/attribution_stability_clean_rv.json"))["methods"]
    b = json.load(open("results/tables/nsm_attribution_stability.json"))["methods"]
    methods = list(a.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (title, D) in zip(axes, (("CIC-IDS2017 (5 fold models)", a), ("NSM-IDS-2026 (4 session-fold models)", b))):
        x = np.arange(len(methods))
        ax.bar(x - 0.27, [D[m]["interfold_rho_global"]["mean"] for m in methods], 0.27, label="inter-fold ρ, global importance")
        ax.bar(x, [D[m]["per_class_rho_mean"] for m in methods], 0.27, label="inter-fold ρ, per-class profile")
        ax.bar(x + 0.27, [D[m]["dominant_agreement_mean"] for m in methods], 0.27, label="dominant-group agreement")
        ax.set_xticks(x); ax.set_xticklabels([m.replace("HiT-IDS ", "HiT-IDS\n").replace(" TreeSHAP", "\nTreeSHAP") for m in methods], fontsize=8)
        ax.set_ylim(0, 1.02); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("stability across retrained models"); axes[0].legend(fontsize=7, loc="lower left")
    save(fig, "stability_cic_nsm")


def transfer():
    t = json.load(open("results/tables/transfer_cic_to_nsm.json"))
    pc = t["normalisation"]["source_scaled"]["HiT-IDS"]["per_nsm_class"]
    cls = [c for c in pc if c != "BENIGN"]
    dr = [pc[c]["detection_rate"] for c in cls]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = ["C0" if "type_recall_given_detected" in pc[c] else "0.6" for c in cls]
    ax.bar(range(len(cls)), dr, color=cols)
    for i, c in enumerate(cls):
        if "type_recall_given_detected" in pc[c]:
            ax.text(i, dr[i] + 0.02, f"type {pc[c]['type_recall_given_detected']:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("detection rate on NSM (zero-shot)"); ax.set_ylim(0, 1.1)
    ax.axhline(pc["BENIGN"]["false_positive_rate"], color="C3", ls="--", lw=0.8, label=f"benign false-positive rate {pc['BENIGN']['false_positive_rate']:.2f}")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("HiT-IDS trained on CIC-IDS2017 (48 portable features), applied to NSM-IDS-2026 without retraining", fontsize=9)
    save(fig, "transfer_per_class")


if __name__ == "__main__":
    faithfulness(); stability(); transfer()
    print(f"  figures written to {OUT}/")

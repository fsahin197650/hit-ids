#!/usr/bin/env python3
"""
Remaining manuscript figures.

  faithfulness_metrics.{png,pdf}   11 faithfulness metrics, attention / gradient / random,
                                   CIC-IDS2017 and UNSW-NB15, mean with 95% CI (per-sample npz)
  xcs_distribution.{png,pdf}       per-flow Spearman rho histograms (rollout, attention x grad,
                                   gradient alone vs SHAP) from xcs_both_clean_per_sample.npz
  latency_bars.{png,pdf}           explanation latency, log scale, one-flow and batched

Usage: python scripts/plot_paper_extra_figures.py --out results/figures/clean_rv_thr
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = [("deletion_auc", "Deletion AUC ↓"), ("insertion_auc", "Insertion AUC ↑"), ("aopc", "AOPC ↑"),
           ("comprehensiveness_k5", "Comp@5 ↑"), ("comprehensiveness_k10", "Comp@10 ↑"),
           ("comprehensiveness_k15", "Comp@15 ↑"), ("comprehensiveness_k20", "Comp@20 ↑"),
           ("sufficiency_k5", "Suff@5 ↓"), ("sufficiency_k10", "Suff@10 ↓"),
           ("sufficiency_k15", "Suff@15 ↓"), ("sufficiency_k20", "Suff@20 ↓")]
METHODS = [("attention", "attention rollout", "C0"), ("gradient", "input gradient", "C1"), ("random", "random", "0.6")]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def faithfulness(out):
    files = [("CIC-IDS2017 (attack-type model, n=496)", "results/tables/faithfulness_per_sample.npz"),
             ("UNSW-NB15 (binary model, n=500)", "results/unsw_nb15/tables/faithfulness_nb15_per_sample.npz")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(METRICS))
    w = 0.26
    for ax, (title, f) in zip(axes, files):
        d = np.load(f)
        for i, (key, lab, col) in enumerate(METHODS):
            m = np.array([d[f"{key}_{mk}"].mean() for mk, _ in METRICS])
            ci = np.array([1.96 * d[f"{key}_{mk}"].std(ddof=1) / np.sqrt(len(d[f"{key}_{mk}"])) for mk, _ in METRICS])
            ax.bar(x + (i - 1) * w, m, w, yerr=ci, capsize=2, label=lab, color=col)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("metric value")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l for _, l in METRICS], rotation=30, ha="right", fontsize=8)
    fig.suptitle("Faithfulness of attributions (mean ± 95% CI over flows); ↓ lower is better, ↑ higher is better", fontsize=10)
    save(fig, out, "faithfulness_metrics")


def xcs(out):
    f = "results/tables/xcs_both_clean_per_sample.npz"
    if not os.path.exists(f):
        print("  (xcs per-sample file missing, skipping)")
        return
    d = np.load(f)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    for key, lab, col in [("rollout_vs_shap_predclass", "attention rollout vs SHAP", "C0"),
                          ("attn_x_grad_vs_shap_predclass", "attention × |gradient| vs SHAP", "C2"),
                          ("gradient_only_vs_shap_predclass", "|gradient| alone vs SHAP", "C1")]:
        if key in d:
            r = d[key]
            ax.hist(r, bins=bins, alpha=0.45, color=col, label=f"{lab} (mean {np.nanmean(r):.2f})")
    ax.set_xlabel("per-flow Spearman ρ with GradientSHAP (predicted class)")
    ax.set_ylabel("flows")
    ax.legend(fontsize=8)
    ax.set_title("Consistency between built-in and post-hoc attributions (500 flows)", fontsize=10)
    save(fig, out, "xcs_distribution")


def latency(out):
    d = json.load(open("results/tables/explanation_latency_clean.json"))["results_ms_per_sample"]
    order = [("HiT-IDS prediction only (no explanation)", "HiT-IDS forward only"),
             ("HiT-IDS attention rollout", "HiT-IDS attention rollout"),
             ("HiT-IDS input x gradient", "HiT-IDS input × gradient"),
             ("XGBoost TreeSHAP", "XGBoost TreeSHAP"),
             ("Random Forest TreeSHAP", "Random Forest TreeSHAP"),
             ("Vanilla Transformer GradientSHAP", "Vanilla Transformer GradientSHAP"),
             ("HiT-IDS GradientSHAP", "HiT-IDS GradientSHAP"),
             ("BiLSTM GradientSHAP", "BiLSTM GradientSHAP")]
    labels = [l for k, l in order if k in d]
    single = [d[k]["single_ms_per_sample"]["median"] for k, _ in order if k in d]
    batched = [d[k]["batched_ms_per_sample"]["median"] for k, _ in order if k in d]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y - 0.2, single, 0.4, label="one flow (batch = 1)", color="C3")
    ax.barh(y + 0.2, batched, 0.4, label="batched (100 flows)", color="C0")
    for yi, s, b in zip(y, single, batched):
        ax.text(s * 1.15, yi - 0.2, f"{s:.2f}", va="center", fontsize=7)
        ax.text(b * 1.15, yi + 0.2, f"{b:.3f}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ms per flow (log scale, median of 5 repeats)")
    ax.set_xlim(1e-2, 8e3)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Explanation cost under one protocol (RTX 5090)", fontsize=10)
    save(fig, out, "latency_bars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/clean_rv_thr")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    faithfulness(a.out)
    xcs(a.out)
    latency(a.out)
    print(f"  figures written to {a.out}")

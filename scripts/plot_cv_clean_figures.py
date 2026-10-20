#!/usr/bin/env python3
"""
Figures for the clean CV protocol (Reviewer 2: confusion matrix, ROC, per-class).

Inputs : results/tables/cv_evaluation_<tag>.json
         results/tables/cv_preds_<tag>/fold*_<model>.npz  (y_true, y_pred, proba)
Outputs: results/figures/<tag>/
         cm_hit_ids.{png,pdf}            confusion matrix summed over folds (log colour)
         roc_per_class_hit_ids.{png,pdf} one-vs-rest ROC, pooled over folds
         pr_per_class_hit_ids.{png,pdf}  one-vs-rest precision–recall, pooled
         per_class_f1_models.{png,pdf}   per-class F1 (mean ± std) for all models
         binary_roc_models.{png,pdf}     attack-vs-benign ROC for all models

Usage:
    python scripts/plot_cv_clean_figures.py --tag clean
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn.metrics import (auc, precision_recall_curve, roc_curve,
                             average_precision_score, roc_auc_score)

DISPLAY = {
    "HiT-IDS_Cascade": "HiT-IDS", "random_forest": "Random Forest", "xgboost": "XGBoost",
    "bilstm": "BiLSTM", "vanilla_transformer": "Flat Transformer", "cnn_1d": "1D-CNN",
}
PLOT_MODELS = ["HiT-IDS_Cascade", "random_forest", "xgboost", "bilstm", "vanilla_transformer"]


def load_preds(pred_dir, model_key):
    ys, ps, prs = [], [], []
    for f in sorted(glob.glob(os.path.join(pred_dir, f"fold*_{model_key}.npz"))):
        d = np.load(f)
        ys.append(d["y_true"].astype(int))
        ps.append(d["y_pred"].astype(int))
        prs.append(d["proba"].astype(np.float32) if "proba" in d else None)
    if not ys:
        return None
    proba = np.concatenate(prs) if all(p is not None for p in prs) else None
    return np.concatenate(ys), np.concatenate(ps), proba


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cm(cm, names, out_dir, title):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm + 1, cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=max(cm.max(), 10)))
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            v = int(cm[i, j])
            if v:
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > cm.max() / 20 else "black")
    fig.colorbar(im, ax=ax, label="count (log scale)")
    save(fig, out_dir, "cm_hit_ids")


def plot_curves(y, proba, names, out_dir):
    C = len(names)
    fig_r, ax_r = plt.subplots(figsize=(7, 6))
    fig_p, ax_p = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab20")
    for c in range(C):
        yc = (y == c).astype(int)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        fpr, tpr, _ = roc_curve(yc, proba[:, c])
        ax_r.plot(fpr, tpr, color=cmap(c), lw=1.2,
                  label=f"{names[c]} (AUC {auc(fpr, tpr):.3f}, n={yc.sum():,})")
        pr, rc, _ = precision_recall_curve(yc, proba[:, c])
        ax_p.plot(rc, pr, color=cmap(c), lw=1.2,
                  label=f"{names[c]} (AP {average_precision_score(yc, proba[:, c]):.3f})")
    ax_r.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_r.set_xscale("log")
    ax_r.set_xlim(1e-5, 1)
    ax_r.set_xlabel("False positive rate (log)")
    ax_r.set_ylabel("True positive rate")
    ax_r.set_title("HiT-IDS one-vs-rest ROC (pooled over folds)")
    ax_r.legend(fontsize=6.5, loc="lower right")
    save(fig_r, out_dir, "roc_per_class_hit_ids")
    ax_p.set_xlabel("Recall")
    ax_p.set_ylabel("Precision")
    ax_p.set_title("HiT-IDS one-vs-rest precision–recall (pooled over folds)")
    ax_p.legend(fontsize=6.5, loc="lower left")
    save(fig_p, out_dir, "pr_per_class_hit_ids")


def plot_per_class_f1(cv, out_dir):
    names = cv["class_names"]
    S = cv["summary"]
    keys = [("HiT-IDS (Cascade)", "HiT-IDS"), ("random_forest", "Random Forest"),
            ("xgboost", "XGBoost"), ("bilstm", "BiLSTM"), ("vanilla_transformer", "Flat Transformer")]
    keys = [(k, n) for k, n in keys if k in S and S[k].get("per_class")]
    x = np.arange(len(names))
    w = 0.8 / max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for i, (k, n) in enumerate(keys):
        m = [S[k]["per_class"][c]["f1_mean"] for c in names]
        s = [S[k]["per_class"][c]["f1_std"] for c in names]
        ax.bar(x + (i - len(keys) / 2 + 0.5) * w, m, w, yerr=s, capsize=2, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 (mean ± std over folds)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=len(keys), loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    ax.set_title("Per-class F1 on real test folds")
    save(fig, out_dir, "per_class_f1_models")


def plot_binary_roc(pred_dir, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for key in PLOT_MODELS:
        d = load_preds(pred_dir, key)
        if d is None or d[2] is None:
            continue
        y, _, proba = d
        yb = (y > 0).astype(int)
        pa = 1.0 - proba[:, 0]
        fpr, tpr, _ = roc_curve(yb, pa)
        ax.plot(fpr, tpr, lw=1.3, label=f"{DISPLAY[key]} (AUC {roc_auc_score(yb, pa):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_xlabel("False positive rate (log)")
    ax.set_ylabel("Detection rate")
    ax.set_title("Attack-vs-benign ROC (pooled over folds)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out_dir, "binary_roc_models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--tables-dir", default="results/tables")
    args = ap.parse_args()
    cv_path = os.path.join(args.tables_dir, f"cv_evaluation_{args.tag}.json")
    pred_dir = os.path.join(args.tables_dir, f"cv_preds_{args.tag}")
    out_dir = os.path.join("results/figures", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(cv_path) as f:
        cv = json.load(f)
    names = cv["class_names"]

    hit = load_preds(pred_dir, "HiT-IDS_Cascade")
    if hit is not None:
        y, p, proba = hit
        cm = np.zeros((len(names), len(names)), dtype=np.int64)
        np.add.at(cm, (y, p), 1)
        plot_cm(cm, names, out_dir,
                f"HiT-IDS cascade confusion matrix ({cv['k_folds']}-fold CV, "
                f"{len(y):,} real test rows)")
        if proba is not None:
            plot_curves(y, proba, names, out_dir)
    plot_per_class_f1(cv, out_dir)
    plot_binary_roc(pred_dir, out_dir)
    print(f"  figures written to {out_dir}/")


if __name__ == "__main__":
    main()

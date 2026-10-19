#!/usr/bin/env python3
"""
Stability of the group-level explanation across CV folds.

For each of the K fold models of a clean CV run, compute the attention-rollout
group importance (temporal / volume / protocol / behavioral) of correctly
classified attack rows in that fold's REAL test rows, per attack class. Report
mean ± std across folds, and the inter-fold Spearman correlation of the 72-d
global feature importance. If the hierarchy is learning something real, the
per-class profiles should agree across independently trained models.

Writes results/tables/fold_group_importance_<tag>.json
       results/figures/<tag>/group_importance_folds.{png,pdf}

Usage:
    python scripts/fold_group_importance.py --tag clean_rv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.cv_data import load_clean_dataset, scale_fold, merge_labels_np, MERGED_NAMES, NUM_MERGED
from src.xai.attention_rollout import AttentionRollout
from scripts.calibrate_cascade_threshold import build, probs, cascade_predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTACKS = MERGED_NAMES[1:]


@torch.no_grad()
def group_importance(model, rollout, X, gnames, bs=2048):
    g, f = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=DEVICE)
        _, ex = model(xb, return_explanations=True)
        hi = rollout.hierarchical_importance(ex["raw_intra_attn"], ex["raw_inter_attn"], gnames)
        g.append(hi["group"].cpu().numpy())
        f.append(hi["global_flat"].cpu().numpy())
    return np.concatenate(g), np.concatenate(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--tag", default="clean_rv")
    ap.add_argument("--per-class-cap", type=int, default=2000)
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    gnames = list(config["feature_groups"].keys())
    rollout = AttentionRollout(residual_weight=config["xai"]["rollout"].get("residual_weight", 0.5),
                               head_fusion=config["xai"]["rollout"].get("head_fusion", "mean"))
    ckpt_dir = os.path.join(config["paths"]["checkpoints"], f"cv_{args.tag}")
    thr_path = os.path.join(config["paths"]["tables"], f"cv_evaluation_{args.tag}_thr.json")
    thresholds = None
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            thresholds = json.load(f)["summary"]["HiT-IDS (Cascade)"].get("stage1_thresholds")

    X_all, y15, _, gidx = load_clean_dataset(args.config)
    y_all = merge_labels_np(y15)
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    rng = np.random.RandomState(42)

    per_fold_class = np.full((K, len(ATTACKS), len(gnames)), np.nan)
    per_fold_global = []
    for fold_i, (tr, te) in enumerate(skf.split(X_all, y_all)):
        Xtr, Xte, _ = scale_fold(X_all[tr], X_all[te], gidx)
        yte = y_all[te]
        bin_m, atk_m = build(config, 2), build(config, NUM_MERGED - 1)
        bin_m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_i}_binary.pt"), map_location=DEVICE))
        atk_m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_i}_attack.pt"), map_location=DEVICE))
        t = thresholds[fold_i] if thresholds else 0.5
        pred = cascade_predict(probs(bin_m, Xte), probs(atk_m, Xte), t)
        sel = []
        for c in range(1, NUM_MERGED):
            ok = np.where((yte == c) & (pred == c))[0]
            if len(ok) > args.per_class_cap:
                ok = rng.choice(ok, args.per_class_cap, replace=False)
            sel.append(ok)
        sel = np.concatenate(sel)
        g, fl = group_importance(atk_m, rollout, Xte[sel], gnames)
        ysel = yte[sel]
        for c in range(1, NUM_MERGED):
            m = ysel == c
            if m.sum() > 0:
                per_fold_class[fold_i, c - 1] = g[m].mean(0)
        per_fold_global.append(fl.mean(0))
        print(f"  fold {fold_i+1}: {len(sel):,} rows, mean group importance "
              f"{dict(zip(gnames, np.round(g.mean(0), 3)))}")

    mean = np.nanmean(per_fold_class, axis=0)
    std = np.nanstd(per_fold_class, axis=0)
    G = np.array(per_fold_global)
    rho = np.array([[spearmanr(G[i], G[j])[0] for j in range(K)] for i in range(K)])
    off = rho[np.triu_indices(K, 1)]
    dominant = [gnames[int(np.argmax(mean[c]))] for c in range(len(ATTACKS))]
    dom_agree = [float(np.mean([gnames[int(np.argmax(per_fold_class[k, c]))] == dominant[c]
                                for k in range(K) if not np.isnan(per_fold_class[k, c]).any()]))
                 for c in range(len(ATTACKS))]
    out = {
        "tag": args.tag, "k_folds": K, "groups": gnames, "classes": ATTACKS,
        "per_class_mean": {c: dict(zip(gnames, mean[i].round(4).tolist())) for i, c in enumerate(ATTACKS)},
        "per_class_std": {c: dict(zip(gnames, std[i].round(4).tolist())) for i, c in enumerate(ATTACKS)},
        "dominant_group": dict(zip(ATTACKS, dominant)),
        "dominant_group_agreement_across_folds": dict(zip(ATTACKS, dom_agree)),
        "overall_mean": dict(zip(gnames, np.nanmean(mean, axis=0).round(4).tolist())),
        "interfold_spearman_global72": {"mean": float(off.mean()), "min": float(off.min()),
                                        "max": float(off.max()), "matrix": rho.round(4).tolist()},
        "stage1_thresholds_used": thresholds,
    }
    tp = os.path.join(config["paths"]["tables"], f"fold_group_importance_{args.tag}.json")
    with open(tp, "w") as f:
        json.dump(out, f, indent=2)

    fig_dir = os.path.join("results/figures", args.tag)
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(mean, cmap="Blues", vmin=0, vmax=max(0.5, np.nanmax(mean)))
    ax.set_xticks(range(len(gnames)))
    ax.set_xticklabels([g.capitalize() for g in gnames])
    ax.set_yticks(range(len(ATTACKS)))
    ax.set_yticklabels(ATTACKS, fontsize=8)
    for i in range(len(ATTACKS)):
        for j in range(len(gnames)):
            ax.text(j, i, f"{mean[i,j]:.2f}\n±{std[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mean[i, j] > 0.3 else "black")
    ax.set_title(f"Group importance per attack type (mean ± std over {K} fold models)")
    fig.colorbar(im, ax=ax, label="attention rollout group importance")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"group_importance_folds.{ext}"), dpi=200, bbox_inches="tight")
    print(f"\n  inter-fold Spearman of 72-d importance: mean {off.mean():.3f} (min {off.min():.3f})")
    print(f"  dominant-group agreement across folds: {out['dominant_group_agreement_across_folds']}")
    print(f"  wrote {tp}")


if __name__ == "__main__":
    main()

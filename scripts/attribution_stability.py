#!/usr/bin/env python3
"""
Is explanation instability specific to attention? Stability of FOUR attribution
methods across the K independently trained fold models:

    HiT-IDS attention rollout      (built-in)
    HiT-IDS GradientSHAP           (post-hoc, same model)
    XGBoost TreeSHAP (native)      (post-hoc, tree model trained on the same fold)
    Random Forest TreeSHAP         (post-hoc, tree model trained on the same fold)

For every fold model: attributions of the same (per-fold) attack test flows,
|value| of the predicted class, normalised per flow. From these:
    * global importance vector (mean over flows)  → inter-fold Spearman ρ (all pairs)
    * per-class group profile (sum of feature attributions within each semantic
      group, mean over flows of that true class) → dominant group agreement across folds
    * per-class inter-fold Spearman of the per-class feature profile

Writes results/tables/attribution_stability_<tag>.json and
       results/figures/<tag>/attribution_stability.{png,pdf}

Usage: python scripts/attribution_stability.py --tag clean_rv --n-shap 300 --n-tree 2000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.cv_data import load_clean_dataset, scale_fold, merge_labels_np, MERGED_NAMES, NUM_MERGED
from src.xai.attention_rollout import AttentionRollout
from src.xai.shap_analysis import SHAPAnalyzer
from scripts.calibrate_cascade_threshold import build, probs, cascade_predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTACKS = MERGED_NAMES[1:]


def norm(a):
    a = np.abs(np.asarray(a, dtype=np.float64))
    return a / (a.sum(1, keepdims=True) + 1e-12)


def rollout_attr(model, rollout, X, gnames, bs=2048):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            _, ex = model(torch.tensor(X[i:i+bs], dtype=torch.float32, device=DEVICE), return_explanations=True)
            out.append(rollout.hierarchical_importance(ex["raw_intra_attn"], ex["raw_inter_attn"], gnames)["global_flat"].cpu().numpy())
    return norm(np.concatenate(out))


def gshap_attr(model, X, bg):
    an = SHAPAnalyzer(model, torch.tensor(bg, dtype=torch.float32), device=DEVICE, method="gradient")
    sv = an.compute_shap_values(torch.tensor(X, dtype=torch.float32), max_samples=None)
    if sv.ndim == 2:
        sv = sv[:, :, None]
    with torch.no_grad():
        yh = model(torch.tensor(X, dtype=torch.float32, device=DEVICE), return_explanations=False).argmax(1).cpu().numpy()
    return norm(sv[np.arange(len(yh)), :, yh])


def xgb_attr(model, X):
    import xgboost as xgb
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)   # (N, C, F+1) or (N, F+1)
    if c.ndim == 3:
        yh = model.predict(X).astype(int)
        cls_index = {int(k): i for i, k in enumerate(getattr(model, "classes_", range(c.shape[1])))}
        c = c[np.arange(len(yh)), [cls_index[int(k)] for k in yh], :]
    return norm(c[:, :-1])


def rf_attr(model, X):
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    yh = model.predict(X).astype(int)
    cls_index = {int(k): i for i, k in enumerate(model.classes_)}
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)          # (N, F, C)
    return norm(sv[np.arange(len(yh)), :, [cls_index[int(k)] for k in yh]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--tag", default="clean_rv")
    ap.add_argument("--n-shap", type=int, default=300, help="flows per fold for GradientSHAP and RF TreeSHAP")
    ap.add_argument("--n-tree", type=int, default=2000, help="flows per fold for rollout and XGB TreeSHAP")
    ap.add_argument("--per-class-cap", type=int, default=200)
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    gnames = list(config["feature_groups"].keys())
    gsizes = [len(config["feature_groups"][g]) for g in gnames]
    gidx_list = []
    off = 0
    for sz in gsizes:
        gidx_list.append(list(range(off, off + sz)))
        off += sz
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
    cv_json = os.path.join(config["paths"]["tables"], f"cv_evaluation_{args.tag}.json")
    with open(cv_json) as f:
        cvj = json.load(f)
    K = cvj["k_folds"]
    if cvj.get("protocol", {}).get("smoke"):
        from src.data.cv_data import smoke_subsample
        X_all, y_all = smoke_subsample(X_all, y_all, frac=0.03)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    rng = np.random.RandomState(42)
    methods = ["HiT-IDS rollout", "HiT-IDS GradientSHAP", "XGBoost TreeSHAP", "Random Forest TreeSHAP"]
    glob = {m: [] for m in methods}                       # per fold: (F,)
    prof = {m: [] for m in methods}                       # per fold: dict class -> (F,)
    grp = {m: [] for m in methods}                        # per fold: dict class -> (G,)

    for fold_i, (tr, te) in enumerate(skf.split(X_all, y_all)):
        Xtr, Xte, _ = scale_fold(X_all[tr], X_all[te], gidx)
        ytr, yte = y_all[tr], y_all[te]
        tr_i, _ = train_test_split(np.arange(len(ytr)), test_size=0.10, stratify=ytr, random_state=42 + fold_i)
        bg = Xtr[tr_i][rng.choice(np.where(ytr[tr_i] > 0)[0], 100, replace=False)]

        bin_m, atk_m = build(config, 2), build(config, NUM_MERGED - 1)
        bin_m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_i}_binary.pt"), map_location=DEVICE))
        atk_m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_i}_attack.pt"), map_location=DEVICE))
        with open(os.path.join(ckpt_dir, f"fold{fold_i}_xgboost.pkl"), "rb") as f:
            xgbm = pickle.load(f)
        with open(os.path.join(ckpt_dir, f"fold{fold_i}_random_forest.pkl"), "rb") as f:
            rfm = pickle.load(f)
        t = thresholds[fold_i] if thresholds else 0.5
        pred = cascade_predict(probs(bin_m, Xte), probs(atk_m, Xte), t)

        sel = []
        for c in range(1, NUM_MERGED):
            ok = np.where((yte == c) & (pred == c))[0]
            if len(ok) > args.per_class_cap:
                ok = rng.choice(ok, args.per_class_cap, replace=False)
            sel.append(ok)
        sel = np.concatenate(sel)
        sel_small = rng.choice(sel, min(args.n_shap, len(sel)), replace=False)
        Xl, yl = Xte[sel], yte[sel]
        Xs, ys = Xte[sel_small], yte[sel_small]
        attrs = {
            "HiT-IDS rollout": (rollout_attr(atk_m, rollout, Xl, gnames), yl),
            "HiT-IDS GradientSHAP": (gshap_attr(atk_m, Xs, bg), ys),
            "XGBoost TreeSHAP": (xgb_attr(xgbm, Xl), yl),
            "Random Forest TreeSHAP": (rf_attr(rfm, Xs), ys),
        }
        for m, (A, y) in attrs.items():
            glob[m].append(A.mean(0))
            pc, pg = {}, {}
            for c in range(1, NUM_MERGED):
                mm = y == c
                if mm.sum() > 0:
                    v = A[mm].mean(0)
                    pc[MERGED_NAMES[c]] = v
                    pg[MERGED_NAMES[c]] = np.array([v[idx].sum() for idx in gidx_list])
            prof[m].append(pc)
            grp[m].append(pg)
        print(f"  fold {fold_i+1}: {len(sel):,} flows (rollout/XGB), {len(sel_small):,} flows (GradientSHAP/RF)")

    out = {"tag": args.tag, "k_folds": K, "groups": gnames, "methods": {}}
    for m in methods:
        G = np.array(glob[m])
        rho = [spearmanr(G[i], G[j])[0] for i in range(K) for j in range(i + 1, K)]
        per_class = {}
        for c in ATTACKS:
            folds_with = [k for k in range(K) if c in prof[m][k]]
            if len(folds_with) < 2:
                continue
            rc = [spearmanr(prof[m][a][c], prof[m][b][c])[0] for a in folds_with for b in folds_with if a < b]
            doms = [gnames[int(np.argmax(grp[m][k][c]))] for k in folds_with]
            dom = max(set(doms), key=doms.count)
            per_class[c] = {"interfold_rho_mean": float(np.nanmean(rc)), "dominant_group": dom,
                            "dominant_agreement": float(doms.count(dom) / len(doms)),
                            "group_mean": dict(zip(gnames, np.mean([grp[m][k][c] for k in folds_with], 0).round(4).tolist()))}
        out["methods"][m] = {"interfold_rho_global": {"mean": float(np.mean(rho)), "min": float(np.min(rho)), "max": float(np.max(rho))},
                             "per_class_rho_mean": float(np.nanmean([v["interfold_rho_mean"] for v in per_class.values()])),
                             "dominant_agreement_mean": float(np.mean([v["dominant_agreement"] for v in per_class.values()])),
                             "per_class": per_class}
        print(f"  {m:24s} inter-fold ρ(global) {np.mean(rho):.3f} [{np.min(rho):.3f}, {np.max(rho):.3f}] | "
              f"per-class ρ {out['methods'][m]['per_class_rho_mean']:.3f} | dominant-group agreement {out['methods'][m]['dominant_agreement_mean']:.2f}")
    tp = os.path.join(config["paths"]["tables"], f"attribution_stability_{args.tag}.json")
    with open(tp, "w") as f:
        json.dump(out, f, indent=2)

    fig_dir = os.path.join(config["paths"]["figures"], args.tag)
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(methods))
    g = [out["methods"][m]["interfold_rho_global"]["mean"] for m in methods]
    pcr = [out["methods"][m]["per_class_rho_mean"] for m in methods]
    da = [out["methods"][m]["dominant_agreement_mean"] for m in methods]
    ax.bar(x - 0.27, g, 0.27, label="inter-fold ρ, global importance")
    ax.bar(x, pcr, 0.27, label="inter-fold ρ, per-class profile")
    ax.bar(x + 0.27, da, 0.27, label="dominant-group agreement")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("stability across the 5 fold models")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"attribution_stability.{ext}"), dpi=300, bbox_inches="tight")
    print(f"  wrote {tp}")


if __name__ == "__main__":
    main()

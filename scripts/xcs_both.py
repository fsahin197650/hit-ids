#!/usr/bin/env python3
"""
XCS (attention–SHAP consistency) computed BOTH ways, on the deployment model,
so the paper can define exactly what was compared (Reviewer 3, item 8):

    attribution A1 : attention rollout            phi_attn  = R̂[CLS, 1:]  (hierarchical, normalised)
    attribution A2 : attention × |input-gradient| phi_axg   = phi_attn ⊙ |∂ z_ŷ / ∂ x|, normalised
                     (z_ŷ = logit of the predicted class of the attack model)
    reference   S1 : GradientSHAP, |SHAP| of the predicted class
    reference   S2 : GradientSHAP, mean_c |SHAP_c| over the 12 attack classes
                     (this is what the submitted paper used)

For every (A, S) pair: per-sample Spearman rho (mean/std/median, % with p<0.05)
and top-k overlap (k = 5, 10, 15). N correctly-classified attack rows from the
real test split; SHAP background = 100 attack rows from the training split.

Writes results/tables/xcs_both_clean.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.deployment import (load_config, load_deployment_models, load_split,
                                       cascade_probs, cascade_predict, read_threshold, ATTACK_CKPT)
from src.xai.attention_rollout import AttentionRollout
from src.xai.shap_analysis import SHAPAnalyzer


def normalise(a):
    a = np.abs(a)
    return a / (a.sum(1, keepdims=True) + 1e-12)


PER_SAMPLE = {}


def pair_stats(A, S, ks=(5, 10, 15), name=None):
    rhos, ps = [], []
    for a, s in zip(A, S):
        r, p = spearmanr(a, s)
        rhos.append(r)
        ps.append(p)
    rhos, ps = np.array(rhos), np.array(ps)
    if name:
        PER_SAMPLE[name] = rhos
    out = {"mean_rho": float(np.nanmean(rhos)), "std_rho": float(np.nanstd(rhos)),
           "median_rho": float(np.nanmedian(rhos)), "min_rho": float(np.nanmin(rhos)),
           "max_rho": float(np.nanmax(rhos)), "pct_significant_p05": float(np.mean(ps < 0.05) * 100)}
    for k in ks:
        ta = np.argsort(-A, axis=1)[:, :k]
        ts = np.argsort(-S, axis=1)[:, :k]
        out[f"top{k}_overlap"] = float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ta, ts)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bg", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    bin_m, atk_m, gnames = load_deployment_models(config, device)
    feature_names = [f for g in config["feature_groups"].values() for f in g]

    Xte, yte = load_split("test")
    Xtr, ytr = load_split("train")
    pb, pa = cascade_probs(bin_m, atk_m, Xte, device)
    pred = cascade_predict(pb, pa, read_threshold())
    ok = np.where((yte > 0) & (pred == yte))[0]
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(ok, min(args.n, len(ok)), replace=False)
    Xs = Xte[idx].to(device)
    ys = yte[idx] - 1
    bg = Xtr[rng.choice(np.where(ytr > 0)[0], args.bg, replace=False)]

    # A1 attention rollout
    rollout = AttentionRollout(residual_weight=config["xai"]["rollout"].get("residual_weight", 0.5),
                               head_fusion=config["xai"]["rollout"].get("head_fusion", "mean"))
    with torch.no_grad():
        logits, ex = atk_m(Xs, return_explanations=True)
        hi = rollout.hierarchical_importance(ex["raw_intra_attn"], ex["raw_inter_attn"], gnames)
    phi_attn = normalise(hi["global_flat"].cpu().numpy())
    group_imp = hi["group"].cpu().numpy()
    yhat = logits.argmax(1)

    # A2 attention × |gradient of predicted-class logit|
    xg = Xs.clone().requires_grad_(True)
    lg = atk_m(xg, return_explanations=False)
    lg.gather(1, yhat.unsqueeze(1)).sum().backward()
    grad = xg.grad.abs().cpu().numpy()
    phi_axg = normalise(phi_attn * grad)

    # S1 / S2 GradientSHAP
    analyzer = SHAPAnalyzer(atk_m, bg, device=device, method="gradient")
    sv = analyzer.compute_shap_values(Xs, max_samples=None)        # (N, 72, C)
    if sv.ndim == 2:
        sv = sv[:, :, None]
    yh = yhat.cpu().numpy()
    shap_pred = normalise(sv[np.arange(len(yh)), :, yh])
    shap_mean = normalise(np.abs(sv).mean(axis=2))

    results = {
        "protocol": {"model": ATTACK_CKPT, "n_samples": int(len(idx)),
                     "samples": "correctly classified attack rows of data/splits/test.pt",
                     "shap": f"GradientExplainer, nsamples=200, background={args.bg} attack rows of train split",
                     "gradient": "d logit(predicted class) / d x, absolute value",
                     "seed": args.seed},
        "pairs": {
            "rollout_vs_shap_predclass": pair_stats(phi_attn, shap_pred, name="rollout_vs_shap_predclass"),
            "rollout_vs_shap_meanclass": pair_stats(phi_attn, shap_mean, name="rollout_vs_shap_meanclass"),
            "attn_x_grad_vs_shap_predclass": pair_stats(phi_axg, shap_pred, name="attn_x_grad_vs_shap_predclass"),
            "attn_x_grad_vs_shap_meanclass": pair_stats(phi_axg, shap_mean, name="attn_x_grad_vs_shap_meanclass"),
            "gradient_only_vs_shap_predclass": pair_stats(normalise(grad), shap_pred, name="gradient_only_vs_shap_predclass"),
            "rollout_vs_attn_x_grad": pair_stats(phi_attn, phi_axg, name="rollout_vs_attn_x_grad"),
        },
        "global_top10": {
            "rollout": [feature_names[i] for i in np.argsort(-phi_attn.mean(0))[:10]],
            "attn_x_grad": [feature_names[i] for i in np.argsort(-phi_axg.mean(0))[:10]],
            "shap_predclass": [feature_names[i] for i in np.argsort(-shap_pred.mean(0))[:10]],
        },
        "global_importance": {
            "rollout": phi_attn.mean(0).tolist(), "attn_x_grad": phi_axg.mean(0).tolist(),
            "shap_predclass": shap_pred.mean(0).tolist(), "shap_meanclass": shap_mean.mean(0).tolist(),
            "feature_names": feature_names,
        },
        "group_importance_mean": dict(zip(gnames, group_imp.mean(0).tolist())),
        "group_importance_per_class": {
            cname: dict(zip(gnames, group_imp[ys == c].mean(0).tolist()))
            for c, cname in enumerate([n for n in ["Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
                                                   "DoS Slowhttptest", "DoS slowloris", "FTP-Patator",
                                                   "Heartbleed", "Infiltration", "PortScan",
                                                   "SSH-Patator", "Web Attack"]])
            if (ys == c).sum() > 0},
        "class_counts_in_sample": {str(c): int((ys == c).sum()) for c in np.unique(ys)},
    }
    np.savez_compressed("results/tables/xcs_both_clean_per_sample.npz", **PER_SAMPLE)
    out = "results/tables/xcs_both_clean.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n  pair                                 mean rho   median   %sig   top10")
    for k, v in results["pairs"].items():
        print(f"  {k:36s} {v['mean_rho']:.3f}    {v['median_rho']:.3f}   {v['pct_significant_p05']:5.1f}   {v['top10_overlap']:.3f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

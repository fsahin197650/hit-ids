#!/usr/bin/env python3
"""
Explanation experiments on NSM-IDS-2026 using the session-CV fold models of
scripts/nsm_session_cv.py:

  (1) faithfulness on the reference test sessions {3,7,11} (fold 1 model):
      attention rollout / input gradient / GradientSHAP / random, the eleven
      deletion–insertion–AOPC–comprehensiveness–sufficiency metrics, paired tests
  (2) stability of rollout / GradientSHAP / XGBoost TreeSHAP / RF TreeSHAP
      across the four session folds (same definitions as attribution_stability.py)

Writes results/tables/nsm_faithfulness.json, results/tables/nsm_faithfulness_per_sample.npz,
       results/tables/nsm_attribution_stability.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import yaml
from scipy import stats
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.nsm_session_cv import load_nsm, nsm_config, build, FOLDS, CLASSES, META
from scripts.attribution_stability import rollout_attr, gshap_attr, xgb_attr, rf_attr, norm
from scripts.calibrate_cascade_threshold import probs
from src.data.cv_data import scale_fold
from src.xai.attention_rollout import AttentionRollout
from src.xai.faithfulness import ComprehensiveFaithfulness

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOWER = {"deletion_auc", "sufficiency_k5", "sufficiency_k10", "sufficiency_k15", "sufficiency_k20"}


def grad_attr(model, X, y):
    out = []
    for i in range(0, len(X), 256):
        xb = torch.tensor(X[i:i+256], dtype=torch.float32, device=DEVICE).requires_grad_(True)
        yb = torch.tensor(y[i:i+256], device=DEVICE)
        lg = model(xb, return_explanations=False)
        lg.gather(1, yb.unsqueeze(1)).sum().backward()
        out.append(xb.grad.abs().cpu().numpy())
        model.zero_grad()
    return norm(np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--csv", default="data/nsm/NSM-IDS-2026_ML_v1.2.csv")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tag", default="nsm")
    ap.add_argument("--n-small", type=int, default=300, help="flows per fold for GradientSHAP / RF TreeSHAP")
    ap.add_argument("--skip-faithfulness", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    tables = config["paths"]["tables"]
    cvj = json.load(open(os.path.join(tables, f"{args.tag}_session_cv.json")))
    ckpt_dir = os.path.join(config["paths"]["checkpoints"], args.tag)
    d, y_all = load_nsm(args.csv)
    cfg = nsm_config(config, set(c for c in d.columns if c not in META))
    feats = cvj["protocol"]["features"]
    gnames = list(cfg["feature_groups"].keys())
    gidx, gidx_list, off = {}, [], 0
    for g in gnames:
        n = len(cfg["feature_groups"][g])
        gidx[g] = list(range(off, off + n))
        gidx_list.append(gidx[g])
        off += n
    X_all = d[feats].values.astype(np.float32)
    sess = d["session_id"].values
    if cvj["protocol"].get("smoke"):
        keep = np.random.RandomState(0).rand(len(y_all)) < 0.15
        X_all, y_all, sess = X_all[keep], y_all[keep], sess[keep]
    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")
    C = len(CLASSES)
    rng = np.random.RandomState(42)
    thresholds = cvj["summary"]["HiT-IDS (Cascade)"]["stage1_thresholds"]

    def fold_data(fold_i):
        test_s = FOLDS[fold_i]
        train_s = [s for s in range(12) if s not in test_s]
        val_s = train_s[fold_i % len(train_s)]
        tr_s = [s for s in train_s if s != val_s]
        m_tr, m_te = np.isin(sess, tr_s), np.isin(sess, test_s)
        Xtr, Xte, _ = scale_fold(X_all[m_tr], X_all[m_te], gidx)
        return Xtr, y_all[m_tr], Xte, y_all[m_te]

    def load_fold(fold_i):
        bin_m, atk_m = build(cfg, 2).eval(), build(cfg, C - 1).eval()
        bin_m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_i}_binary.pt"), map_location=DEVICE))
        atk_m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_i}_attack.pt"), map_location=DEVICE))
        return bin_m, atk_m

    # ── (1) faithfulness on fold 1 ───────────────────────────────────────
    if args.skip_faithfulness:
        print("  (faithfulness skipped)")
    Xtr, ytr, Xte, yte = fold_data(0)
    if not args.skip_faithfulness:
        bin_m, atk_m = load_fold(0)
        pb, pa = probs(bin_m, Xte), probs(atk_m, Xte)
        pred = np.where(pb[:, 1] >= thresholds[0], pa.argmax(1) + 1, 0)
        ok = np.where((yte > 0) & (pred == yte))[0]
        sel = rng.choice(ok, min(args.n, len(ok)), replace=False)
        Xe, ye = Xte[sel], yte[sel] - 1
        bg = Xtr[rng.choice(np.where(ytr > 0)[0], 100, replace=False)]
        attrs = {"attention": rollout_attr(atk_m, rollout, Xe, gnames), "gradient": grad_attr(atk_m, Xe, ye),
                 "shap": gshap_attr(atk_m, Xe, bg), "random": norm(rng.rand(len(Xe), Xe.shape[1]))}
        ev = ComprehensiveFaithfulness(atk_m, device=DEVICE, masking_value=0.0)
        Xt, yt = torch.tensor(Xe, dtype=torch.float32), torch.tensor(ye, dtype=torch.long)
        res, per_sample = {}, {}
        for m, A in attrs.items():
            r = ev.evaluate_all(Xt, A, yt, k_values=[5, 10, 15, 20], steps=20)
            res[m] = {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in r.items() if isinstance(v, dict) and "mean" in v}
            for k, v in r.items():
                if isinstance(v, dict) and "per_sample" in v:
                    per_sample[f"{m}_{k}"] = np.asarray(v["per_sample"])
        tests = {}
        for other in ("random", "gradient", "shap"):
            rows = []
            for k in res["attention"]:
                a, b = per_sample[f"attention_{k}"], per_sample[f"{other}_{k}"]
                t, p2 = stats.ttest_rel(a, b)
                better = (a.mean() < b.mean()) if k in LOWER else (a.mean() > b.mean())
                rows.append({"metric": k, "attention_mean": float(a.mean()), f"{other}_mean": float(b.mean()),
                             "attention_better": bool(better), "p_two_sided": float(p2)})
            tests[f"attention_vs_{other}"] = rows
            wins = sum(r["attention_better"] and r["p_two_sided"] < 0.05 for r in rows)
            print(f"  faithfulness: attention better than {other} on {wins}/11 metrics (p<0.05)")
        with open(os.path.join(tables, f"{args.tag}_faithfulness.json"), "w") as f:
            json.dump({"dataset": "NSM-IDS-2026", "fold": 0, "test_sessions": FOLDS[0], "n_samples": int(len(sel)),
                       "n_features": int(Xe.shape[1]), "results": res, "tests": tests}, f, indent=2)
        np.savez(os.path.join(tables, f"{args.tag}_faithfulness_per_sample.npz"), **per_sample)

    # ── (2) stability across the 4 session folds ─────────────────────────
    methods = ["HiT-IDS rollout", "HiT-IDS GradientSHAP", "XGBoost TreeSHAP", "Random Forest TreeSHAP"]
    glob, prof, grp = ({m: [] for m in methods} for _ in range(3))
    for fold_i in range(len(FOLDS)):
        Xtr, ytr, Xte, yte = fold_data(fold_i)
        bin_m, atk_m = load_fold(fold_i)
        with open(os.path.join(ckpt_dir, f"fold{fold_i}_xgboost.pkl"), "rb") as f:
            xgbm = pickle.load(f)
        with open(os.path.join(ckpt_dir, f"fold{fold_i}_random_forest.pkl"), "rb") as f:
            rfm = pickle.load(f)
        pb, pa = probs(bin_m, Xte), probs(atk_m, Xte)
        pred = np.where(pb[:, 1] >= thresholds[fold_i], pa.argmax(1) + 1, 0)
        sel = []
        for c in range(1, C):
            ok = np.where((yte == c) & (pred == c))[0]
            sel.append(rng.choice(ok, min(200, len(ok)), replace=False))
        sel = np.concatenate(sel)
        small = rng.choice(sel, min(args.n_small, len(sel)), replace=False)
        bg = Xtr[rng.choice(np.where(ytr > 0)[0], 100, replace=False)]
        A = {"HiT-IDS rollout": (rollout_attr(atk_m, rollout, Xte[sel], gnames), yte[sel]),
             "HiT-IDS GradientSHAP": (gshap_attr(atk_m, Xte[small], bg), yte[small]),
             "XGBoost TreeSHAP": (xgb_attr(xgbm, Xte[sel]), yte[sel]),
             "Random Forest TreeSHAP": (rf_attr(rfm, Xte[small]), yte[small])}
        for m, (a, y) in A.items():
            glob[m].append(a.mean(0))
            pc, pg = {}, {}
            for c in range(1, C):
                mm = y == c
                if mm.sum() > 0:
                    v = a[mm].mean(0)
                    pc[CLASSES[c]] = v
                    pg[CLASSES[c]] = np.array([v[idx].sum() for idx in gidx_list])
            prof[m].append(pc)
            grp[m].append(pg)
        print(f"  stability fold {fold_i+1}: {len(sel)} / {len(small)} flows")
    K = len(FOLDS)
    out = {"dataset": "NSM-IDS-2026", "k_folds": K, "groups": gnames, "methods": {}}
    for m in methods:
        G = np.array(glob[m])
        rho = [spearmanr(G[i], G[j])[0] for i in range(K) for j in range(i + 1, K)]
        pcs = {}
        for c in CLASSES[1:]:
            fw = [k for k in range(K) if c in prof[m][k]]
            if len(fw) < 2:
                continue
            rc = [spearmanr(prof[m][a][c], prof[m][b][c])[0] for a in fw for b in fw if a < b]
            doms = [gnames[int(np.argmax(grp[m][k][c]))] for k in fw]
            dom = max(set(doms), key=doms.count)
            pcs[c] = {"interfold_rho_mean": float(np.nanmean(rc)), "dominant_group": dom,
                      "dominant_agreement": float(doms.count(dom) / len(doms))}
        out["methods"][m] = {"interfold_rho_global": {"mean": float(np.mean(rho)), "min": float(np.min(rho)), "max": float(np.max(rho))},
                             "per_class_rho_mean": float(np.nanmean([v["interfold_rho_mean"] for v in pcs.values()])),
                             "dominant_agreement_mean": float(np.mean([v["dominant_agreement"] for v in pcs.values()])),
                             "per_class": pcs}
        print(f"  {m:24s} inter-fold ρ {np.mean(rho):.3f} | per-class ρ {out['methods'][m]['per_class_rho_mean']:.3f} "
              f"| dominant agreement {out['methods'][m]['dominant_agreement_mean']:.2f}")
    with open(os.path.join(tables, f"{args.tag}_attribution_stability.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("  nsm_xai done")


if __name__ == "__main__":
    main()

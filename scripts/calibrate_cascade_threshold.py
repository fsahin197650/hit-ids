#!/usr/bin/env python3
"""
Post-hoc Stage-1 threshold calibration for the HiT-IDS cascade.

Why: the binary stage is trained on a resampled set (BENIGN capped at 200K vs
~440K attack rows) and deployed with argmax (p_attack >= 0.5). On the real
83%-benign distribution this over-triggers (FPR ~0.9%), and the attack stage
then has to assign every leaked benign flow to SOME attack class — most land
in "Bot", which is what destroys Bot precision.

What: for each CV fold, re-create the exact fold / real-validation split used
by evaluate_cv.py (--val-source real), load the saved fold checkpoints, choose
the Stage-1 threshold t* that maximises 13-class macro-F1 of the full cascade
on the REAL validation rows, then evaluate on the REAL test fold with t*.
The test fold is never used to pick the threshold.

Writes results/tables/cv_evaluation_<in_tag>_thr.json (same schema as the CV
file, HiT-IDS rows replaced, baselines carried over, stats recomputed) plus
per-fold predictions, so make_paper_tables.py / plot_cv_clean_figures.py work
with --tag <in_tag>_thr.

Usage:
    python scripts/calibrate_cascade_threshold.py --tag clean_rv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.cv_data import load_clean_dataset, scale_fold
from src.models.hit_ids import HiTIDS
from scripts.evaluate_cv import (merge_labels_np, make_hit_groups, fold_metrics,
                                 save_fold_predictions, cohens_d, MERGED_NAMES, NUM_MERGED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build(config, num_classes):
    fg = config["feature_groups"]
    names = list(fg.keys())
    mc = config["model"]
    m = HiTIDS(group_sizes=[len(fg[g]) for g in names], group_names=names,
               d_emb=mc["d_emb"], nhead=mc["nhead"], intra_layers=mc["intra_group_layers"],
               inter_layers=mc["inter_group_layers"], dim_feedforward=mc["dim_feedforward"],
               dropout=mc["dropout"], num_classes=num_classes)
    return m.to(DEVICE).eval()


@torch.no_grad()
def probs(model, X, bs=4096):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=DEVICE)
        out.append(torch.softmax(model(xb, return_explanations=False), 1).cpu().numpy())
    return np.concatenate(out)


def cascade_predict(p_bin, p_atk, t):
    pred = np.zeros(len(p_bin), dtype=np.int64)
    m = p_bin[:, 1] >= t
    pred[m] = p_atk[m].argmax(1) + 1
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--tag", default="clean_rv", help="tag of the CV run to calibrate")
    ap.add_argument("--grid", type=int, default=199, help="thresholds in (0,1)")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    tables = config["paths"]["tables"]
    cv_path = os.path.join(tables, f"cv_evaluation_{args.tag}.json")
    with open(cv_path) as f:
        cv = json.load(f)
    assert "REAL training-fold rows" in cv["protocol"].get("validation_for_early_stopping", ""), \
        "this CV run did not carve a real validation split; nothing to calibrate on"
    K = cv["k_folds"]
    ckpt_dir = os.path.join(config["paths"]["checkpoints"], f"cv_{args.tag}")
    out_tag = f"{args.tag}_thr"
    pred_dir = os.path.join(tables, f"cv_preds_{out_tag}")

    X_all, y15, _, gidx = load_clean_dataset(args.config)
    y_all = merge_labels_np(y15)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    grid = np.linspace(0.005, 0.995, args.grid)

    new_rows, thresholds = [], []
    for fold_i, (tr, te) in enumerate(skf.split(X_all, y_all)):
        Xtr, ytr, Xte, yte = X_all[tr], y_all[tr], X_all[te], y_all[te]
        Xtr, Xte, _ = scale_fold(Xtr, Xte, gidx)                     # identical to evaluate_cv
        tr_i, va_i = train_test_split(np.arange(len(ytr)), test_size=0.10,
                                      stratify=ytr, random_state=42 + fold_i)
        Xva, yva = Xtr[va_i], ytr[va_i]

        bin_m, atk_m = build(config, 2), build(config, NUM_MERGED - 1)
        bin_m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_i}_binary.pt"), map_location=DEVICE))
        atk_m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_i}_attack.pt"), map_location=DEVICE))

        pb_va, pa_va = probs(bin_m, Xva), probs(atk_m, Xva)
        scores = [f1_score(yva, cascade_predict(pb_va, pa_va, t), average="macro", zero_division=0)
                  for t in grid]
        t_star = float(grid[int(np.argmax(scores))])
        f_default = f1_score(yva, cascade_predict(pb_va, pa_va, 0.5), average="macro", zero_division=0)

        pb_te, pa_te = probs(bin_m, Xte), probs(atk_m, Xte)
        proba = np.concatenate([pb_te[:, :1], pb_te[:, 1:2] * pa_te], axis=1)
        y_pred = cascade_predict(pb_te, pa_te, t_star)
        r = fold_metrics(yte, y_pred, proba)
        r["stage1_threshold"] = t_star
        r["val_macro_f1_at_threshold"] = float(max(scores))
        r["val_macro_f1_at_0.5"] = float(f_default)
        r["test_macro_f1_at_0.5"] = float(f1_score(yte, cascade_predict(pb_te, pa_te, 0.5),
                                                   average="macro", zero_division=0))
        new_rows.append(r)
        thresholds.append(t_star)
        save_fold_predictions(pred_dir, fold_i, "HiT-IDS (Cascade)", yte, y_pred, proba)
        print(f"  fold {fold_i+1}: t*={t_star:.3f}  val F1 {f_default:.4f}->{max(scores):.4f}  "
              f"test F1 {r['test_macro_f1_at_0.5']:.4f}->{r['macro_f1']:.4f}  "
              f"Bot F1 {r['per_class']['Bot']['f1']:.3f} (P={r['per_class']['Bot']['precision']:.3f})  "
              f"FPR {r['binary']['fpr']*100:.2f}%")

    # ── Baselines: same calibration on the saved validation probabilities ──
    src_pred = os.path.join(tables, f"cv_preds_{args.tag}")
    bl_thresholds = {}
    for bl in [m for m in cv["per_fold_results"] if m != "HiT-IDS (Cascade)"]:
        rows = []
        for fold_i in range(K):
            fv = os.path.join(src_pred, f"fold{fold_i}_{bl}_val.npz")
            ft = os.path.join(src_pred, f"fold{fold_i}_{bl}.npz")
            if not (os.path.exists(fv) and os.path.exists(ft)):
                rows = None
                break
            v, t = np.load(fv), np.load(ft)
            yv, pv = v["y_true"].astype(int), v["proba"].astype(np.float32)
            yt, pt = t["y_true"].astype(int), t["proba"].astype(np.float32)

            def pred_at(p, th):
                out = np.zeros(len(p), dtype=np.int64)
                m = (1.0 - p[:, 0]) >= th
                out[m] = p[m, 1:].argmax(1) + 1
                return out
            sc = [f1_score(yv, pred_at(pv, th), average="macro", zero_division=0) for th in grid]
            th = float(grid[int(np.argmax(sc))])
            yp = pred_at(pt, th)
            r = fold_metrics(yt, yp, pt)
            r["stage1_threshold"] = th
            r["test_macro_f1_at_argmax"] = float(f1_score(yt, pt.argmax(1), average="macro", zero_division=0))
            rows.append(r)
            save_fold_predictions(pred_dir, fold_i, bl, yt, yp, pt)
            bl_thresholds.setdefault(bl, []).append(th)
        if rows:
            cv["per_fold_results"][bl] = rows
            a = np.array([r["macro_f1"] for r in rows])
            S_bl = cv["summary"][bl]
            S_bl.update({"mean_macro_f1": float(a.mean()), "std_macro_f1": float(a.std()),
                         "min_macro_f1": float(a.min()), "max_macro_f1": float(a.max()), "per_fold_f1": a.tolist(),
                         "mean_accuracy": float(np.mean([r["accuracy"] for r in rows])),
                         "mean_weighted_f1": float(np.mean([r["weighted_f1"] for r in rows])),
                         "stage1_thresholds": bl_thresholds[bl],
                         "mean_macro_f1_argmax": float(np.mean([r["test_macro_f1_at_argmax"] for r in rows]))})
            S_bl["per_class"] = {c: {**{f"{k}_mean": float(np.mean([r["per_class"][c][k] for r in rows])) for k in ("precision", "recall", "f1")},
                                     **{f"{k}_std": float(np.std([r["per_class"][c][k] for r in rows])) for k in ("precision", "recall", "f1")},
                                     "support_per_fold": [r["per_class"][c]["support"] for r in rows]} for c in MERGED_NAMES}
            S_bl["binary"] = {k: {"mean": float(np.mean([r["binary"][k] for r in rows])), "std": float(np.std([r["binary"][k] for r in rows]))}
                              for k in rows[0]["binary"]}
            S_bl["confusion_matrix_sum"] = np.sum([np.array(r["confusion_matrix"]) for r in rows], axis=0).tolist()
            print(f"  {bl}: calibrated thresholds {np.round(bl_thresholds[bl], 3).tolist()} → macro-F1 {a.mean():.4f}")

    # ── Assemble a CV file with the same schema ───────────────────────────
    cv["per_fold_results"]["HiT-IDS (Cascade)"] = new_rows
    arr = np.array([r["macro_f1"] for r in new_rows])
    S = cv["summary"]["HiT-IDS (Cascade)"]
    S.update({"mean_macro_f1": float(arr.mean()), "std_macro_f1": float(arr.std()),
              "min_macro_f1": float(arr.min()), "max_macro_f1": float(arr.max()),
              "per_fold_f1": arr.tolist(),
              "mean_accuracy": float(np.mean([r["accuracy"] for r in new_rows])),
              "mean_weighted_f1": float(np.mean([r["weighted_f1"] for r in new_rows])),
              "stage1_thresholds": thresholds})
    S["per_class"] = {c: {**{f"{k}_mean": float(np.mean([r["per_class"][c][k] for r in new_rows])) for k in ("precision", "recall", "f1")},
                          **{f"{k}_std": float(np.std([r["per_class"][c][k] for r in new_rows])) for k in ("precision", "recall", "f1")},
                          "support_per_fold": [r["per_class"][c]["support"] for r in new_rows]}
                      for c in MERGED_NAMES}
    for key in ("macro_roc_auc_ovr", "macro_pr_auc"):
        S[f"mean_{key}"] = float(np.mean([r[key] for r in new_rows if key in r]))
    S["binary"] = {k: {"mean": float(np.mean([r["binary"][k] for r in new_rows])),
                       "std": float(np.std([r["binary"][k] for r in new_rows]))}
                   for k in new_rows[0]["binary"]}
    S["confusion_matrix_sum"] = np.sum([np.array(r["confusion_matrix"]) for r in new_rows], axis=0).tolist()

    tests = {}
    for bl, rows in cv["per_fold_results"].items():
        if bl == "HiT-IDS (Cascade)" or len(rows) < K:
            continue
        b = np.array([r["macro_f1"] for r in rows])
        delta = arr - b
        _, p = stats.ttest_rel(arr, b)          # two-sided
        d = cohens_d(arr, b)
        tcrit = stats.t.ppf(0.975, K - 1)
        tests[bl] = {"test": "Paired t-test (two-sided)", "delta_mean": float(delta.mean()),
                     "delta_ci": [float(delta.mean() - tcrit * delta.std(ddof=1) / np.sqrt(K)),
                                  float(delta.mean() + tcrit * delta.std(ddof=1) / np.sqrt(K))],
                     "cohens_dz": float(delta.mean() / delta.std(ddof=1)),
                     "cohens_d": float(d),
                     "effect_size": "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small",
                     "p_value": float(p), "significant": bool(p < 0.05)}
    cv["statistical_tests"] = tests
    cv["protocol"]["stage1_threshold"] = ("chosen per fold on the REAL validation split by maximising "
                                          "13-class macro-F1; test fold untouched; applied to HiT-IDS AND to every "
                                          "baseline with saved validation probabilities (attack score = 1 - p(BENIGN))")
    cv["protocol"]["baseline_thresholds"] = bl_thresholds
    cv["protocol"]["source_cv_tag"] = args.tag
    out = os.path.join(tables, f"cv_evaluation_{out_tag}.json")
    with open(out, "w") as f:
        json.dump(cv, f, indent=2)
    # Baselines without saved validation probabilities keep their argmax predictions
    for fn in os.listdir(src_pred):
        if "HiT-IDS" not in fn and "_val" not in fn and not os.path.exists(os.path.join(pred_dir, fn)):
            os.symlink(os.path.abspath(os.path.join(src_pred, fn)), os.path.join(pred_dir, fn))
    print(f"\n  HiT-IDS macro-F1 with calibrated Stage-1: {arr.mean():.4f} ± {arr.std():.4f} "
          f"(thresholds {np.round(thresholds, 3).tolist()})\n  wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NSM-IDS-2026 (Zenodo record 23392184, v1.1 ML table): session-grouped
cross-validation with the SAME fold-internal protocol as CIC-IDS2017.

Protocol
    * exact duplicate rows (features + label) removed
    * 12 capture sessions → 4 folds of 3 sessions; fold 1 is the dataset's own
      reference test split {3, 7, 11}; the others are {0,4,8}, {1,5,9}, {2,6,10}
    * inside each fold: one training session held out as the real validation
      session (early stopping + Stage-1 threshold); RobustScaler per group and
      adaptive SMOTE fitted on the remaining training sessions only
    * features: the HiT-IDS groups of configs/config.yaml intersected with the
      NSM ML table (the 8 Active/Idle and the Subflow columns are absent in v1.1)
    * models: HiT-IDS cascade (calibrated) + RF, XGBoost, BiLSTM, Vanilla Transformer
      (each with the same validation-calibrated attack threshold)

Writes results/tables/nsm_session_cv.json, results/tables/nsm_preds/,
       results/checkpoints/nsm/fold{i}_*.{pt,pkl}

Usage: python scripts/nsm_session_cv.py --csv <path to NSM-IDS-2026_ML_v1.1.csv>
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.evaluate_cv as evcv
from scripts.evaluate_cv import (apply_smote_fold, train_hit_model, phase2_refine, cohens_d,
                                 eval_cascade_on_test_full, set_seed, save_fold_predictions,
                                 train_eval_baseline_fold)
from src.data.cv_data import scale_fold
from src.models.hit_ids import HiTIDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLDS = [[3, 7, 11], [0, 4, 8], [1, 5, 9], [2, 6, 10]]
CLASSES = ["BENIGN", "DoS GoldenEye", "DoS SYN", "DoS Slowhttptest", "DoS Slowloris",
           "FTP-Patator", "Infiltration", "PortScan", "SSH-Patator", "Web Attack"]


META = ("Label", "session_id", "split_fragment", "parent_label")


def load_nsm(csv, include_fragments=False):
    """v1.2 ML table. Reverse-direction CICFlowMeter tails (split_fragment=1,
    22.9% of rows) are excluded by default, as the dataset authors recommend."""
    d = pd.read_csv(csv)
    d.columns = d.columns.str.strip()
    n0 = len(d)
    if "split_fragment" in d.columns and not include_fragments:
        d = d[d["split_fragment"] == 0]
        print(f"  NSM rows {n0:,} → {len(d):,} after excluding split fragments")
    feats = [c for c in d.columns if c not in META]
    n1 = len(d)
    d = d.drop_duplicates(subset=feats + ["Label"]).reset_index(drop=True)
    print(f"  NSM rows {n1:,} → {len(d):,} after exact de-duplication")
    y = d["Label"].map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
    assert not np.isnan(y.astype(float)).any(), "unknown label"
    return d, y


def nsm_config(config, columns):
    cfg = json.loads(json.dumps(config))
    cfg["feature_groups"] = {g: [f for f in feats if f in columns] for g, feats in config["feature_groups"].items()}
    return cfg


def fold_metrics_nsm(y_true, y_pred, proba):
    """fold_metrics() with the NSM class list."""
    old = (evcv.MERGED_NAMES, evcv.NUM_MERGED)
    evcv.MERGED_NAMES, evcv.NUM_MERGED = CLASSES, len(CLASSES)
    try:
        return evcv.fold_metrics(y_true, y_pred, proba)
    finally:
        evcv.MERGED_NAMES, evcv.NUM_MERGED = old


def build(cfg, nc):
    fg = cfg["feature_groups"]
    names = list(fg.keys())
    mc = cfg["model"]
    return HiTIDS(group_sizes=[len(fg[g]) for g in names], group_names=names, d_emb=mc["d_emb"], nhead=mc["nhead"],
                  intra_layers=mc["intra_group_layers"], inter_layers=mc["inter_group_layers"],
                  dim_feedforward=mc["dim_feedforward"], dropout=mc["dropout"], num_classes=nc).to(DEVICE)


def cascade_cal(bin_m, atk_m, Xv, yv, Xt, yt, gv, gt, bs, grid):
    """Calibrate Stage-1 threshold on the validation session, evaluate on test sessions."""
    _, _, pv = eval_cascade_on_test_full(bin_m, atk_m, Xv, yv, gv, bs, DEVICE)
    _, _, pt = eval_cascade_on_test_full(bin_m, atk_m, Xt, yt, gt, bs, DEVICE)

    def pred(p, th):  # p = combined proba: p[:,0] benign, p[:,1:] = p_attack * p_type
        out = np.zeros(len(p), dtype=np.int64)
        m = (1 - p[:, 0]) >= th
        out[m] = p[m, 1:].argmax(1) + 1
        return out
    sc = [f1_score(yv, pred(pv, th), average="macro", zero_division=0) for th in grid]
    th = float(grid[int(np.argmax(sc))])
    return th, pred(pt, th), pt, pred(pt, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--csv", default="data/nsm/NSM-IDS-2026_ML_v1.2.csv")
    ap.add_argument("--include-fragments", action="store_true")
    ap.add_argument("--smote-target", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-phase2", action="store_true")
    args = ap.parse_args()
    if args.no_phase2:
        import scripts.evaluate_cv as _evcv
        _evcv.PHASE2 = False
    if args.smoke:
        evcv.EPOCH_CAP = 1
    with open(args.config) as f:
        config = yaml.safe_load(f)
    tables, ckpt_root = config["paths"]["tables"], config["paths"]["checkpoints"]
    tag = "nsm" + ("_smoke" if args.smoke else "")
    pred_dir, ckpt_dir = os.path.join(tables, f"{tag}_preds"), os.path.join(ckpt_root, tag)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    d, y_all = load_nsm(args.csv, args.include_fragments)
    cfg = nsm_config(config, set(c for c in d.columns if c not in META))
    feats = [f for g in cfg["feature_groups"].values() for f in g]
    gidx, off = {}, 0
    for g, fs in cfg["feature_groups"].items():
        gidx[g] = list(range(off, off + len(fs)))
        off += len(fs)
    X_all = d[feats].values.astype(np.float32)
    sess = d["session_id"].values
    print(f"  features used: {len(feats)} {dict((g, len(v)) for g, v in cfg['feature_groups'].items())}")
    if args.smoke:
        keep = np.random.RandomState(0).rand(len(y_all)) < 0.15
        X_all, y_all, sess = X_all[keep], y_all[keep], sess[keep]

    baselines = ["random_forest", "xgboost", "bilstm", "vanilla_transformer"]
    results = {"HiT-IDS (Cascade)": []}
    results.update({b: [] for b in baselines})
    grid = np.linspace(0.005, 0.995, 199)
    bs = cfg["training"]["phase1"]["batch_size"]
    fold_info = []
    C = len(CLASSES)

    for fold_i, test_s in enumerate(FOLDS):
        train_s = [s for s in range(12) if s not in test_s]
        val_s = train_s[fold_i % len(train_s)]           # one real validation session
        tr_s = [s for s in train_s if s != val_s]
        m_tr, m_va, m_te = np.isin(sess, tr_s), sess == val_s, np.isin(sess, test_s)
        Xtr, Xte, _ = scale_fold(X_all[m_tr], X_all[m_te], gidx)
        _, Xva, _ = scale_fold(X_all[m_tr], X_all[m_va], gidx)
        ytr, yva, yte = y_all[m_tr], y_all[m_va], y_all[m_te]
        print(f"\n{'═'*70}\n  FOLD {fold_i+1}/4  test sessions {test_s}  val session {val_s}  "
              f"train {m_tr.sum():,} / val {m_va.sum():,} / test {m_te.sum():,}\n{'═'*70}")
        fold_info.append({"fold": fold_i, "test_sessions": test_s, "val_session": int(val_s),
                          "n_train_real": int(m_tr.sum()), "n_val_real": int(m_va.sum()), "n_test_real": int(m_te.sum()),
                          "test_class_counts": {c: int((yte == i).sum()) for i, c in enumerate(CLASSES)}})
        Xs, ys = apply_smote_fold(Xtr, ytr, target_count=args.smote_target, random_state=42 + fold_i)
        fold_info[-1]["n_train_after_smote"] = int(len(ys))
        g_s, g_va, g_te = (evcv.make_hit_groups(X, cfg) for X in (Xs, Xva, Xte))

        # ── HiT-IDS cascade ──
        set_seed(42 + fold_i)
        t0 = time.time()
        bin_m = build(cfg, 2)
        train_hit_model(bin_m, Xs, (ys > 0).astype(np.int64), Xva, (yva > 0).astype(np.int64), g_s, g_va, cfg, DEVICE,
                        max_epochs=50, patience=10, num_classes=2, rare_boost=False)
        phase2_refine(bin_m, Xs, (ys > 0).astype(np.int64), g_s, cfg, DEVICE, num_classes=2, max_epochs=10, patience=5)
        ms, mv = ys > 0, yva > 0
        atk_m = build(cfg, C - 1)
        train_hit_model(atk_m, Xs[ms], ys[ms] - 1, Xva[mv], yva[mv] - 1, {k: v[ms] for k, v in g_s.items()},
                        {k: v[mv] for k, v in g_va.items()}, cfg, DEVICE, max_epochs=100, patience=15,
                        num_classes=C - 1, rare_boost=True)
        phase2_refine(atk_m, Xs[ms], ys[ms] - 1, {k: v[ms] for k, v in g_s.items()}, cfg, DEVICE,
                      num_classes=C - 1, max_epochs=10, patience=5)
        torch.save(bin_m.state_dict(), os.path.join(ckpt_dir, f"fold{fold_i}_binary.pt"))
        torch.save(atk_m.state_dict(), os.path.join(ckpt_dir, f"fold{fold_i}_attack.pt"))
        old = (evcv.NUM_MERGED, evcv.MERGED_NAMES)
        evcv.NUM_MERGED, evcv.MERGED_NAMES = C, CLASSES
        th, yp, pt, yp05 = cascade_cal(bin_m, atk_m, Xva, yva, Xte, yte, g_va, g_te, bs, grid)
        evcv.NUM_MERGED, evcv.MERGED_NAMES = old
        r = fold_metrics_nsm(yte, yp, pt)
        r["stage1_threshold"] = th
        r["test_macro_f1_at_0.5"] = float(f1_score(yte, yp05, average="macro", zero_division=0))
        results["HiT-IDS (Cascade)"].append(r)
        save_fold_predictions(pred_dir, fold_i, "HiT-IDS (Cascade)", yte, yp, pt)
        print(f"  [HiT-IDS] t*={th:.3f} macro-F1 {r['macro_f1']:.4f} (argmax {r['test_macro_f1_at_0.5']:.4f}) "
              f"FPR {r['binary']['fpr']*100:.2f}% [{time.time()-t0:.0f}s]")

        # ── baselines (argmax, then calibrated from their validation probabilities) ──
        for bl in baselines:
            t0 = time.time()
            evcv.NUM_MERGED, evcv.MERGED_NAMES = C, CLASSES
            try:
                r0 = train_eval_baseline_fold(bl, Xs, ys, Xte, yte, Xs.shape[1], C, DEVICE, seed=42 + fold_i,
                                              fold_i=fold_i, pred_dir=pred_dir, X_val=Xva, y_val=yva, ckpt_dir=ckpt_dir)
            finally:
                evcv.NUM_MERGED, evcv.MERGED_NAMES = old
            v = np.load(os.path.join(pred_dir, f"fold{fold_i}_{bl}_val.npz"))
            t = np.load(os.path.join(pred_dir, f"fold{fold_i}_{bl}.npz"))
            pv, ptb = v["proba"].astype(np.float32), t["proba"].astype(np.float32)

            def pred(p, thr):
                out = np.zeros(len(p), dtype=np.int64)
                m = (1 - p[:, 0]) >= thr
                out[m] = p[m, 1:].argmax(1) + 1
                return out
            sc = [f1_score(yva, pred(pv, thr), average="macro", zero_division=0) for thr in grid]
            thr = float(grid[int(np.argmax(sc))])
            rb = fold_metrics_nsm(yte, pred(ptb, thr), ptb)
            rb["stage1_threshold"] = thr
            rb["test_macro_f1_at_argmax"] = r0["macro_f1"]
            results[bl].append(rb)
            save_fold_predictions(pred_dir, fold_i, bl, yte, pred(ptb, thr), ptb)
            print(f"  [{bl}] t*={thr:.3f} macro-F1 {rb['macro_f1']:.4f} (argmax {r0['macro_f1']:.4f}) "
                  f"FPR {rb['binary']['fpr']*100:.2f}% [{time.time()-t0:.0f}s]")

    # ── summary + paired tests ──
    summary, tests = {}, {}
    hit = np.array([r["macro_f1"] for r in results["HiT-IDS (Cascade)"]])
    K = len(FOLDS)
    for m, rows in results.items():
        a = np.array([r["macro_f1"] for r in rows])
        summary[m] = {"mean_macro_f1": float(a.mean()), "std_macro_f1": float(a.std()), "per_fold_f1": a.tolist(),
                      "mean_macro_f1_argmax": float(np.mean([r.get("test_macro_f1_at_0.5", r.get("test_macro_f1_at_argmax")) for r in rows])),
                      "mean_accuracy": float(np.mean([r["accuracy"] for r in rows])),
                      "binary": {k: {"mean": float(np.mean([r["binary"][k] for r in rows])), "std": float(np.std([r["binary"][k] for r in rows]))}
                                 for k in rows[0]["binary"]},
                      "per_class": {c: {"f1_mean": float(np.mean([r["per_class"][c]["f1"] for r in rows])),
                                        "f1_std": float(np.std([r["per_class"][c]["f1"] for r in rows])),
                                        "precision_mean": float(np.mean([r["per_class"][c]["precision"] for r in rows])),
                                        "recall_mean": float(np.mean([r["per_class"][c]["recall"] for r in rows])),
                                        "support_per_fold": [r["per_class"][c]["support"] for r in rows]} for c in CLASSES},
                      "stage1_thresholds": [r["stage1_threshold"] for r in rows],
                      "confusion_matrix_sum": np.sum([np.array(r["confusion_matrix"]) for r in rows], 0).tolist()}
        if m != "HiT-IDS (Cascade)":
            delta = hit - a
            _, p = stats.ttest_rel(hit, a)
            tc = stats.t.ppf(0.975, K - 1)
            tests[m] = {"delta_mean": float(delta.mean()),
                        "delta_ci": [float(delta.mean() - tc * delta.std(ddof=1) / np.sqrt(K)), float(delta.mean() + tc * delta.std(ddof=1) / np.sqrt(K))],
                        "cohens_d": float(cohens_d(hit, a)), "p_value": float(p), "significant": bool(p < 0.05)}
        print(f"  {m:22s} macro-F1 {a.mean():.4f} ± {a.std():.4f}")
    out = {"dataset": "NSM-IDS-2026 v1.2 (Zenodo 23392184)", "protocol": {
               "folds": FOLDS, "validation": "one training session per fold", "dedup": "exact duplicates removed", "split_fragments": "included" if args.include_fragments else "excluded (split_fragment=1)",
               "features": feats, "groups": {g: len(v) for g, v in cfg["feature_groups"].items()},
               "resampling": f"adaptive SMOTE (target {args.smote_target}) + BENIGN cap, training sessions only",
               "threshold": "validation-calibrated for HiT-IDS and every baseline", "smoke": args.smoke},
           "class_names": CLASSES, "k_folds": K, "folds": fold_info, "summary": summary,
           "statistical_tests": tests, "per_fold_results": results}
    with open(os.path.join(tables, f"{tag}_session_cv.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {os.path.join(tables, f'{tag}_session_cv.json')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Zero-shot transfer CIC-IDS2017 → NSM-IDS-2026 on the PORTABLE feature subset.

Portable features = HiT-IDS features ∩ NSM ML table ∩ NSM_portable_features.txt
(the dataset authors exclude Destination Port, Init_Win_bytes_* and
min_seg_size_forward as lab-topology artefacts). A HiT-IDS cascade, a Random
Forest and an XGBoost are trained on CIC-IDS2017 (85% of the real rows, 15%
real validation for early stopping and the Stage-1 threshold) with those
features only, then applied to every NSM flow without any NSM label.

Two normalisations are reported:
    source-scaled : CIC-fitted RobustScaler applied to NSM
    target-scaled : RobustScaler re-fitted on the NSM features (label-free)

Metrics on NSM: attack-vs-benign (F1, detection rate, FPR, ROC-AUC) at the
calibrated threshold; per-NSM-class detection rate; and, for the seven NSM
attack types that exist in CIC-IDS2017, the attack-type recall of Stage 2 among
detected flows. DoS SYN has no CIC-IDS2017 counterpart and is reported as
"detected as some attack" only.

Writes results/tables/transfer_cic_to_nsm.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.evaluate_cv as evcv
from scripts.evaluate_cv import (apply_smote_fold, train_hit_model, phase2_refine, set_seed, make_hit_groups,
                                 eval_cascade_on_test_full, MERGED_NAMES, NUM_MERGED, merge_labels_np)
from scripts.nsm_session_cv import load_nsm, CLASSES as NSM_CLASSES, build
from src.data.cv_data import load_clean_dataset, scale_fold
from src.models.baselines import SklearnBaseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAB = {"Destination Port", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward"}
NSM_TO_CIC = {"PortScan": "PortScan", "SSH-Patator": "SSH-Patator", "FTP-Patator": "FTP-Patator",
              "Web Attack": "Web Attack", "DoS Slowloris": "DoS slowloris", "DoS Slowhttptest": "DoS Slowhttptest",
              "DoS GoldenEye": "DoS GoldenEye", "Infiltration": "Infiltration"}


def binary_report(y_bin, score, thr):
    pred = (score >= thr).astype(int)
    return {"threshold": float(thr), "f1": float(f1_score(y_bin, pred)), "precision": float(precision_score(y_bin, pred, zero_division=0)),
            "detection_rate": float(recall_score(y_bin, pred)), "fpr": float(((y_bin == 0) & (pred == 1)).sum() / max((y_bin == 0).sum(), 1)),
            "roc_auc": float(roc_auc_score(y_bin, score)), "macro_f1_binary": float(f1_score(y_bin, pred, average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--csv", default="data/nsm/NSM-IDS-2026_ML_v1.2.csv")
    ap.add_argument("--portable", default="data/nsm/NSM-IDS-2026_portable_features.txt")
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
    set_seed(42)

    nsm, y_nsm = load_nsm(args.csv)
    portable = {l.strip() for l in open(args.portable) if l.strip()}
    X_all, y15, cols, _ = load_clean_dataset(args.config)
    y_all = merge_labels_np(y15)
    cfg = json.loads(json.dumps(config))
    cfg["feature_groups"] = {g: [f for f in fs if f in nsm.columns and f in portable and f not in LAB]
                             for g, fs in config["feature_groups"].items()}
    feats = [f for fs in cfg["feature_groups"].values() for f in fs]
    col_idx = [cols.index(f) for f in feats]
    gidx, off = {}, 0
    for g, fs in cfg["feature_groups"].items():
        gidx[g] = list(range(off, off + len(fs)))
        off += len(fs)
    print(f"  portable features: {len(feats)} {dict((g, len(v)) for g, v in cfg['feature_groups'].items())}")

    Xc = X_all[:, col_idx]
    if args.smoke:
        from src.data.cv_data import smoke_subsample
        Xc, y_all = smoke_subsample(Xc, y_all, frac=0.03)
    tr_i, va_i = train_test_split(np.arange(len(y_all)), test_size=0.15, stratify=y_all, random_state=42)
    Xtr, Xva, scalers = scale_fold(Xc[tr_i], Xc[va_i], gidx)
    ytr, yva = y_all[tr_i], y_all[va_i]
    Xs, ys = apply_smote_fold(Xtr, ytr, target_count=20000, random_state=42)
    g_s, g_va = make_hit_groups(Xs, cfg), make_hit_groups(Xva, cfg)
    print(f"  CIC train {len(ytr):,} → resampled {len(ys):,}; val {len(yva):,}")

    # ── HiT-IDS cascade on CIC ──
    bin_m = build(cfg, 2)
    train_hit_model(bin_m, Xs, (ys > 0).astype(np.int64), Xva, (yva > 0).astype(np.int64), g_s, g_va, cfg, DEVICE,
                    max_epochs=50, patience=10, num_classes=2, rare_boost=False)
    phase2_refine(bin_m, Xs, (ys > 0).astype(np.int64), g_s, cfg, DEVICE, num_classes=2, max_epochs=10, patience=5)
    ms, mv = ys > 0, yva > 0
    atk_m = build(cfg, NUM_MERGED - 1)
    train_hit_model(atk_m, Xs[ms], ys[ms] - 1, Xva[mv], yva[mv] - 1, {k: v[ms] for k, v in g_s.items()},
                    {k: v[mv] for k, v in g_va.items()}, cfg, DEVICE, max_epochs=100, patience=15,
                    num_classes=NUM_MERGED - 1, rare_boost=True)
    phase2_refine(atk_m, Xs[ms], ys[ms] - 1, {k: v[ms] for k, v in g_s.items()}, cfg, DEVICE,
                  num_classes=NUM_MERGED - 1, max_epochs=10, patience=5)
    cd = os.path.join(config["paths"]["checkpoints"], "transfer_portable")
    os.makedirs(cd, exist_ok=True)
    torch.save({"model_state_dict": bin_m.state_dict(), "features": feats}, os.path.join(cd, "hit_ids_binary_portable.pt"))
    torch.save({"model_state_dict": atk_m.state_dict(), "features": feats}, os.path.join(cd, "hit_ids_attack_portable.pt"))

    # CIC validation: in-domain reference + threshold on p(attack)
    _, _, pva = eval_cascade_on_test_full(bin_m, atk_m, Xva, yva, g_va, 4096, DEVICE)
    grid = np.linspace(0.005, 0.995, 199)
    yvb = (yva > 0).astype(int)
    thr = float(grid[int(np.argmax([f1_score(yvb, ((1 - pva[:, 0]) >= t).astype(int)) for t in grid]))])
    in_domain = {"binary": binary_report(yvb, 1 - pva[:, 0], thr)}
    pred13 = np.where((1 - pva[:, 0]) >= thr, pva[:, 1:].argmax(1) + 1, 0)
    in_domain["macro_f1_13"] = float(f1_score(yva, pred13, average="macro", zero_division=0))

    rf, xg = SklearnBaseline("random_forest"), SklearnBaseline("xgboost")
    rf.fit(Xs, ys)
    xg.fit(Xs, ys)
    bl_thr = {}
    for name, m in (("random_forest", rf), ("xgboost", xg)):
        s = 1 - evcv._sk_proba(m, Xva, NUM_MERGED)[:, 0]
        bl_thr[name] = float(grid[int(np.argmax([f1_score(yvb, (s >= t).astype(int)) for t in grid]))])
        in_domain[name] = binary_report(yvb, s, bl_thr[name])

    # ── NSM evaluation ──
    Xn_raw = nsm[feats].values.astype(np.float32)
    y_bin = (y_nsm > 0).astype(int)
    Xn_src = Xn_raw.copy()
    for g, idx in gidx.items():
        Xn_src[:, idx] = scalers[g].transform(Xn_raw[:, idx])
    Xn_tgt, _, _ = scale_fold(Xn_raw, Xn_raw[:1], gidx)
    results = {"portable_features": feats, "n_nsm_flows": int(len(y_nsm)), "cic_in_domain_validation": in_domain,
               "stage1_threshold_from_cic_validation": thr, "normalisation": {}}
    cic_name_to_idx = {n: i for i, n in enumerate(MERGED_NAMES)}
    for norm_name, Xn in (("source_scaled", Xn_src), ("target_scaled", Xn_tgt)):
        g_n = make_hit_groups(Xn, cfg)
        _, _, pn = eval_cascade_on_test_full(bin_m, atk_m, Xn, y_bin, g_n, 4096, DEVICE)
        score = 1 - pn[:, 0]
        rep = {"HiT-IDS": {"at_cic_threshold": binary_report(y_bin, score, thr), "at_0.5": binary_report(y_bin, score, 0.5)}}
        det = score >= thr
        per_class = {}
        for i, c in enumerate(NSM_CLASSES):
            m = y_nsm == i
            entry = {"n": int(m.sum()), "detection_rate" if i else "false_positive_rate": float(det[m].mean())}
            if c in NSM_TO_CIC and det[m].sum() > 0:
                typ = pn[m & det, 1:].argmax(1) + 1
                entry["type_recall_given_detected"] = float((typ == cic_name_to_idx[NSM_TO_CIC[c]]).mean())
                top = np.bincount(typ, minlength=NUM_MERGED).argmax()
                entry["most_frequent_cic_label"] = MERGED_NAMES[int(top)]
            elif det[m].sum() > 0:
                typ = pn[m & det, 1:].argmax(1) + 1
                entry["most_frequent_cic_label"] = MERGED_NAMES[int(np.bincount(typ, minlength=NUM_MERGED).argmax())]
            per_class[c] = entry
        rep["HiT-IDS"]["per_nsm_class"] = per_class
        for name, m in (("random_forest", rf), ("xgboost", xg)):
            s = 1 - evcv._sk_proba(m, Xn, NUM_MERGED)[:, 0]
            rep[name] = {"at_cic_threshold": binary_report(y_bin, s, bl_thr[name]), "at_0.5": binary_report(y_bin, s, 0.5)}
        results["normalisation"][norm_name] = rep
        print(f"  [{norm_name}] HiT-IDS binary F1 {rep['HiT-IDS']['at_cic_threshold']['f1']:.3f} "
              f"DR {rep['HiT-IDS']['at_cic_threshold']['detection_rate']:.3f} FPR {rep['HiT-IDS']['at_cic_threshold']['fpr']*100:.2f}% "
              f"AUC {rep['HiT-IDS']['at_cic_threshold']['roc_auc']:.3f} | RF F1 {rep['random_forest']['at_cic_threshold']['f1']:.3f} "
              f"| XGB F1 {rep['xgboost']['at_cic_threshold']['f1']:.3f}")
    out = os.path.join(config["paths"]["tables"], "transfer_cic_to_nsm" + ("_smoke" if args.smoke else "") + ".json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

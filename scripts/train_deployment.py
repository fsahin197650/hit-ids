#!/usr/bin/env python3
"""
Train the DEPLOYMENT models on the 70/15/15 split with exactly the protocol used
inside each CV fold (scripts/evaluate_cv.py):

    train.pt (real, scaled with train-fit scaler)  → adaptive SMOTE + BENIGN cap
    val.pt   (real)                                 → early stopping only
    test.pt  (real)                                 → reported by deployment_eval.py

Saves (results/checkpoints/):
    hit_ids_binary_phase1.pt, hit_ids_binary.pt              (C = 2)
    hit_ids_attack_merged_phase1.pt, hit_ids_attack_merged.pt (C = 12, merged labels)
    baseline_rf.pkl, baseline_xgb.pkl, baseline_bilstm.pt, baseline_vt.pt
The *_phase1 checkpoints are the models BEFORE the attention-entropy fine-tuning,
kept for the Phase-2 faithfulness comparison.

Usage: python scripts/train_deployment.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate_cv import (apply_smote_fold, train_hit_model, phase2_refine, make_hit_groups,
                                 fold_metrics, set_seed, merge_labels_np, NUM_MERGED, MERGED_NAMES)
from src.models.hit_ids import HiTIDS
from src.models.baselines import SklearnBaseline, create_baseline, train_pytorch_baseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build(config, nc):
    fg = config["feature_groups"]
    names = list(fg.keys())
    mc = config["model"]
    return HiTIDS(group_sizes=[len(fg[g]) for g in names], group_names=names, d_emb=mc["d_emb"],
                  nhead=mc["nhead"], intra_layers=mc["intra_group_layers"],
                  inter_layers=mc["inter_group_layers"], dim_feedforward=mc["dim_feedforward"],
                  dropout=mc["dropout"], num_classes=nc).to(DEVICE)


def save_ckpt(model, path, nc, phase):
    torch.save({"model_state_dict": model.state_dict(), "num_classes": nc, "phase": phase}, path)
    print(f"    saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--smote-target", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 3%% subsample, separate checkpoint dir")
    ap.add_argument("--no-phase2", action="store_true")
    args = ap.parse_args()
    if args.no_phase2:
        import scripts.evaluate_cv as _evcv
        _evcv.PHASE2 = False
    if args.smoke:
        import scripts.evaluate_cv as evcv
        evcv.EPOCH_CAP = 1
    with open(args.config) as f:
        config = yaml.safe_load(f)
    set_seed(args.seed)
    sd, cd = config["paths"]["splits"], config["paths"]["checkpoints"]
    if args.smoke:
        cd = cd + "_smoke"
    os.makedirs(cd, exist_ok=True)

    tr = torch.load(f"{sd}/train.pt", weights_only=False)
    va = torch.load(f"{sd}/val.pt", weights_only=False)
    te = torch.load(f"{sd}/test.pt", weights_only=False)
    Xtr, ytr = tr["X"].numpy(), merge_labels_np(tr["y"].numpy())
    Xva, yva = va["X"].numpy(), merge_labels_np(va["y"].numpy())
    Xte, yte = te["X"].numpy(), merge_labels_np(te["y"].numpy())
    assert set(np.unique(ytr)) <= set(range(NUM_MERGED))
    if args.smoke:
        from src.data.cv_data import smoke_subsample
        Xtr, ytr = smoke_subsample(Xtr, ytr, frac=0.02)
        Xva, yva = smoke_subsample(Xva, yva, frac=0.02)
        Xte, yte = smoke_subsample(Xte, yte, frac=0.02)
    print(f"  train {len(ytr):,} real rows | val {len(yva):,} | test {len(yte):,} | features {Xtr.shape[1]}")

    t0 = time.time()
    Xs, ys = apply_smote_fold(Xtr, ytr, target_count=args.smote_target, random_state=args.seed)
    print(f"  SMOTE: {len(ytr):,} → {len(ys):,} ({time.time()-t0:.0f}s)")
    g_s, g_va = make_hit_groups(Xs, config), make_hit_groups(Xva, config)

    # ── binary stage ──
    print("\n  [binary stage]")
    bin_m = build(config, 2)
    f1 = train_hit_model(bin_m, Xs, (ys > 0).astype(np.int64), Xva, (yva > 0).astype(np.int64),
                         g_s, g_va, config, DEVICE, max_epochs=50, patience=10, num_classes=2, rare_boost=False)
    print(f"    phase-1 val binary F1 {f1:.4f}")
    save_ckpt(bin_m, f"{cd}/hit_ids_binary_phase1.pt", 2, "phase1")
    phase2_refine(bin_m, Xs, (ys > 0).astype(np.int64), g_s, config, DEVICE, num_classes=2, max_epochs=10, patience=5)
    save_ckpt(bin_m, f"{cd}/hit_ids_binary.pt", 2, "phase2")

    # ── attack-type stage (12 merged classes) ──
    print("\n  [attack stage]")
    m_s, m_va = ys > 0, yva > 0
    atk_m = build(config, NUM_MERGED - 1)
    f1 = train_hit_model(atk_m, Xs[m_s], ys[m_s] - 1, Xva[m_va], yva[m_va] - 1,
                         {k: v[m_s] for k, v in g_s.items()}, {k: v[m_va] for k, v in g_va.items()},
                         config, DEVICE, max_epochs=100, patience=15, num_classes=NUM_MERGED - 1, rare_boost=True)
    print(f"    phase-1 val attack macro-F1 {f1:.4f}")
    save_ckpt(atk_m, f"{cd}/hit_ids_attack_merged_phase1.pt", NUM_MERGED - 1, "phase1")
    phase2_refine(atk_m, Xs[m_s], ys[m_s] - 1, {k: v[m_s] for k, v in g_s.items()}, config, DEVICE,
                  num_classes=NUM_MERGED - 1, max_epochs=10, patience=5)
    save_ckpt(atk_m, f"{cd}/hit_ids_attack_merged.pt", NUM_MERGED - 1, "phase2")

    # ── baselines on the same resampled rows ──
    summary = {}
    for name in ("random_forest", "xgboost"):
        print(f"\n  [{name}]")
        b = SklearnBaseline(model_type=name)
        b.fit(Xs, ys)
        with open(f"{cd}/baseline_{'rf' if name == 'random_forest' else 'xgb'}.pkl", "wb") as f:
            pickle.dump(b.model, f)
        summary[name] = fold_metrics(yte, b.predict(Xte))["macro_f1"]
        print(f"    test macro-F1 (argmax) {summary[name]:.4f}")
    nf = Xs.shape[1]
    for name, fn, extra in (("bilstm", "baseline_bilstm.pt", {}),
                            ("vanilla_transformer", "baseline_vt.pt",
                             {"d_emb": 64, "nhead": 4, "num_layers": 4, "dim_feedforward": 256})):
        print(f"\n  [{name}]")
        set_seed(args.seed)
        model = create_baseline(name, nf, NUM_MERGED, **extra)
        tl = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(ys)), batch_size=256, shuffle=True)
        vl = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=1024)
        train_pytorch_baseline(model, tl, vl, num_epochs=1 if args.smoke else 30, lr=1e-3, device=DEVICE)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xte), 4096):
                preds.append(model(torch.tensor(Xte[i:i+4096]).to(DEVICE)).argmax(1).cpu())
        summary[name] = fold_metrics(yte, torch.cat(preds).numpy())["macro_f1"]
        torch.save({"model_state_dict": model.state_dict(), "num_features": nf, "num_classes": NUM_MERGED, **extra},
                   f"{cd}/{fn}")
        print(f"    test macro-F1 (argmax) {summary[name]:.4f}")

    with open(f"{cd}/deployment_training_summary.json", "w") as f:
        json.dump({"n_train_real": int(len(ytr)), "n_train_resampled": int(len(ys)),
                   "resampled_counts": {MERGED_NAMES[c]: int((ys == c).sum()) for c in range(NUM_MERGED)},
                   "baseline_test_macro_f1_argmax": summary}, f, indent=2)
    print("\n  deployment training done")


if __name__ == "__main__":
    main()

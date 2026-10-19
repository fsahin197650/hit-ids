#!/usr/bin/env python3
"""
Train all baseline models and save checkpoints.

Models: RF, XGBoost, BiLSTM, Vanilla Transformer
All trained on the same train split with merged labels (13 classes).

Usage:
    python scripts/train_baselines.py --config configs/config.yaml
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
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baselines import (
    SklearnBaseline,
    BiLSTMClassifier,
    VanillaTransformer,
    train_pytorch_baseline,
)

# ── Label Merging ────────────────────────────────────────────────────────
MERGE_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
    12: 12, 13: 12, 14: 12,
}
MERGED_NAMES = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS slowloris", "FTP-Patator",
    "Heartbleed", "Infiltration", "PortScan", "SSH-Patator",
    "Web Attack",
]
NUM_MERGED = 13


def merge_labels(y):
    if isinstance(y, np.ndarray):
        return np.array([MERGE_MAP[int(v)] for v in y])
    return np.array([MERGE_MAP[int(v)] for v in y.numpy()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    splits_dir = config["paths"]["splits"]
    ckpt_dir = config["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 70)
    print("  BASELINE MODEL TRAINING")
    print("=" * 70)

    # ── Load Data ──────────────────────────────────────────────────────
    print("\n  Loading data...")
    train_data = torch.load(f"{splits_dir}/train.pt", weights_only=False)
    test_data = torch.load(f"{splits_dir}/test.pt", weights_only=False)

    X_train = train_data["X"].numpy()
    y_train = merge_labels(train_data["y"])
    X_test = test_data["X"].numpy()
    y_test = merge_labels(test_data["y"])

    num_features = X_train.shape[1]
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Features: {num_features}  |  Classes: {NUM_MERGED}")

    results = {}

    # ══════════════════════════════════════════════════════════════════
    # 1. RANDOM FOREST
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  1. RANDOM FOREST")
    print("─" * 70)

    rf = SklearnBaseline("random_forest", n_estimators=200)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_preds, average="macro", zero_division=0)
    print(f"  Macro-F1: {rf_f1:.4f}")
    print(classification_report(y_test, rf_preds, target_names=MERGED_NAMES,
                                zero_division=0, labels=list(range(NUM_MERGED))))

    # Save
    rf_path = os.path.join(ckpt_dir, "baseline_rf.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(rf.model, f)
    print(f"  Saved: {rf_path}")
    results["Random Forest"] = rf_f1

    # ══════════════════════════════════════════════════════════════════
    # 2. XGBOOST
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  2. XGBOOST")
    print("─" * 70)

    xgb = SklearnBaseline("xgboost", n_estimators=200, max_depth=8)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_preds, average="macro", zero_division=0)
    print(f"  Macro-F1: {xgb_f1:.4f}")
    print(classification_report(y_test, xgb_preds, target_names=MERGED_NAMES,
                                zero_division=0, labels=list(range(NUM_MERGED))))

    # Save
    xgb_path = os.path.join(ckpt_dir, "baseline_xgb.pkl")
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb.model, f)
    print(f"  Saved: {xgb_path}")
    results["XGBoost"] = xgb_f1

    # ══════════════════════════════════════════════════════════════════
    # 3. BiLSTM
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3. BiLSTM")
    print("─" * 70)

    bilstm = BiLSTMClassifier(num_features=num_features, num_classes=NUM_MERGED)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test[:20000], dtype=torch.float32),
                            torch.tensor(y_test[:20000], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    result = train_pytorch_baseline(bilstm, train_loader, val_loader,
                                    num_epochs=30, lr=1e-3, device=device)
    bilstm.to(device).eval()

    # Full test F1
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            batch = torch.tensor(X_test[i:i+512], dtype=torch.float32).to(device)
            preds = bilstm(batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
    bilstm_f1 = f1_score(y_test, np.array(all_preds), average="macro", zero_division=0)
    print(f"  Macro-F1: {bilstm_f1:.4f}")
    print(classification_report(y_test, np.array(all_preds), target_names=MERGED_NAMES,
                                zero_division=0, labels=list(range(NUM_MERGED))))

    # Save
    bilstm_path = os.path.join(ckpt_dir, "baseline_bilstm.pt")
    torch.save({"model_state_dict": bilstm.state_dict(),
                "num_features": num_features, "num_classes": NUM_MERGED}, bilstm_path)
    print(f"  Saved: {bilstm_path}")
    results["BiLSTM"] = bilstm_f1

    # ══════════════════════════════════════════════════════════════════
    # 4. VANILLA TRANSFORMER
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  4. VANILLA TRANSFORMER")
    print("─" * 70)

    vt = VanillaTransformer(num_features=num_features, num_classes=NUM_MERGED,
                            d_emb=64, nhead=4, num_layers=4, dim_feedforward=256)
    result = train_pytorch_baseline(vt, train_loader, val_loader,
                                    num_epochs=30, lr=1e-3, device=device)
    vt.to(device).eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            batch = torch.tensor(X_test[i:i+512], dtype=torch.float32).to(device)
            preds = vt(batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
    vt_f1 = f1_score(y_test, np.array(all_preds), average="macro", zero_division=0)
    print(f"  Macro-F1: {vt_f1:.4f}")
    print(classification_report(y_test, np.array(all_preds), target_names=MERGED_NAMES,
                                zero_division=0, labels=list(range(NUM_MERGED))))

    # Save
    vt_path = os.path.join(ckpt_dir, "baseline_vt.pt")
    torch.save({"model_state_dict": vt.state_dict(),
                "num_features": num_features, "num_classes": NUM_MERGED,
                "d_emb": 64, "nhead": 4, "num_layers": 4, "dim_feedforward": 256},
               vt_path)
    print(f"  Saved: {vt_path}")
    results["Vanilla Transformer"] = vt_f1

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE — Summary")
    print("=" * 70)
    for name, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:25s}: F1 = {f1:.4f}")

    print(f"\n  Checkpoints saved to: {ckpt_dir}/")
    print("    baseline_rf.pkl")
    print("    baseline_xgb.pkl")
    print("    baseline_bilstm.pt")
    print("    baseline_vt.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()

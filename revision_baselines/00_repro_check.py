#!/usr/bin/env python3
"""
00_repro_check.py  —  Sanity gate: reproduce BiLSTM 5-fold macro-F1.

Run this FIRST. If the reported macro-F1 is far from 0.9966 ± 0.0003,
the pipeline wiring is broken and no new baseline numbers can be trusted.

Expected:  BiLSTM macro-F1 ≈ 0.9966 ± 0.0003

Usage:
    python revision_baselines/00_repro_check.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Pipeline hooks ────────────────────────────────────────────────────────
from pipeline_hooks import (
    MERGED_NAMES,
    NUM_FEATURES,
    NUM_MERGED,
    apply_smote_fold,
    compute_fold_metrics,
    get_fold_splitter,
    load_all_data,
    set_seed,
)
from pipeline_hooks import BiLSTMClassifier, train_pytorch_baseline


def main():
    print("=" * 70)
    print("  REPRO CHECK — BiLSTM 5-fold macro-F1")
    print("  EXPECT ≈ 0.9966 ± 0.0003")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load all data
    print("\n  Loading data...")
    X_all, y_all = load_all_data()
    print(f"  Total samples: {len(y_all):,}  |  Features: {X_all.shape[1]}")

    skf = get_fold_splitter(n_splits=5)
    fold_f1s = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n{'─' * 70}")
        print(f"  FOLD {fold_i + 1}/5")
        t0 = time.time()

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # SMOTE (same random_state as evaluate_cv.py)
        X_train_s, y_train_s = apply_smote_fold(
            X_train, y_train,
            target_count=20_000,
            random_state=42 + fold_i,
        )
        print(f"  SMOTE: {len(X_train):,} → {len(X_train_s):,}")

        # Create BiLSTM (same architecture as existing baseline)
        set_seed(42 + fold_i)
        model = BiLSTMClassifier(
            num_features=NUM_FEATURES,
            num_classes=NUM_MERGED,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
        )

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train_s, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_test[:20_000], dtype=torch.float32),
            torch.tensor(y_test[:20_000], dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        # Train
        train_pytorch_baseline(
            model, train_loader, val_loader,
            num_epochs=30, lr=1e-3, device=device,
        )

        # Evaluate on full test fold
        model.to(device).eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                batch = torch.tensor(
                    X_test[i:i + 512], dtype=torch.float32
                ).to(device)
                preds = model(batch).argmax(1).cpu().numpy()
                all_preds.extend(preds)
        y_pred = np.array(all_preds)

        metrics = compute_fold_metrics(y_test, y_pred)
        fold_f1s.append(metrics["macro_f1"])
        print(f"  Fold {fold_i + 1} macro-F1: {metrics['macro_f1']:.4f}  "
              f"[{time.time() - t0:.0f}s]")

    # ── Summary ──────────────────────────────────────────────────────────
    arr = np.array(fold_f1s)
    print("\n" + "=" * 70)
    print("  REPRO CHECK RESULTS")
    print("=" * 70)
    print(f"  Per-fold F1: {[f'{f:.4f}' for f in fold_f1s]}")
    print(f"  Mean ± Std:  {arr.mean():.4f} ± {arr.std():.4f}")
    print()

    # Pass/fail
    expected_mean = 0.9966
    tolerance = 0.005  # ±0.5% is acceptable given stochasticity
    if abs(arr.mean() - expected_mean) < tolerance:
        print("  ✅ PASS — BiLSTM macro-F1 is within expected range.")
        print(f"       Expected ≈ {expected_mean}, got {arr.mean():.4f}")
        print("       Pipeline wiring is correct. Proceed with new baselines.")
    else:
        print("  ❌ FAIL — BiLSTM macro-F1 is NOT within expected range!")
        print(f"       Expected ≈ {expected_mean} ± {tolerance}")
        print(f"       Got:       {arr.mean():.4f} ± {arr.std():.4f}")
        print()
        print("  BEFORE trusting any new numbers, fix the pipeline:")
        print("    1. Check SPLITS_DIR in pipeline_hooks.py")
        print("    2. Check MERGE_MAP matches evaluate_cv.py")
        print("    3. Check apply_smote_fold() parameters")
        print("    4. Check BiLSTM architecture matches baselines.py")
        sys.exit(1)


if __name__ == "__main__":
    main()

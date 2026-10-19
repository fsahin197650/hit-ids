#!/usr/bin/env python3
"""
01_deepids.py  —  Deep-IDS (Racha et al., 2024) LSTM baseline.

Deep-IDS is an LSTM-based IDS (not a transformer).
Architecture: LSTM(64 units) + dense classification head.
Reported macro-F1 ≈ 0.9791 on CIC-IDS2017.

We replicate it as a 13-class baseline under the Table 4 protocol
(same folds, SMOTE, evaluation) so differences reflect the MODEL,
not the harness.

Writes:  revision_results/results_deepids.json

Usage:
    python revision_baselines/01_deepids.py
    python revision_baselines/01_deepids.py --epochs 50 --lr 5e-4
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

# ── Pipeline hooks ────────────────────────────────────────────────────────
from pipeline_hooks import (
    MERGED_NAMES,
    NUM_FEATURES,
    NUM_MERGED,
    apply_smote_fold,
    aggregate_folds,
    compute_full_metrics,
    get_fold_splitter,
    load_all_data,
    scale_fold,
    save_json,
    set_seed,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — user can tweak these
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    # Model
    "lstm_units": 64,           # Deep-IDS paper: 64
    "lstm_layers": 1,           # single-layer LSTM (paper default)
    "dropout": 0.3,
    # Training — mirrors BiLSTM budget for fair comparison
    "epochs": 30,
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    # Fold
    "n_folds": 5,
    "smote_target": 20_000,
    "seed": 42,
}


# ═══════════════════════════════════════════════════════════════════════════
# Model: Deep-IDS LSTM
# ═══════════════════════════════════════════════════════════════════════════


class DeepIDSModel(nn.Module):
    """
    Deep-IDS (Racha et al., 2024) — LSTM(64) + dense head.

    Input:  (batch, 72)  — flat features, each treated as a time step.
    Output: (batch, num_classes) logits.

    The original Deep-IDS treats each feature as a 1-D time step and feeds
    them sequentially into an LSTM. We mirror that convention.
    """

    def __init__(
        self,
        num_features: int = 72,
        num_classes: int = 13,
        lstm_units: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_features = num_features

        self.lstm = nn.LSTM(
            input_size=1,               # each feature = 1-D
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_units, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features) flat features
        Returns:
            logits: (batch, num_classes)
        """
        # (batch, 72) → (batch, 72, 1)
        x = x.unsqueeze(-1)
        # LSTM → take last hidden state
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, lstm_units) → take last layer
        last_hidden = h_n[-1]  # (batch, lstm_units)
        return self.classifier(last_hidden)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop (mirrors train_pytorch_baseline from baselines.py)
# ═══════════════════════════════════════════════════════════════════════════


def train_deepids(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
) -> dict:
    """Train Deep-IDS with same budget as BiLSTM baseline."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_targets.extend(y_batch.tolist())

        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}: "
                  f"loss={total_loss/len(train_loader):.4f}, val_f1={val_f1:.4f}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    return {"best_val_f1": best_f1}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Deep-IDS (LSTM) baseline")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lstm-units", type=int, default=DEFAULT_CONFIG["lstm_units"])
    parser.add_argument("--lstm-layers", type=int, default=DEFAULT_CONFIG["lstm_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--smote-target", type=int, default=DEFAULT_CONFIG["smote_target"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()

    config = {
        "lstm_units": args.lstm_units,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": DEFAULT_CONFIG["weight_decay"],
        "smote_target": args.smote_target,
        "seed": args.seed,
        "n_folds": 5,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  DEEP-IDS (LSTM) BASELINE — 5-Fold CV")
    print("=" * 70)
    print(f"  Config: {config}")
    print(f"  Device: {device}")

    # Load data
    print("\n  Loading data...")
    X_all, y_all = load_all_data()
    print(f"  Total: {len(y_all):,} samples, {X_all.shape[1]} features")

    skf = get_fold_splitter(n_splits=config["n_folds"])
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n{'═' * 70}")
        print(f"  FOLD {fold_i + 1}/{config['n_folds']}")
        t0 = time.time()

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # Scale (fit on training fold only) — leak-free
        X_train, X_test = scale_fold(X_train, X_test)

        # Early-stopping validation: 10% of the REAL training-fold rows,
        # carved out BEFORE resampling (same as evaluate_cv.py --val-source real).
        from sklearn.model_selection import train_test_split
        _tr, _va = train_test_split(
            np.arange(len(y_train)), test_size=0.10, stratify=y_train,
            random_state=config["seed"] + fold_i)
        X_val_s, y_val_s = X_train[_va], y_train[_va]
        X_train, y_train = X_train[_tr], y_train[_tr]

        # SMOTE
        X_train_s, y_train_s = apply_smote_fold(
            X_train, y_train,
            target_count=config["smote_target"],
            random_state=config["seed"] + fold_i,
        )
        print(f"  SMOTE: {len(X_train):,} → {len(X_train_s):,}  (real val: {len(y_val_s):,})")

        # Create model
        set_seed(config["seed"] + fold_i)
        model = DeepIDSModel(
            num_features=NUM_FEATURES,
            num_classes=NUM_MERGED,
            lstm_units=config["lstm_units"],
            lstm_layers=config["lstm_layers"],
            dropout=config["dropout"],
        )
        n_params = sum(p.numel() for p in model.parameters())
        if fold_i == 0:
            print(f"  Model parameters: {n_params:,}")

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train_s, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_s, dtype=torch.float32),
            torch.tensor(y_val_s, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4,
        )

        # Train
        train_deepids(
            model, train_loader, val_loader,
            num_epochs=config["epochs"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device=device,
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

        # Full metrics
        metrics = compute_full_metrics(y_test, y_pred, MERGED_NAMES)
        fold_results.append(metrics)

        elapsed = time.time() - t0
        print(f"  Fold {fold_i + 1}: macro-F1 = {metrics['macro_f1']:.4f}  "
              f"[{elapsed:.0f}s]")

    # ── Aggregate & Save ─────────────────────────────────────────────────
    summary = aggregate_folds(fold_results)
    output = {
        "model": "Deep-IDS (LSTM)",
        "reference": "Racha et al., 2024",
        "config": config,
        "n_params": n_params,
        "num_classes": NUM_MERGED,
        "class_names": MERGED_NAMES,
        **summary,
    }

    output["protocol"] = "clean: real rows only, RobustScaler + SMOTE inside training fold, val split from training fold"
    save_json(output, f"results_deepids_{__import__('pipeline_hooks').CV_TAG}.json")

    # Print summary
    print("\n" + "=" * 70)
    print("  DEEP-IDS RESULTS")
    print("=" * 70)
    per_fold = [r["macro_f1"] for r in fold_results]
    arr = np.array(per_fold)
    print(f"  Per-fold F1: {[f'{f:.4f}' for f in per_fold]}")
    print(f"  Mean ± Std:  {arr.mean():.4f} ± {arr.std():.4f}")
    print(f"  Accuracy:    {summary['mean_accuracy']:.4f}")
    print(f"  Weighted-F1: {summary['mean_weighted_f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

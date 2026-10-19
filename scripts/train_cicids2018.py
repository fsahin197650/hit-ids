#!/usr/bin/env python3
"""
HiT-IDS In-Domain Training on CIC-IDS2018.

End-to-end pipeline:
  1. Load preprocessed CIC-IDS2018 splits (from preprocess_cicids2018.py)
  2. Train HiT-IDS cascade (binary + multi-class) using SAME hyperparams as CIC-IDS2017
  3. Evaluate and save results

Prerequisites:
    python scripts/preprocess_cicids2018.py --mode train

Usage:
    python scripts/train_cicids2018.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hit_ids import HiTIDS

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_PATH = "configs/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

LABEL_MAP = {"Benign": 0, "Bot": 1, "Infilteration": 2}
CLASS_NAMES = list(LABEL_MAP.keys())
NUM_CLASSES = len(LABEL_MAP)

# Use CIC-IDS2017 model architecture (same group structure)
fg_cfg = config["feature_groups"]
GROUP_NAMES = list(fg_cfg.keys())
GROUP_SIZES = [len(fg_cfg[n]) for n in GROUP_NAMES]

# Directories
SPLITS_DIR = "data/cicids2018/splits"
CKPT_DIR = "results/cicids2018/checkpoints"
TABLES_DIR = "results/cicids2018/tables"
FIGURES_DIR = "results/cicids2018/figures"

for d in [CKPT_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=4.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model, train_X, train_y, val_X, val_y, num_classes,
                epochs=150, batch_size=256, lr=0.001, patience=30,
                warmup_epochs=5, tag=""):
    """Train a HiT-IDS model with LR warmup and weight capping."""
    # Class weights (inverse frequency, capped)
    y_np = train_y.numpy() if isinstance(train_y, torch.Tensor) else train_y
    counts = np.bincount(y_np, minlength=num_classes)
    total = counts.sum()
    weights = torch.tensor(
        [total / (num_classes * max(c, 1)) for c in counts],
        dtype=torch.float32
    ).to(DEVICE)
    weights = torch.clamp(weights / weights.mean(), min=0.1, max=10.0)
    print(f"    Class weights: {[f'{w:.2f}' for w in weights.cpu().tolist()]}")

    criterion = FocalLoss(gamma=4.5, weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1 = 0
    best_state = None
    no_improve = 0

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)
    val_ds = torch.utils.data.TensorDataset(val_X, val_y)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch, return_explanations=False)
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
                X_batch = X_batch.to(DEVICE)
                logits = model(X_batch, return_explanations=False)
                preds = logits.argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_targets.extend(y_batch.tolist())

        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    [{tag}] Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={avg_loss:.4f}, val_f1={val_f1:.4f}, lr={current_lr:.6f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    [{tag}] Early stop at epoch {epoch+1}, best_f1={best_f1:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    return best_f1


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  HiT-IDS In-Domain Training: CIC-IDS2018")
    print("  (Same architecture as CIC-IDS2017 — no re-tuning)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Groups: {dict(zip(GROUP_NAMES, GROUP_SIZES))}")
    print(f"  Classes: {NUM_CLASSES} → {CLASS_NAMES}")
    start_time = time.time()

    # ── Load splits ──────────────────────────────────────────────────
    if not os.path.exists(os.path.join(SPLITS_DIR, "train.pt")):
        print("\n  [ERROR] Splits not found! Run preprocessing first:")
        print("    python scripts/preprocess_cicids2018.py --mode train")
        sys.exit(1)

    train_data = torch.load(os.path.join(SPLITS_DIR, "train.pt"), weights_only=False)
    test_data = torch.load(os.path.join(SPLITS_DIR, "test.pt"), weights_only=False)

    X_full = train_data["X"]
    y_full_bin = train_data["y_bin"]
    y_full_multi = train_data["y"]
    X_test = test_data["X"]
    y_test_bin = test_data["y_bin"]
    y_test_multi = test_data["y"]

    # ── Train/Val split (85/15 from training data) ─────────────────────
    idx_all = np.arange(len(X_full))
    idx_train, idx_val = train_test_split(
        idx_all, test_size=0.15, stratify=y_full_multi.numpy(), random_state=SEED
    )
    X_train = X_full[idx_train]
    y_train_bin = y_full_bin[idx_train]
    y_train_multi = y_full_multi[idx_train]
    X_val = X_full[idx_val]
    y_val_bin = y_full_bin[idx_val]
    y_val_multi = y_full_multi[idx_val]

    print(f"\n  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    print(f"  Feature dim: {X_train.shape[1]}")

    # ── SMOTE for binary balance ───────────────────────────────────────
    if HAS_IMBLEARN:
        print(f"\n  Applying SMOTE-Tomek for binary balance...")
        benign_count = int((y_train_bin == 0).sum())
        attack_count = int((y_train_bin == 1).sum())
        print(f"    Before: Benign={benign_count:,}, Attack={attack_count:,}")

        try:
            smote = SMOTETomek(random_state=SEED)
            X_train_np, y_train_multi_np = smote.fit_resample(
                X_train.numpy(), y_train_multi.numpy()
            )
            y_train_bin_np = (y_train_multi_np > 0).astype(np.int64)

            X_train = torch.tensor(X_train_np, dtype=torch.float32)
            y_train_multi = torch.tensor(y_train_multi_np, dtype=torch.long)
            y_train_bin = torch.tensor(y_train_bin_np, dtype=torch.long)

            benign_count = int((y_train_bin == 0).sum())
            attack_count = int((y_train_bin == 1).sum())
            print(f"    After:  Benign={benign_count:,}, Attack={attack_count:,}")
            print(f"    Total:  {len(X_train):,} samples")
        except Exception as e:
            print(f"    [WARN] SMOTE failed: {e}")

    model_cfg = config["model"]

    # ── Stage 1: Binary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  STAGE 1: Binary Classification (Benign vs Attack)")
    print("=" * 60)

    bin_model = HiTIDS(
        group_sizes=GROUP_SIZES, group_names=GROUP_NAMES,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"], num_classes=2,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in bin_model.parameters())
    print(f"  Parameters: {n_params:,}")

    best_f1_bin = train_model(
        bin_model, X_train, y_train_bin, X_val, y_val_bin,
        num_classes=2, epochs=150, patience=30, tag="Binary"
    )

    torch.save(
        {"model_state_dict": bin_model.state_dict()},
        os.path.join(CKPT_DIR, "hit_ids_binary_2018.pt"),
    )
    print(f"  ✓ Binary model saved (best F1={best_f1_bin:.4f})")

    # ── Stage 2: Attack multi-class ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  STAGE 2: Attack Classification ({NUM_CLASSES - 1} classes)")
    print("=" * 60)

    atk_mask = y_train_multi > 0
    X_train_atk = X_train[atk_mask]
    y_train_atk = y_train_multi[atk_mask] - 1  # 0-indexed

    atk_mask_val = y_val_multi > 0
    X_val_atk = X_val[atk_mask_val]
    y_val_atk = y_val_multi[atk_mask_val] - 1

    print(f"  Attack train: {len(X_train_atk):,}")
    print(f"  Attack val:   {len(X_val_atk):,}")

    num_attack_classes = NUM_CLASSES - 1  # 2 (Bot=0, Infilteration=1)

    atk_model = HiTIDS(
        group_sizes=GROUP_SIZES, group_names=GROUP_NAMES,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"], num_classes=num_attack_classes,
    ).to(DEVICE)

    best_f1_atk = train_model(
        atk_model, X_train_atk, y_train_atk, X_val_atk, y_val_atk,
        num_classes=num_attack_classes, epochs=100, patience=20, tag="Attack"
    )

    torch.save(
        {"model_state_dict": atk_model.state_dict()},
        os.path.join(CKPT_DIR, "hit_ids_attack_2018.pt"),
    )
    print(f"  ✓ Attack model saved (best F1={best_f1_atk:.4f})")

    # ── Stage 3: Cascade evaluation ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  CASCADE EVALUATION on CIC-IDS2018 Test Set")
    print("=" * 60)

    bin_model.eval()
    atk_model.eval()

    all_preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i:i+batch_size].to(DEVICE)
            bp = bin_model(xb, return_explanations=False).argmax(1)
            final = torch.zeros(len(xb), dtype=torch.long)
            mask = bp == 1
            if mask.any():
                ap = atk_model(xb[mask], return_explanations=False).argmax(1).cpu() + 1
                final[mask] = ap
            all_preds.extend(final.tolist())

    y_true = y_test_multi.numpy()
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy:     {acc:.4f}")
    print(f"  Macro-F1:     {f1_macro:.4f}")
    print(f"  Weighted-F1:  {f1_weighted:.4f}")
    print(f"\n  Per-class:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                 digits=4, zero_division=0))

    # Inference latency
    print("  Measuring inference latency...")
    dummy = X_test[:1].to(DEVICE)
    for _ in range(50):
        bin_model(dummy, return_explanations=False)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        bin_model(dummy, return_explanations=False)
    torch.cuda.synchronize()
    latency_ms = (time.time() - start) / 1000 * 1000
    print(f"  Latency: {latency_ms:.2f} ms/sample")

    # Save results
    results = {
        "dataset": "CIC-IDS2018",
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "binary_f1": float(best_f1_bin),
        "attack_f1": float(best_f1_atk),
        "latency_ms": float(latency_ms),
        "num_classes": NUM_CLASSES,
        "label_map": LABEL_MAP,
        "per_class": {},
    }

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask_true = y_true == cls_id
        if mask_true.sum() > 0:
            cls_f1 = f1_score(
                (y_true == cls_id).astype(int),
                (y_pred == cls_id).astype(int),
                zero_division=0
            )
            results["per_class"][cls_name] = {
                "support": int(mask_true.sum()),
                "f1": float(cls_f1),
            }

    results_path = os.path.join(TABLES_DIR, "cicids2018_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to {results_path}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  COMPLETED in {elapsed/60:.1f} minutes")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro-F1:    {f1_macro:.4f}")
    print(f"  Binary F1:   {best_f1_bin:.4f}")
    print(f"  Attack F1:   {best_f1_atk:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

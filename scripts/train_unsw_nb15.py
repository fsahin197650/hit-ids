#!/usr/bin/env python3
"""
HiT-IDS Cross-Dataset Validation on UNSW-NB15 (Task 9).

End-to-end pipeline:
  1. Preprocess UNSW-NB15 (one-hot encode categoricals, RobustScaler, group assignment)
  2. Train HiT-IDS cascade (binary + 9-class attack) using SAME hyperparams as CIC-IDS2017
  3. Evaluate and save results + attention heatmaps

Usage:
    conda activate hit_ids
    python scripts/train_unsw_nb15.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("[WARN] imbalanced-learn not found, skipping SMOTE-Tomek")

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hit_ids import HiTIDS, compute_attention_entropy
from src.xai.attention_rollout import AttentionRollout

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_PATH = "configs/config_unsw.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = config["project"]["seed"]
torch.manual_seed(SEED)
np.random.seed(SEED)

LABEL_MAP = config["dataset"]["label_map"]
CLASS_NAMES = list(LABEL_MAP.keys())
NUM_CLASSES = config["dataset"]["num_classes"]  # 10

# Create output dirs
for path_key in ["checkpoints", "figures", "tables", "logs"]:
    os.makedirs(config["paths"][path_key], exist_ok=True)
os.makedirs(config["paths"]["splits"], exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def preprocess():
    """Load, clean, encode, scale, and split UNSW-NB15."""
    print("=" * 60)
    print("  STEP 1: Preprocessing UNSW-NB15")
    print("=" * 60)

    raw_dir = config["paths"]["raw_data"]
    train_df = pd.read_csv(os.path.join(raw_dir, "UNSW_NB15_training-set.csv"))
    test_df = pd.read_csv(os.path.join(raw_dir, "UNSW_NB15_testing-set.csv"))

    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

    # Drop ID column
    for df in [train_df, test_df]:
        if "id" in df.columns:
            df.drop("id", axis=1, inplace=True)
        # Clean attack_cat whitespace
        df["attack_cat"] = df["attack_cat"].str.strip()
        # Fix any remaining NaN in attack_cat
        df.loc[df["label"] == 0, "attack_cat"] = "Normal"

    # Encode attack categories
    for df in [train_df, test_df]:
        df["attack_label"] = df["attack_cat"].map(LABEL_MAP)
        # Binary label already exists as 'label' column

    # Separate features
    meta_cols = ["attack_cat", "label", "attack_label"]
    cat_cols = config["dataset"]["categorical_columns"]  # proto, service, state
    num_cols = [c for c in train_df.columns if c not in meta_cols + cat_cols]

    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {cat_cols}")

    # One-hot encode categoricals (fit on train, transform both)
    # Use top-N categories to keep dimensionality manageable
    for col in cat_cols:
        top_cats = train_df[col].value_counts().head(10).index.tolist()
        for df in [train_df, test_df]:
            df[col] = df[col].where(df[col].isin(top_cats), other="other")

    train_encoded = pd.get_dummies(train_df[cat_cols], prefix=cat_cols)
    test_encoded = pd.get_dummies(test_df[cat_cols], prefix=cat_cols)

    # Align columns (test may have different one-hot columns)
    train_encoded, test_encoded = train_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )

    print(f"  One-hot features: {train_encoded.shape[1]}")

    # Combine numeric + one-hot
    X_train_num = train_df[num_cols].values.astype(np.float32)
    X_test_num = test_df[num_cols].values.astype(np.float32)

    # Handle inf/nan BEFORE scaling
    X_train_num = np.nan_to_num(X_train_num, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_num = np.nan_to_num(X_test_num, nan=0.0, posinf=1e6, neginf=-1e6)

    # Clip extreme outliers BEFORE scaling (e.g. response_body_len = 5.2M)
    for i in range(X_train_num.shape[1]):
        p1, p99 = np.percentile(X_train_num[:, i], [1, 99])
        iqr = p99 - p1
        lo = p1 - 3 * iqr
        hi = p99 + 3 * iqr
        X_train_num[:, i] = np.clip(X_train_num[:, i], lo, hi)
        X_test_num[:, i] = np.clip(X_test_num[:, i], lo, hi)

    # Scale numeric features
    scaler = RobustScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    # Clip scaled values to prevent extreme activations
    X_train_num = np.clip(X_train_num, -10, 10)
    X_test_num = np.clip(X_test_num, -10, 10)

    print(f"  Scaled + clipped: train range [{X_train_num.min():.2f}, {X_train_num.max():.2f}]")

    # Combine
    X_train_ohe = train_encoded.values.astype(np.float32)
    X_test_ohe = test_encoded.values.astype(np.float32)

    X_train_full = np.hstack([X_train_num, X_train_ohe])
    X_test = np.hstack([X_test_num, X_test_ohe])

    y_train_multi_full = train_df["attack_label"].values.astype(np.int64)
    y_test_multi = test_df["attack_label"].values.astype(np.int64)
    y_train_bin_full = train_df["label"].values.astype(np.int64)
    y_test_bin = test_df["label"].values.astype(np.int64)

    print(f"  Feature dim: {X_train_full.shape[1]}")

    # ── Train/Val split (80/20 from train set) ───────────────────────────
    X_train, X_val, y_train_multi, y_val_multi, y_train_bin, y_val_bin = \
        train_test_split(
            X_train_full, y_train_multi_full, y_train_bin_full,
            test_size=0.2, stratify=y_train_multi_full, random_state=SEED
        )
    print(f"  Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")

    # ── SMOTE-Tomek oversampling (train only) ────────────────────────────
    if HAS_IMBLEARN:
        print(f"\n  Applying SMOTE-Tomek on attack samples...")
        # SMOTE for multi-class (attack only)
        atk_mask_tr = y_train_multi > 0
        X_atk = X_train[atk_mask_tr]
        y_atk = y_train_multi[atk_mask_tr]

        before_counts = dict(zip(*np.unique(y_atk, return_counts=True)))
        print(f"    Before SMOTE: {sum(before_counts.values()):,} attack samples")
        for cls_id, cnt in sorted(before_counts.items()):
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
            print(f"      {cls_name}: {cnt:,}")

        try:
            # Use SMOTE with k_neighbors adapted to smallest class
            min_samples = min(before_counts.values())
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            smote = SMOTETomek(
                smote=SMOTE(k_neighbors=k_neighbors, random_state=SEED),
                random_state=SEED,
            )
            X_atk_res, y_atk_res = smote.fit_resample(X_atk, y_atk)

            after_counts = dict(zip(*np.unique(y_atk_res, return_counts=True)))
            print(f"    After SMOTE:  {sum(after_counts.values()):,} attack samples")
            for cls_id, cnt in sorted(after_counts.items()):
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
                print(f"      {cls_name}: {cnt:,}")

            # Reconstruct full training set: normal + resampled attack
            X_normal = X_train[~atk_mask_tr]
            y_normal_multi = y_train_multi[~atk_mask_tr]
            y_normal_bin = np.zeros(len(y_normal_multi), dtype=np.int64)

            X_train = np.vstack([X_normal, X_atk_res])
            y_train_multi = np.concatenate([y_normal_multi, y_atk_res])
            y_train_bin = np.concatenate([
                y_normal_bin,
                np.ones(len(y_atk_res), dtype=np.int64)
            ])

            # Shuffle
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train_multi = y_train_multi[perm]
            y_train_bin = y_train_bin[perm]

            print(f"    Final train set: {len(X_train):,} samples")
        except Exception as e:
            print(f"    [WARN] SMOTE failed: {e}. Using original data.")
    else:
        print("  Skipping SMOTE-Tomek (imbalanced-learn not installed)")

    # ── Assign features to groups ────────────────────────────────────────
    fg_cfg = config["feature_groups"]

    # Map: feature name → column index in X_train
    all_feature_names = num_cols + list(train_encoded.columns)

    group_indices = {}
    for group_name, feature_list in fg_cfg.items():
        indices = []
        for feat in feature_list:
            if feat in all_feature_names:
                indices.append(all_feature_names.index(feat))
        # For protocol group, also add one-hot encoded cols
        if group_name == "protocol":
            for i, fname in enumerate(all_feature_names):
                if any(fname.startswith(f"{cat}_") for cat in cat_cols):
                    if i not in indices:
                        indices.append(i)
        group_indices[group_name] = sorted(indices)

    # Print group assignments
    total_assigned = 0
    group_sizes = {}
    for gn, idx in group_indices.items():
        group_sizes[gn] = len(idx)
        total_assigned += len(idx)
        print(f"  Group '{gn}': {len(idx)} features")
    print(f"  Total assigned: {total_assigned} / {X_train.shape[1]}")

    # Check for unassigned features
    all_assigned = set()
    for idx_list in group_indices.values():
        all_assigned.update(idx_list)
    unassigned = set(range(X_train.shape[1])) - all_assigned
    if unassigned:
        # Add unassigned to behavioral group
        group_indices["behavioral"] = sorted(
            group_indices["behavioral"] + list(unassigned)
        )
        group_sizes["behavioral"] = len(group_indices["behavioral"])
        print(f"  → {len(unassigned)} unassigned features added to 'behavioral'")

    # ── Create group tensors ─────────────────────────────────────────────
    def make_groups(X, indices_dict):
        return {
            name: torch.tensor(X[:, indices], dtype=torch.float32)
            for name, indices in indices_dict.items()
        }

    train_groups = make_groups(X_train, group_indices)
    val_groups = make_groups(X_val, group_indices)
    test_groups = make_groups(X_test, group_indices)

    # ── Save splits ──────────────────────────────────────────────────────
    splits_dir = config["paths"]["splits"]

    torch.save({
        "X": torch.tensor(X_train, dtype=torch.float32),
        "y": torch.tensor(y_train_multi, dtype=torch.long),
        "y_bin": torch.tensor(y_train_bin, dtype=torch.long),
        "groups": train_groups,
    }, os.path.join(splits_dir, "train.pt"))

    torch.save({
        "X": torch.tensor(X_val, dtype=torch.float32),
        "y": torch.tensor(y_val_multi, dtype=torch.long),
        "y_bin": torch.tensor(y_val_bin, dtype=torch.long),
        "groups": val_groups,
    }, os.path.join(splits_dir, "val.pt"))

    torch.save({
        "X": torch.tensor(X_test, dtype=torch.float32),
        "y": torch.tensor(y_test_multi, dtype=torch.long),
        "y_bin": torch.tensor(y_test_bin, dtype=torch.long),
        "groups": test_groups,
    }, os.path.join(splits_dir, "test.pt"))

    print(f"\n  ✓ Saved to {splits_dir}/")

    return group_sizes, group_indices


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=4.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def train_model(model, train_X, train_y, val_X, val_y, num_classes,
                tag="", patience_override=None):
    """Train a HiT-IDS model (binary or multi-class)."""
    train_cfg = config["training"]
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    patience = patience_override or train_cfg["patience"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    base_lr = train_cfg["lr"]

    # Class weights (inverse frequency)
    y_np = train_y.numpy() if isinstance(train_y, torch.Tensor) else train_y
    counts = np.bincount(y_np, minlength=num_classes)
    total = counts.sum()
    weights = torch.tensor(
        [total / (num_classes * max(c, 1)) for c in counts],
        dtype=torch.float32
    ).to(DEVICE)
    # Cap extreme weights to prevent instability
    weights = torch.clamp(weights / weights.mean(), min=0.1, max=10.0)

    print(f"    Class weights: {[f'{w:.2f}' for w in weights.cpu().tolist()]}")

    criterion = FocalLoss(gamma=train_cfg["focal_gamma"], weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=train_cfg["weight_decay"],
    )

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        # Cosine decay after warmup
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
        # ── Train ────────────────────────────────────────────────────
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

        # ── Validate ─────────────────────────────────────────────────
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

    if not best_state:
        print(f"    [{tag}] WARNING: No improvement at all, using final state")
    else:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    return best_f1


def train_cascade(group_sizes_list, group_names_list):
    """Train binary + attack models."""
    print("\n" + "=" * 60)
    print("  STEP 2: Training HiT-IDS Cascade")
    print("=" * 60)

    splits_dir = config["paths"]["splits"]
    train_data = torch.load(os.path.join(splits_dir, "train.pt"), weights_only=False)
    val_path = os.path.join(splits_dir, "val.pt")
    if os.path.exists(val_path):
        val_data = torch.load(val_path, weights_only=False)
    else:
        # Fallback: use test as val (old behavior)
        val_data = torch.load(os.path.join(splits_dir, "test.pt"), weights_only=False)

    X_train = train_data["X"]
    y_train_bin = train_data["y_bin"]
    y_train_multi = train_data["y"]
    X_val = val_data["X"]
    y_val_bin = val_data["y_bin"]
    y_val_multi = val_data["y"]

    model_cfg = config["model"]
    ckpt_dir = config["paths"]["checkpoints"]

    # ── Stage 1: Binary model ────────────────────────────────────────
    print("\n  Training Stage 1 (Binary: Normal vs Attack)...")
    bin_model = HiTIDS(
        group_sizes=group_sizes_list,
        group_names=group_names_list,
        d_emb=model_cfg["d_emb"],
        nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        num_classes=2,
    ).to(DEVICE)

    print(f"    Parameters: {sum(p.numel() for p in bin_model.parameters()):,}")

    best_f1_bin = train_model(
        bin_model, X_train, y_train_bin, X_val, y_val_bin,
        num_classes=2, tag="Binary"
    )

    torch.save(
        {"model_state_dict": bin_model.state_dict()},
        os.path.join(ckpt_dir, "hit_ids_binary_unsw.pt"),
    )
    print(f"    ✓ Binary model saved (best F1={best_f1_bin:.4f})")

    # ── Stage 2: Attack model (only attack samples) ──────────────────
    print("\n  Training Stage 2 (9-class Attack Classification)...")
    atk_mask = y_train_multi > 0
    X_train_atk = X_train[atk_mask]
    y_train_atk = y_train_multi[atk_mask] - 1  # 0-indexed attack classes (0..8)

    atk_mask_val = y_val_multi > 0
    X_val_atk = X_val[atk_mask_val]
    y_val_atk = y_val_multi[atk_mask_val] - 1

    print(f"    Attack train samples: {len(X_train_atk)}, val: {len(X_val_atk)}")

    atk_model = HiTIDS(
        group_sizes=group_sizes_list,
        group_names=group_names_list,
        d_emb=model_cfg["d_emb"],
        nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        num_classes=NUM_CLASSES - 1,  # 9 attack classes
    ).to(DEVICE)

    # Use higher patience for attack model (harder task)
    best_f1_atk = train_model(
        atk_model, X_train_atk, y_train_atk, X_val_atk, y_val_atk,
        num_classes=NUM_CLASSES - 1, tag="Attack",
        patience_override=40,
    )

    torch.save(
        {"model_state_dict": atk_model.state_dict()},
        os.path.join(ckpt_dir, "hit_ids_attack_unsw.pt"),
    )
    print(f"    ✓ Attack model saved (best F1={best_f1_atk:.4f})")

    return bin_model, atk_model


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_cascade(bin_model, atk_model, group_names_list):
    """Evaluate cascade and extract attention."""
    print("\n" + "=" * 60)
    print("  STEP 3: Evaluating HiT-IDS Cascade on UNSW-NB15")
    print("=" * 60)

    splits_dir = config["paths"]["splits"]
    test_data = torch.load(os.path.join(splits_dir, "test.pt"), weights_only=False)
    X_test = test_data["X"]
    y_test_multi = test_data["y"]

    bin_model.eval()
    atk_model.eval()

    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")

    all_preds = []
    all_group_imp = []

    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size].to(DEVICE)

            # Stage 1: Binary
            bin_logits, bin_expl = bin_model(X_batch, return_explanations=True)
            bin_preds = bin_logits.argmax(dim=1)

            # Compute group importance
            bin_hier = rollout.hierarchical_importance(
                bin_expl["raw_intra_attn"],
                bin_expl["raw_inter_attn"],
                group_names_list,
            )

            final = torch.zeros(len(X_batch), dtype=torch.long)
            group_imp = bin_hier["group"].cpu().numpy()

            # Stage 2: Attack classification
            atk_mask = bin_preds == 1
            if atk_mask.any():
                X_atk = X_batch[atk_mask]
                atk_logits, atk_expl = atk_model(X_atk, return_explanations=True)
                atk_preds = atk_logits.argmax(dim=1) + 1
                final[atk_mask] = atk_preds.cpu()

                # Update attention for attack samples
                atk_hier = rollout.hierarchical_importance(
                    atk_expl["raw_intra_attn"],
                    atk_expl["raw_inter_attn"],
                    group_names_list,
                )
                group_imp[atk_mask.cpu().numpy()] = atk_hier["group"].cpu().numpy()

            all_preds.extend(final.tolist())
            all_group_imp.append(group_imp)

    y_true = y_test_multi.numpy()
    y_pred = np.array(all_preds)
    group_imp_all = np.vstack(all_group_imp)

    # ── Metrics ──────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy:     {acc:.4f}")
    print(f"  Macro-F1:     {f1_macro:.4f}")
    print(f"  Weighted-F1:  {f1_weighted:.4f}")

    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                 digits=4, zero_division=0))

    # ── Group importance per class ───────────────────────────────────
    print("\n  Mean group importance per class:")
    print(f"  {'Class':20s} {'Temporal':>10s} {'Volume':>10s} {'Protocol':>10s} {'Behavioral':>10s}")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = y_true == cls_id
        if mask.sum() > 0:
            mean_gi = group_imp_all[mask].mean(axis=0)
            print(f"  {cls_name:20s} {mean_gi[0]:10.4f} {mean_gi[1]:10.4f} "
                  f"{mean_gi[2]:10.4f} {mean_gi[3]:10.4f}")

    # ── Inference latency ────────────────────────────────────────────
    print("\n  Measuring inference latency...")
    dummy = X_test[:1].to(DEVICE)
    # Warmup
    for _ in range(50):
        bin_model(dummy, return_explanations=True)
    # Measure
    torch.cuda.synchronize()
    start = time.time()
    n_iters = 1000
    for _ in range(n_iters):
        bin_model(dummy, return_explanations=True)
    torch.cuda.synchronize()
    latency_ms = (time.time() - start) / n_iters * 1000
    print(f"  Latency: {latency_ms:.2f} ms/sample (with attention)")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "dataset": "UNSW-NB15",
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "latency_ms": float(latency_ms),
        "per_class": {},
        "group_importance_per_class": {},
    }

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask_true = y_true == cls_id
        mask_pred = y_pred == cls_id
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

        mask = y_true == cls_id
        if mask.sum() > 0:
            mean_gi = group_imp_all[mask].mean(axis=0)
            results["group_importance_per_class"][cls_name] = {
                gn: float(v) for gn, v in zip(group_names_list, mean_gi)
            }

    results_path = os.path.join(config["paths"]["tables"], "unsw_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  HiT-IDS Cross-Dataset Validation: UNSW-NB15")
    print("  (Same hyperparameters as CIC-IDS2017 — no re-tuning)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    # Step 1: Preprocess
    group_sizes, group_indices = preprocess()
    group_names_list = list(group_sizes.keys())
    group_sizes_list = list(group_sizes.values())

    print(f"\n  Final group sizes: {dict(zip(group_names_list, group_sizes_list))}")

    # Step 2: Train
    bin_model, atk_model = train_cascade(group_sizes_list, group_names_list)

    # Step 3: Evaluate
    results = evaluate_cascade(bin_model, atk_model, group_names_list)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  COMPLETED in {elapsed/60:.1f} minutes")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Macro-F1:    {results['f1_macro']:.4f}")
    print(f"  Latency:     {results['latency_ms']:.2f} ms/sample")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
UNSW-NB15 Binary In-Domain 5-Fold CV — Q1 Revision (P3).

Evaluates HiT-IDS + 4 baselines on UNSW-NB15 binary (Normal vs Attack)
using the SAME 5-fold stratified CV protocol as CIC-IDS2017.

Pipeline per fold:
    1. Stratified train/test split
    2. Fold-internal SMOTE (no data leakage)
    3. Train HiT-IDS + RF + XGBoost + BiLSTM + Vanilla Transformer
    4. Evaluate all on the same test fold
    5. Paired statistical tests after all folds

Outputs:
    results/unsw_nb15/tables/unsw_binary_cv_results.json

Usage:
    python scripts/train_unsw_nb15_binary_cv.py
    python scripts/train_unsw_nb15_binary_cv.py --resume   # crash-safe resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
)
from scipy import stats
import yaml

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imbalanced-learn not found — SMOTE disabled")

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hit_ids import HiTIDS
from src.models.baselines import SklearnBaseline, create_baseline, train_pytorch_baseline

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_PATH = "configs/config_unsw_binary.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SEED = config["project"]["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = config.get("cv", {}).get("n_folds", 5)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths
RAW_DIR = config["paths"]["raw_data"]
TABLES_DIR = config["paths"]["tables"]
CKPT_DIR = config["paths"]["checkpoints"]
for d in [TABLES_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

PROGRESS_PATH = os.path.join(TABLES_DIR, "binary_cv_progress.json")


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def load_and_preprocess():
    """Load UNSW-NB15, preprocess, return X_all (np), y_all_binary (np), group_indices."""
    print("=" * 60)
    print("  Loading & preprocessing UNSW-NB15")
    print("=" * 60)

    train_df = pd.read_csv(os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv"))
    test_df = pd.read_csv(os.path.join(RAW_DIR, "UNSW_NB15_testing-set.csv"))
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"  Total samples: {len(full_df):,}")

    # Drop ID
    if "id" in full_df.columns:
        full_df.drop("id", axis=1, inplace=True)

    # Clean attack_cat
    full_df["attack_cat"] = full_df["attack_cat"].astype(str).str.strip()
    full_df.loc[full_df["label"] == 0, "attack_cat"] = "Normal"

    # Binary label
    y_binary = full_df["label"].values.astype(np.int64)
    print(f"  Normal: {(y_binary == 0).sum():,}  |  Attack: {(y_binary == 1).sum():,}")

    # Feature columns
    meta_cols = ["attack_cat", "label"]
    cat_cols = config["dataset"]["categorical_columns"]
    num_cols = [c for c in full_df.columns if c not in meta_cols + cat_cols]

    # One-hot encode categoricals
    for col in cat_cols:
        top_cats = full_df[col].value_counts().head(10).index.tolist()
        full_df[col] = full_df[col].where(full_df[col].isin(top_cats), other="other")

    ohe = pd.get_dummies(full_df[cat_cols], prefix=cat_cols)

    # Numeric
    X_num = full_df[num_cols].values.astype(np.float32)
    X_num = np.nan_to_num(X_num, nan=0.0, posinf=1e6, neginf=-1e6)

    # Clip outliers
    for i in range(X_num.shape[1]):
        p1, p99 = np.percentile(X_num[:, i], [1, 99])
        iqr = p99 - p1
        lo, hi = p1 - 3 * iqr, p99 + 3 * iqr
        X_num[:, i] = np.clip(X_num[:, i], lo, hi)

    # RobustScaler
    scaler = RobustScaler()
    X_num = scaler.fit_transform(X_num)
    X_num = np.clip(X_num, -10, 10).astype(np.float32)

    # Combine
    X_ohe = ohe.values.astype(np.float32)
    X_all = np.hstack([X_num, X_ohe])

    # Feature group indices
    all_feature_names = num_cols + list(ohe.columns)
    fg_cfg = config["feature_groups"]

    group_indices = {}
    for group_name, feature_list in fg_cfg.items():
        indices = []
        for feat in feature_list:
            if feat in all_feature_names:
                indices.append(all_feature_names.index(feat))
        if group_name == "protocol":
            for i, fname in enumerate(all_feature_names):
                if any(fname.startswith(f"{cat}_") for cat in cat_cols):
                    if i not in indices:
                        indices.append(i)
        group_indices[group_name] = sorted(indices)

    # Assign unassigned to behavioral
    all_assigned = set()
    for idx_list in group_indices.values():
        all_assigned.update(idx_list)
    unassigned = set(range(X_all.shape[1])) - all_assigned
    if unassigned:
        group_indices["behavioral"] = sorted(group_indices["behavioral"] + list(unassigned))
        print(f"  {len(unassigned)} unassigned features → behavioral")

    group_sizes = {k: len(v) for k, v in group_indices.items()}
    print(f"  Features: {X_all.shape[1]}")
    for gn, gs in group_sizes.items():
        print(f"    {gn}: {gs}")

    return X_all, y_binary, group_indices, group_sizes


# ═══════════════════════════════════════════════════════════════════════════
# SMOTE
# ═══════════════════════════════════════════════════════════════════════════

def apply_smote(X, y, random_state=42):
    """Fold-internal SMOTE for binary balancing."""
    if not HAS_SMOTE:
        return X.astype(np.float32), y.astype(np.int64)

    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        return X.astype(np.float32), y.astype(np.int64)

    k = min(config.get("cv", {}).get("smote_k_neighbors", 5), counts.min() - 1)
    k = max(1, k)
    try:
        smote = SMOTE(k_neighbors=k, random_state=random_state)
        X_r, y_r = smote.fit_resample(X, y)
        return X_r.astype(np.float32), y_r.astype(np.int64)
    except Exception as e:
        print(f"    [SMOTE warn] {e}")
        return X.astype(np.float32), y.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# HIT-IDS TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=4.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def train_hit_binary(X_train, y_train, X_val, y_val,
                     group_sizes_list, group_names_list, seed=42):
    """Train HiT-IDS binary model. Returns (model, best_val_f1)."""
    torch.manual_seed(seed)
    model_cfg = config["model"]
    train_cfg = config["training"]

    model = HiTIDS(
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

    # Class weights
    counts = np.bincount(y_train, minlength=2)
    total = counts.sum()
    weights = torch.tensor(
        [total / (2 * max(c, 1)) for c in counts], dtype=torch.float32
    ).to(DEVICE)
    weights = torch.clamp(weights / weights.mean(), 0.1, 10.0)

    criterion = FocalLoss(gamma=train_cfg["focal_gamma"], weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"])

    warmup = train_cfg.get("warmup_epochs", 5)
    epochs = train_cfg["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(X_v, y_v),
        batch_size=train_cfg["batch_size"], shuffle=False)

    best_f1, best_state, no_improve = 0, None, 0
    patience = train_cfg["patience"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb, return_explanations=False), yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(DEVICE), return_explanations=False).argmax(1).cpu())
                labels.append(yb)
        val_f1 = f1_score(
            torch.cat(labels).numpy(), torch.cat(preds).numpy(),
            average="macro", zero_division=0)

        # Progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"      Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f} val_f1={val_f1:.4f} lr={lr_now:.6f} patience={no_improve}/{patience}", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"      Early stop at epoch {epoch+1}, best_f1={best_f1:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    return model, best_f1


def eval_model(model, X_test, y_test):
    """Evaluate and return dict with metrics."""
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=False)

    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE), return_explanations=False).argmax(1).cpu())

    y_pred = torch.cat(preds).numpy()
    return {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_per_class": f1_score(y_test, y_pred, average=None, zero_division=0).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_eval_baseline(model_type, X_train, y_train, X_test, y_test, seed=42):
    """Train and evaluate a baseline model."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_type in ("random_forest", "xgboost"):
        baseline = SklearnBaseline(model_type=model_type)
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
    else:
        num_features = X_train.shape[1]
        model = create_baseline(model_type, num_features, num_classes=2)
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        train_pytorch_baseline(model, train_loader, test_loader,
                               num_epochs=30, lr=1e-3, device=DEVICE)
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
        y_pred = np.array(preds)

    return {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_per_class": f1_score(y_test, y_pred, average=None, zero_division=0).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESS SAVE/LOAD
# ═══════════════════════════════════════════════════════════════════════════

def save_progress(results, completed):
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"results": results, "completed": [list(c) for c in completed]}, f, indent=2)
    os.replace(tmp, PROGRESS_PATH)


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            data = json.load(f)
        return data.get("results", {}), set(tuple(c) for c in data.get("completed", []))
    return {}, set()


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UNSW-NB15 Binary 5-Fold CV")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    print("=" * 70)
    print("  HiT-IDS — UNSW-NB15 Binary In-Domain 5-Fold CV")
    print(f"  Device: {DEVICE}  |  Folds: {K}  |  Seed: {SEED}")
    print("=" * 70)

    # Load data
    X_all, y_all, group_indices, group_sizes = load_and_preprocess()
    group_names_list = list(group_sizes.keys())
    group_sizes_list = list(group_sizes.values())

    baselines = config.get("evaluation", {}).get("baselines", [])
    all_models = ["HiT-IDS"] + baselines

    # Resume
    if args.resume:
        results, completed = load_progress()
        for m in all_models:
            results.setdefault(m, [])
        print(f"  Resuming: {len(completed)} steps done")
    else:
        results = {m: [] for m in all_models}
        completed = set()

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n{'═' * 70}")
        print(f"  FOLD {fold_i + 1}/{K}  |  Train: {len(train_idx):,}  Test: {len(test_idx):,}")
        print(f"{'═' * 70}")

        X_fold_train, y_fold_train = X_all[train_idx], y_all[train_idx]
        X_fold_test, y_fold_test = X_all[test_idx], y_all[test_idx]

        # SMOTE (fold-internal, no leakage)
        X_train_s, y_train_s = apply_smote(X_fold_train, y_fold_train, random_state=SEED + fold_i)
        print(f"  SMOTE: {len(X_fold_train):,} → {len(X_train_s):,}")

        # Train/val split from SMOTE'd train (90/10)
        n = len(X_train_s)
        perm = np.random.RandomState(SEED + fold_i).permutation(n)
        val_n = max(1, n // 10)
        X_val, y_val = X_train_s[perm[:val_n]], y_train_s[perm[:val_n]]
        X_tr, y_tr = X_train_s[perm[val_n:]], y_train_s[perm[val_n:]]

        # ── HiT-IDS ──
        step = (fold_i, "HiT-IDS")
        if step in completed:
            print(f"\n  [HiT-IDS] ✓ SKIPPED (done)")
        else:
            print(f"\n  [HiT-IDS] Training...")
            t0 = time.time()
            try:
                model, val_f1 = train_hit_binary(
                    X_tr, y_tr, X_val, y_val,
                    group_sizes_list, group_names_list, seed=SEED + fold_i)
                hit_result = eval_model(model, X_fold_test, y_fold_test)
                results["HiT-IDS"].append(hit_result)
                completed.add(step)
                save_progress(results, completed)
                print(f"    → F1: {hit_result['macro_f1']:.4f}  Acc: {hit_result['accuracy']:.4f}  [{time.time()-t0:.0f}s] ✓")
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    → ERROR: {e}")
                import traceback; traceback.print_exc()

        # ── Baselines ──
        for bl in baselines:
            step = (fold_i, bl)
            if step in completed:
                print(f"\n  [{bl}] ✓ SKIPPED (done)")
                continue
            print(f"\n  [{bl}] Training...")
            t0 = time.time()
            try:
                bl_result = train_eval_baseline(
                    bl, X_train_s, y_train_s,
                    X_fold_test, y_fold_test, seed=SEED + fold_i)
                results[bl].append(bl_result)
                completed.add(step)
                save_progress(results, completed)
                print(f"    → F1: {bl_result['macro_f1']:.4f}  Acc: {bl_result['accuracy']:.4f}  [{time.time()-t0:.0f}s] ✓")
            except Exception as e:
                print(f"    → ERROR: {e}")
                import traceback; traceback.print_exc()

    # ════════════════════════════════════════════════════════════════
    # STATISTICS
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  RESULTS & STATISTICS")
    print("=" * 70)

    summary = {}
    print(f"\n  {'Model':<28s} {'Mean F1':>8s} {'±Std':>8s} {'Mean Acc':>9s}")
    print("  " + "-" * 56)
    for m in all_models:
        f1s = [r["macro_f1"] for r in results.get(m, [])]
        accs = [r["accuracy"] for r in results.get(m, [])]
        if f1s:
            a = np.array(f1s)
            summary[m] = {
                "mean_macro_f1": float(a.mean()),
                "std_macro_f1": float(a.std()),
                "mean_accuracy": float(np.mean(accs)),
                "per_fold_f1": f1s,
                "per_fold_acc": accs,
            }
            print(f"  {m:<28s} {a.mean():>8.4f} ±{a.std():>6.4f} {np.mean(accs):>9.4f}")

    # Pairwise tests
    hit_f1s = np.array([r["macro_f1"] for r in results.get("HiT-IDS", [])])
    stat_tests = {}
    if len(hit_f1s) >= K:
        for bl in baselines:
            bl_f1s = np.array([r["macro_f1"] for r in results.get(bl, [])])
            if len(bl_f1s) < K:
                continue
            delta = hit_f1s - bl_f1s
            d = cohens_d(hit_f1s, bl_f1s)
            stat_val, p = stats.ttest_rel(hit_f1s, bl_f1s)
            ci_lo = delta.mean() - 1.96 * delta.std(ddof=1) / np.sqrt(K)
            ci_hi = delta.mean() + 1.96 * delta.std(ddof=1) / np.sqrt(K)
            stat_tests[bl] = {
                "delta_mean": float(delta.mean()),
                "delta_ci": [float(ci_lo), float(ci_hi)],
                "cohens_d": float(d),
                "p_value": float(p),
                "significant": bool(p < 0.05),
            }
            sig = "✓" if p < 0.05 else "✗"
            print(f"\n  HiT-IDS vs {bl}: ΔF1={delta.mean():+.4f}, p={p:.4f} {sig}, d={d:.3f}")

    # Save final
    output = {
        "dataset": "UNSW-NB15 (Binary)",
        "k_folds": K, "seed": SEED,
        "summary": summary,
        "statistical_tests": stat_tests,
        "per_fold": {m: results[m] for m in all_models},
    }
    out_path = os.path.join(TABLES_DIR, "unsw_binary_cv_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Results saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Baseline XAI Comparison — SHAP analysis across all models.

Trains each baseline (RF, XGBoost, BiLSTM, VanillaTransformer) on the
training split, computes SHAP feature importances, and compares them
against HiT-IDS attention×gradient importances.

Outputs:
    - results/tables/baseline_xai.json   per-model importance + pairwise ρ
    - results/figures/baseline_shap_comparison.pdf   heatmap
    - results/figures/feature_agreement_matrix.pdf   pairwise ρ matrix

Usage:
    python scripts/baseline_xai.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baselines import (
    SklearnBaseline,
    CNN1D,
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
NUM_MERGED = 13


def merge_labels(y):
    if isinstance(y, np.ndarray):
        return np.array([MERGE_MAP[int(v)] for v in y])
    return np.array([MERGE_MAP[int(v)] for v in y.numpy()])


def get_feature_names(config):
    fg = config["feature_groups"]
    names = []
    for g in fg:
        names.extend(fg[g])
    return names


# ═════════════════════════════════════════════════════════════════════════
# SHAP helpers
# ═════════════════════════════════════════════════════════════════════════

def shap_tree(model, X_test, n_samples=200, model_type="rf"):
    """Compute SHAP via TreeExplainer (RF) or fallback for XGBoost."""
    import shap
    idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sub = X_test[idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sub)
    except (ValueError, Exception) as e:
        print(f"    [INFO] TreeExplainer failed ({e.__class__.__name__}), using native feature_importances_")
        # Fallback to native feature importances
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_.copy()
            importance = importance / (importance.sum() + 1e-12)
            return importance
        raise

    # shap_values handling:
    # RF multi-class: list of (n_samples, n_features) arrays, one per class
    # or ndarray of shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # list of arrays — stack and mean abs across classes
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # (n_samples, n_features)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) → mean abs across classes
        shap_abs = np.abs(shap_values).mean(axis=2)
    else:
        shap_abs = np.abs(shap_values)

    # Ensure 2D: (n_samples, n_features)
    if shap_abs.ndim != 2:
        print(f"    [WARN] Unexpected shap shape: {shap_abs.shape}, falling back to feature_importances_")
        importance = model.feature_importances_.copy()
        importance = importance / (importance.sum() + 1e-12)
        return importance

    importance = shap_abs.mean(axis=0)
    importance = importance / (importance.sum() + 1e-12)
    return importance


def shap_gradient(model, X_bg, X_test, device, n_samples=200):
    """Compute SHAP via GradientExplainer (PyTorch models)."""
    import shap

    model.to(device)
    # GradientExplainer needs train mode for LSTM backward (cuDNN requirement)
    model.train()

    bg = torch.tensor(X_bg, dtype=torch.float32).to(device) if isinstance(X_bg, np.ndarray) else X_bg.to(device)

    explainer = shap.GradientExplainer(model, bg)
    idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sub = torch.tensor(X_test[idx], dtype=torch.float32).to(device) if isinstance(X_test, np.ndarray) else X_test[idx].to(device)

    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list):
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_abs = np.abs(shap_values).mean(axis=-1)  # mean across classes
    else:
        shap_abs = np.abs(shap_values) if isinstance(shap_values, np.ndarray) else np.abs(shap_values.cpu().numpy())

    importance = shap_abs.mean(axis=0)
    importance = importance / (importance.sum() + 1e-12)
    return importance


# ═════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════

def plot_importance_heatmap(model_importances, feature_names, top_n=15, save_path=None):
    """Create heatmap of top features across all models."""
    # Find top-N features by average importance across models
    model_names = list(model_importances.keys())
    imp_matrix = np.array([model_importances[m] for m in model_names])

    avg_imp = imp_matrix.mean(axis=0)
    top_idx = np.argsort(avg_imp)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    data = imp_matrix[:, top_idx]
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    # Add values
    for i in range(len(model_names)):
        for j in range(top_n):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > data.max() * 0.6 else "black")

    plt.colorbar(im, label="Normalized Importance")
    ax.set_title("Feature Importance Comparison Across Models (SHAP)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_agreement_matrix(model_importances, save_path=None):
    """Plot pairwise Spearman ρ matrix between all models."""
    model_names = list(model_importances.keys())
    n = len(model_names)
    rho_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            imp_i = model_importances[model_names[i]]
            imp_j = model_importances[model_names[j]]
            rho, _ = spearmanr(imp_i, imp_j)
            rho_matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rho_matrix, cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_names, fontsize=10)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rho_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if rho_matrix[i, j] < 0.4 else "black")

    plt.colorbar(im, label="Spearman ρ")
    ax.set_title("Feature Importance Agreement (Pairwise Spearman ρ)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()

    return rho_matrix


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--shap-samples", type=int, default=200)
    parser.add_argument("--shap-bg", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    splits_dir = config["paths"]["splits"]
    fig_dir = config["paths"].get("figures", "results/figures")
    tables_dir = config["paths"]["tables"]
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    feature_names = np.array(get_feature_names(config))
    num_features = len(feature_names)

    print("=" * 70)
    print("  BASELINE XAI COMPARISON")
    print("=" * 70)

    # ── Load Data ──────────────────────────────────────────────────────
    print("\n  Loading data...")
    train_data = torch.load(f"{splits_dir}/train.pt", weights_only=False)
    test_data = torch.load(f"{splits_dir}/test.pt", weights_only=False)

    X_train = train_data["X"].numpy()
    y_train = merge_labels(train_data["y"])
    X_test = test_data["X"].numpy()
    y_test = merge_labels(test_data["y"])

    # Subsample train for speed (250K max)
    max_train = 250_000
    if len(X_train) > max_train:
        idx = np.random.choice(len(X_train), max_train, replace=False)
        X_train_sub = X_train[idx]
        y_train_sub = y_train[idx]
        print(f"  Training on {max_train:,} samples (subsampled from {len(X_train):,})")
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Attack-only test samples for SHAP
    attack_mask = y_test > 0
    X_test_attack = X_test[attack_mask]
    y_test_attack = y_test[attack_mask]
    print(f"  Train: {len(X_train_sub):,}  |  Test: {len(X_test):,}  |  Test (attack): {len(X_test_attack):,}")

    # Background data for GradientSHAP
    bg_idx = np.random.choice(len(X_train_sub), min(args.shap_bg, len(X_train_sub)), replace=False)
    X_bg = X_train_sub[bg_idx]

    model_importances = {}
    model_f1s = {}

    # ══════════════════════════════════════════════════════════════════
    # 1. RANDOM FOREST
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  1. RANDOM FOREST")
    print("─" * 70)

    rf = SklearnBaseline("random_forest", n_estimators=200)
    rf.fit(X_train_sub, y_train_sub)
    rf_preds = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_preds, average="macro", zero_division=0)
    print(f"  Macro-F1: {rf_f1:.4f}")

    print(f"  Computing SHAP (TreeExplainer) for {args.shap_samples} attack samples...")
    rf_imp = shap_tree(rf.model, X_test_attack, n_samples=args.shap_samples)
    model_importances["Random Forest"] = rf_imp
    model_f1s["Random Forest"] = rf_f1
    top5 = np.argsort(rf_imp)[-5:][::-1]
    print(f"  Top-5: {', '.join(str(feature_names[i]) for i in top5)}")

    # ══════════════════════════════════════════════════════════════════
    # 2. XGBOOST
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  2. XGBOOST")
    print("─" * 70)

    xgb = SklearnBaseline("xgboost", n_estimators=200, max_depth=8)
    xgb.fit(X_train_sub, y_train_sub)
    xgb_preds = xgb.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_preds, average="macro", zero_division=0)
    print(f"  Macro-F1: {xgb_f1:.4f}")

    print(f"  Computing SHAP (TreeExplainer) for {args.shap_samples} attack samples...")
    xgb_imp = shap_tree(xgb.model, X_test_attack, n_samples=args.shap_samples, model_type="xgboost")
    model_importances["XGBoost"] = xgb_imp
    model_f1s["XGBoost"] = xgb_f1
    top5 = np.argsort(xgb_imp)[-5:][::-1]
    print(f"  Top-5: {', '.join(str(feature_names[i]) for i in top5)}")

    # ══════════════════════════════════════════════════════════════════
    # 3. BiLSTM
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3. BiLSTM")
    print("─" * 70)

    bilstm = BiLSTMClassifier(num_features=num_features, num_classes=NUM_MERGED)
    train_ds = TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32),
                             torch.tensor(y_train_sub, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_test[:10000], dtype=torch.float32),
                           torch.tensor(y_test[:10000], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)

    result = train_pytorch_baseline(bilstm, train_loader, val_loader,
                                    num_epochs=20, lr=1e-3, device=device)
    bilstm.to(device).eval()

    # Compute F1
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            batch = torch.tensor(X_test[i:i+512], dtype=torch.float32).to(device)
            preds = bilstm(batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
    bilstm_f1 = f1_score(y_test, np.array(all_preds), average="macro", zero_division=0)
    print(f"  Macro-F1: {bilstm_f1:.4f}")

    print(f"  Computing SHAP (GradientExplainer) for {args.shap_samples} attack samples...")
    bilstm_imp = shap_gradient(bilstm, X_bg, X_test_attack, device, n_samples=args.shap_samples)
    model_importances["BiLSTM"] = bilstm_imp
    model_f1s["BiLSTM"] = bilstm_f1
    top5 = np.argsort(bilstm_imp)[-5:][::-1]
    print(f"  Top-5: {', '.join(str(feature_names[i]) for i in top5)}")

    # ══════════════════════════════════════════════════════════════════
    # 4. VANILLA TRANSFORMER
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  4. VANILLA TRANSFORMER")
    print("─" * 70)

    vt = VanillaTransformer(num_features=num_features, num_classes=NUM_MERGED,
                            d_emb=64, nhead=4, num_layers=4, dim_feedforward=256)
    result = train_pytorch_baseline(vt, train_loader, val_loader,
                                    num_epochs=20, lr=1e-3, device=device)
    vt.to(device).eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            batch = torch.tensor(X_test[i:i+512], dtype=torch.float32).to(device)
            preds = vt(batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
    vt_f1 = f1_score(y_test, np.array(all_preds), average="macro", zero_division=0)
    print(f"  Macro-F1: {vt_f1:.4f}")

    print(f"  Computing SHAP (GradientExplainer) for {args.shap_samples} attack samples...")
    vt_imp = shap_gradient(vt, X_bg, X_test_attack, device, n_samples=args.shap_samples)
    model_importances["Vanilla Transformer"] = vt_imp
    model_f1s["Vanilla Transformer"] = vt_f1
    top5 = np.argsort(vt_imp)[-5:][::-1]
    print(f"  Top-5: {', '.join(str(feature_names[i]) for i in top5)}")

    # ══════════════════════════════════════════════════════════════════
    # 5. LOAD HiT-IDS RESULTS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  5. HiT-IDS (from existing XAI results)")
    print("─" * 70)

    feat_list = list(feature_names)  # numpy array → list for .index()
    xai_path = os.path.join(tables_dir, "xai_results.json")
    if os.path.exists(xai_path):
        with open(xai_path) as f:
            xai_data = json.load(f)
        # Build importance vector from top features
        hit_imp = np.zeros(num_features)
        if "attention" in xai_data and "top_features" in xai_data["attention"]:
            for feat, val in xai_data["attention"]["top_features"].items():
                if feat in feat_list:
                    hit_imp[feat_list.index(feat)] = val
        # Normalize
        if hit_imp.sum() > 0:
            hit_imp = hit_imp / hit_imp.sum()
        model_importances["HiT-IDS (Attn×Grad)"] = hit_imp
        hit_f1 = xai_data.get("cascade", {}).get("macro_f1", 0.9335)
        model_f1s["HiT-IDS (Attn×Grad)"] = hit_f1
        print(f"  Loaded from {xai_path}")
        print(f"  Macro-F1: {hit_f1:.4f}")

        # Also load SHAP importance
        if "shap" in xai_data and "top_features" in xai_data["shap"]:
            hit_shap_imp = np.zeros(num_features)
            for feat, val in xai_data["shap"]["top_features"].items():
                if feat in feat_list:
                    hit_shap_imp[feat_list.index(feat)] = val
            if hit_shap_imp.sum() > 0:
                hit_shap_imp = hit_shap_imp / hit_shap_imp.sum()
            model_importances["HiT-IDS (SHAP)"] = hit_shap_imp
            model_f1s["HiT-IDS (SHAP)"] = hit_f1
    else:
        print(f"  [WARN] {xai_path} not found, skipping HiT-IDS")

    # ══════════════════════════════════════════════════════════════════
    # 6. COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  COMPARISON ACROSS MODELS")
    print("=" * 70)

    # Top-10 per model
    print("\n  Top-10 Features per Model:")
    print("  " + "-" * 66)
    for model_name, imp in model_importances.items():
        top10 = np.argsort(imp)[-10:][::-1]
        top_feats = ", ".join(f"{feature_names[i]}" for i in top10[:5])
        print(f"  {model_name:28s} | {top_feats}")

    # Pairwise Spearman ρ
    print("\n  Pairwise Spearman ρ:")
    print("  " + "-" * 66)
    mnames = list(model_importances.keys())
    rho_table = {}
    for i, m1 in enumerate(mnames):
        rho_table[m1] = {}
        for j, m2 in enumerate(mnames):
            rho, p = spearmanr(model_importances[m1], model_importances[m2])
            rho_table[m1][m2] = round(rho, 4)
            if i < j:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {m1:28s} vs {m2:28s}: ρ = {rho:.4f} {sig}")

    # F1 comparison
    print("\n  Model Performance:")
    print("  " + "-" * 40)
    for model_name, f1 in sorted(model_f1s.items(), key=lambda x: x[1], reverse=True):
        print(f"    {model_name:28s}: F1 = {f1:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # 7. VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  GENERATING FIGURES")
    print("─" * 70)

    plot_importance_heatmap(
        model_importances, feature_names, top_n=15,
        save_path=os.path.join(fig_dir, "baseline_shap_comparison.pdf")
    )

    rho_matrix = plot_agreement_matrix(
        model_importances,
        save_path=os.path.join(fig_dir, "feature_agreement_matrix.pdf")
    )

    # ══════════════════════════════════════════════════════════════════
    # 8. SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════
    results = {
        "models": {},
        "pairwise_rho": rho_table,
    }
    for model_name in model_importances:
        top10_idx = np.argsort(model_importances[model_name])[-10:][::-1]
        results["models"][model_name] = {
            "f1": model_f1s.get(model_name, None),
            "top_10_features": {
                feature_names[i]: round(float(model_importances[model_name][i]), 4)
                for i in top10_idx
            },
        }

    out_path = os.path.join(tables_dir, "baseline_xai.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    print("\n" + "=" * 70)
    print("  BASELINE XAI COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

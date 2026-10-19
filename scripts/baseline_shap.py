#!/usr/bin/env python3
"""
SHAP analysis on saved baseline models + HiT-IDS comparison.

Loads trained checkpoints and computes SHAP feature importances.
Generates comparison figures and tables.

Usage:
    python scripts/baseline_shap.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baselines import BiLSTMClassifier, VanillaTransformer

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--shap-samples", type=int, default=200)
    parser.add_argument("--shap-bg", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)

    splits_dir = config["paths"]["splits"]
    ckpt_dir = config["paths"]["checkpoints"]
    fig_dir = config["paths"].get("figures", "results/figures")
    tables_dir = config["paths"]["tables"]
    os.makedirs(fig_dir, exist_ok=True)

    feature_names = get_feature_names(config)
    num_features = len(feature_names)

    print("=" * 70)
    print("  BASELINE SHAP ANALYSIS")
    print("=" * 70)

    # ── Load test data ─────────────────────────────────────────────────
    print("\n  Loading test data...")
    test_data = torch.load(f"{splits_dir}/test.pt", weights_only=False)
    train_data = torch.load(f"{splits_dir}/train.pt", weights_only=False)

    X_test = test_data["X"].numpy()
    y_test = merge_labels(test_data["y"])
    X_train = train_data["X"].numpy()

    attack_mask = y_test > 0
    X_attack = X_test[attack_mask]

    # Sample for SHAP
    shap_idx = np.random.choice(len(X_attack), min(args.shap_samples, len(X_attack)), replace=False)
    X_shap = X_attack[shap_idx]

    bg_idx = np.random.choice(len(X_train), min(args.shap_bg, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]

    print(f"  SHAP samples: {len(X_shap)}  |  Background: {len(X_bg)}")

    model_importances = {}

    # ══════════════════════════════════════════════════════════════════
    # 1. RANDOM FOREST — TreeSHAP
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  1. RANDOM FOREST (TreeSHAP)")
    print("─" * 70)

    import shap
    rf_path = os.path.join(ckpt_dir, "baseline_rf.pkl")
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)

    explainer = shap.TreeExplainer(rf_model)
    sv = explainer.shap_values(X_shap)
    # Multi-class: list of arrays → mean abs across classes
    if isinstance(sv, list):
        shap_abs = np.mean([np.abs(s) for s in sv], axis=0)
    elif sv.ndim == 3:
        shap_abs = np.abs(sv).mean(axis=2)
    else:
        shap_abs = np.abs(sv)
    rf_imp = shap_abs.mean(axis=0)
    rf_imp = rf_imp / (rf_imp.sum() + 1e-12)
    model_importances["Random Forest"] = rf_imp

    top5 = np.argsort(rf_imp)[-5:][::-1]
    print(f"  Top-5: {', '.join(feature_names[i] for i in top5)}")

    # ══════════════════════════════════════════════════════════════════
    # 2. XGBOOST — native feature_importances (TreeSHAP has bug)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  2. XGBOOST (Feature Importance — gain)")
    print("─" * 70)

    xgb_path = os.path.join(ckpt_dir, "baseline_xgb.pkl")
    with open(xgb_path, "rb") as f:
        xgb_model = pickle.load(f)

    xgb_imp = xgb_model.feature_importances_.copy()
    xgb_imp = xgb_imp / (xgb_imp.sum() + 1e-12)
    model_importances["XGBoost"] = xgb_imp

    top5 = np.argsort(xgb_imp)[-5:][::-1]
    print(f"  Top-5: {', '.join(feature_names[i] for i in top5)}")

    # ══════════════════════════════════════════════════════════════════
    # 3. BiLSTM — GradientSHAP (with cuDNN disabled)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3. BiLSTM (GradientSHAP)")
    print("─" * 70)

    bilstm_ckpt = torch.load(os.path.join(ckpt_dir, "baseline_bilstm.pt"),
                              map_location=device, weights_only=False)
    bilstm = BiLSTMClassifier(num_features=bilstm_ckpt["num_features"],
                               num_classes=bilstm_ckpt["num_classes"])
    bilstm.load_state_dict(bilstm_ckpt["model_state_dict"])
    bilstm.to(device)

    # Disable cuDNN for LSTM backward compatibility with SHAP
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    bilstm.train()  # Required for RNN backward

    bg_tensor = torch.tensor(X_bg, dtype=torch.float32).to(device)
    shap_tensor = torch.tensor(X_shap, dtype=torch.float32).to(device)

    explainer = shap.GradientExplainer(bilstm, bg_tensor)
    sv = explainer.shap_values(shap_tensor)

    if isinstance(sv, list):
        shap_abs = np.mean([np.abs(s) for s in sv], axis=0)
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_abs = np.abs(sv).mean(axis=-1)
    else:
        shap_abs = np.abs(sv) if isinstance(sv, np.ndarray) else np.abs(sv.cpu().numpy())
    bilstm_imp = shap_abs.mean(axis=0)
    bilstm_imp = bilstm_imp / (bilstm_imp.sum() + 1e-12)
    model_importances["BiLSTM"] = bilstm_imp

    torch.backends.cudnn.enabled = cudnn_was_enabled

    top5 = np.argsort(bilstm_imp)[-5:][::-1]
    print(f"  Top-5: {', '.join(feature_names[i] for i in top5)}")

    # ══════════════════════════════════════════════════════════════════
    # 4. VANILLA TRANSFORMER — GradientSHAP
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  4. VANILLA TRANSFORMER (GradientSHAP)")
    print("─" * 70)

    vt_ckpt = torch.load(os.path.join(ckpt_dir, "baseline_vt.pt"),
                          map_location=device, weights_only=False)
    vt = VanillaTransformer(num_features=vt_ckpt["num_features"],
                             num_classes=vt_ckpt["num_classes"],
                             d_emb=vt_ckpt["d_emb"], nhead=vt_ckpt["nhead"],
                             num_layers=vt_ckpt["num_layers"],
                             dim_feedforward=vt_ckpt["dim_feedforward"])
    vt.load_state_dict(vt_ckpt["model_state_dict"])
    vt.to(device).eval()

    explainer = shap.GradientExplainer(vt, bg_tensor)
    sv = explainer.shap_values(shap_tensor)

    if isinstance(sv, list):
        shap_abs = np.mean([np.abs(s) for s in sv], axis=0)
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_abs = np.abs(sv).mean(axis=-1)
    else:
        shap_abs = np.abs(sv) if isinstance(sv, np.ndarray) else np.abs(sv.cpu().numpy())
    vt_imp = shap_abs.mean(axis=0)
    vt_imp = vt_imp / (vt_imp.sum() + 1e-12)
    model_importances["Vanilla Transformer"] = vt_imp

    top5 = np.argsort(vt_imp)[-5:][::-1]
    print(f"  Top-5: {', '.join(feature_names[i] for i in top5)}")

    # ══════════════════════════════════════════════════════════════════
    # 5. HiT-IDS (from existing XAI results)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  5. HiT-IDS (from existing results)")
    print("─" * 70)

    feat_list = list(feature_names)
    xai_path = os.path.join(tables_dir, "xai_results.json")
    hit_f1 = None
    if os.path.exists(xai_path):
        with open(xai_path) as f:
            xai_data = json.load(f)

        hit_f1 = xai_data.get("cascade_macro_f1", None)

        # Attention importance (array of {name, importance})
        if "top_10_features_attention" in xai_data:
            hit_imp = np.zeros(num_features)
            for item in xai_data["top_10_features_attention"]:
                name = item["name"]
                if name in feat_list:
                    hit_imp[feat_list.index(name)] = item["importance"]
            if hit_imp.sum() > 0:
                hit_imp = hit_imp / hit_imp.sum()
            model_importances["HiT-IDS (Attention)"] = hit_imp

        # SHAP importance (array of {name, importance})
        if "top_10_features_shap" in xai_data:
            hit_shap = np.zeros(num_features)
            for item in xai_data["top_10_features_shap"]:
                name = item["name"]
                if name in feat_list:
                    hit_shap[feat_list.index(name)] = item["importance"]
            if hit_shap.sum() > 0:
                hit_shap = hit_shap / hit_shap.sum()
            model_importances["HiT-IDS (SHAP)"] = hit_shap

        print(f"  Loaded. Macro-F1: {hit_f1:.4f}" if hit_f1 else "  Loaded (no F1)")
    else:
        print(f"  [WARN] {xai_path} not found")

    # ══════════════════════════════════════════════════════════════════
    # 6. PAIRWISE COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PAIRWISE SPEARMAN ρ")
    print("=" * 70)

    mnames = list(model_importances.keys())
    rho_table = {}
    for i, m1 in enumerate(mnames):
        rho_table[m1] = {}
        for j, m2 in enumerate(mnames):
            rho, p = spearmanr(model_importances[m1], model_importances[m2])
            rho_table[m1][m2] = round(rho, 4)
            if i < j:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {m1:28s} vs {m2:28s} : ρ={rho:.4f} {sig}")

    # ══════════════════════════════════════════════════════════════════
    # 7. FIGURES
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  GENERATING FIGURES")
    print("─" * 70)

    # --- Heatmap: top-15 features across models ---
    imp_matrix = np.array([model_importances[m] for m in mnames])
    avg_imp = imp_matrix.mean(axis=0)
    top_n = 15
    top_idx = np.argsort(avg_imp)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(14, 5 + 0.3 * len(mnames)))
    data = imp_matrix[:, top_idx]
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(mnames)))
    ax.set_yticklabels(mnames, fontsize=10)
    for i in range(len(mnames)):
        for j in range(top_n):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i,j] > data.max()*0.6 else "black")
    plt.colorbar(im, label="Normalized Importance")
    ax.set_title("Feature Importance Comparison Across Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(fig_dir, "baseline_shap_comparison.pdf")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.savefig(heatmap_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {heatmap_path}")
    plt.close()

    # --- Agreement matrix ---
    n = len(mnames)
    rho_matrix = np.array([[rho_table[m1][m2] for m2 in mnames] for m1 in mnames])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rho_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(mnames, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(mnames, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rho_matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if rho_matrix[i,j] < 0.4 else "black")
    plt.colorbar(im, label="Spearman ρ")
    ax.set_title("Feature Importance Agreement (Pairwise Spearman ρ)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    agree_path = os.path.join(fig_dir, "feature_agreement_matrix.pdf")
    plt.savefig(agree_path, dpi=300, bbox_inches="tight")
    plt.savefig(agree_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {agree_path}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════
    # 8. SAVE JSON
    # ══════════════════════════════════════════════════════════════════
    results = {"models": {}, "pairwise_rho": rho_table}
    f1_map = {
        "Random Forest": 0.9410, "XGBoost": 0.9297,
        "BiLSTM": 0.9021, "Vanilla Transformer": 0.8647,
    }
    if hit_f1:
        f1_map["HiT-IDS (Attention)"] = hit_f1
        f1_map["HiT-IDS (SHAP)"] = hit_f1

    for m in mnames:
        top10 = np.argsort(model_importances[m])[-10:][::-1]
        results["models"][m] = {
            "f1": f1_map.get(m),
            "top_10_features": {feature_names[i]: round(float(model_importances[m][i]), 4) for i in top10},
        }

    out_path = os.path.join(tables_dir, "baseline_xai.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    print("\n" + "=" * 70)
    print("  BASELINE SHAP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

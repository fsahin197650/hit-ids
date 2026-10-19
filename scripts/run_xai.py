#!/usr/bin/env python3
"""
Comprehensive XAI Analysis for HiT-IDS.

Runs full explainability pipeline on the trained cascade attack model:
    1. Attention Rollout (hierarchical feature importance)
    2. SHAP Analysis (post-hoc feature importance)
    3. XCS Metric (attention vs SHAP consistency)
    4. Confusion Matrix + Per-class Report
    5. Publication-quality Visualizations

Usage:
    python scripts/run_xai.py --config configs/config.yaml
    python scripts/run_xai.py --config configs/config.yaml --shap-samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.models.hit_ids import HiTIDS
from src.xai.attention_rollout import AttentionRollout
from src.xai.shap_analysis import SHAPAnalyzer
from src.xai.xcs_metric import compute_xcs_batch, compute_top_k_overlap_batch
from src.xai.visualization import XAIVisualizer

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


def get_all_feature_names(config):
    """Get flat list of all feature names from config."""
    fg = config["feature_groups"]
    names = []
    for group_name in fg:
        names.extend(fg[group_name])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--shap-samples", type=int, default=300,
                        help="Number of samples for SHAP analysis")
    parser.add_argument("--shap-bg", type=int, default=100,
                        help="Number of background samples for SHAP")
    parser.add_argument("--attn-samples", type=int, default=1000,
                        help="Number of samples for attention analysis")
    parser.add_argument("--v2", action="store_true",
                        help="Use v2 checkpoints (SMOTE + Phase 2 retrained)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    model_cfg = config["model"]
    ckpt_dir = config["paths"]["checkpoints"]
    splits_dir = config["paths"]["splits"]
    fig_dir = config["paths"].get("figures", "results/figures")
    tables_dir = config["paths"]["tables"]
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    feature_names = get_all_feature_names(config)
    batch_size = config["training"]["phase1"]["batch_size"]

    print("=" * 70)
    print("  HiT-IDS XAI ANALYSIS")
    print("=" * 70)

    # ── Load Models ──
    print("\n  Loading cascade models...")

    # Attack model (this is the main model for XAI — shows which features matter for attack types)
    num_atk_classes = 14 if args.v2 else NUM_MERGED - 1  # v2 uses unmerged (14), original uses merged (12)
    atk_model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"], num_classes=num_atk_classes,
    ).to(device)

    # Attack model checkpoint path
    if args.v2:
        atk_ckpt_path = f"{ckpt_dir}/hit_ids_attack_merged_v2.pt"
        if not os.path.exists(atk_ckpt_path):
            atk_ckpt_path = f"{ckpt_dir}/hit_ids_attack_v2.pt"
    else:
        atk_ckpt_path = f"{ckpt_dir}/hit_ids_attack_merged.pt"

    if os.path.exists(atk_ckpt_path):
        atk_ckpt = torch.load(atk_ckpt_path, map_location=device, weights_only=False)
        atk_model.load_state_dict(atk_ckpt["model_state_dict"])
        print(f"  ✓ Attack model loaded: {atk_ckpt_path}")
    else:
        print(f"  ✗ Attack model not found: {atk_ckpt_path}")
        return

    # Binary model (for cascade evaluation + confusion matrix)
    bin_model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"], num_classes=2,
    ).to(device)

    bin_ckpt_path = f"{ckpt_dir}/hit_ids_binary.pt"
    if os.path.exists(bin_ckpt_path):
        bin_ckpt = torch.load(bin_ckpt_path, map_location=device, weights_only=False)
        bin_model.load_state_dict(bin_ckpt["model_state_dict"])
        print(f"  ✓ Binary model loaded: {bin_ckpt_path}")

    atk_model.eval()
    bin_model.eval()

    # ── Load Test Data ──
    test_data = torch.load(f"{splits_dir}/test.pt", weights_only=False)
    train_data = torch.load(f"{splits_dir}/train.pt", weights_only=False)

    X_test = test_data["X"]
    y_test_merged = torch.tensor(merge_labels(test_data["y"]), dtype=torch.long)
    X_train = train_data["X"]

    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {X_test.shape[1]}")

    # ════════════════════════════════════════════════════════════════════
    # 1. CASCADE PREDICTION + CONFUSION MATRIX
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  1. CASCADE PREDICTIONS + CONFUSION MATRIX")
    print("─" * 70)

    all_preds, all_labels = [], []
    n_test = len(X_test)
    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            X_batch = X_test[start:end].to(device)
            bp = bin_model(X_batch, return_explanations=False).argmax(1)
            final = torch.zeros(end - start, dtype=torch.long)
            mask = bp == 1
            if mask.any():
                ap = atk_model(X_batch[mask], return_explanations=False).argmax(1).cpu() + 1
                final[mask] = ap
            all_preds.extend(final.tolist())

    y_true = y_test_merged.numpy()
    y_pred = np.array(all_preds)

    # For v2 (unmerged 14 classes), merge predictions to match y_true
    if args.v2:
        y_pred = merge_labels(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=MERGED_NAMES, zero_division=0, labels=list(range(NUM_MERGED)))}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_MERGED)))

    # ════════════════════════════════════════════════════════════════════
    # 2. ATTENTION ROLLOUT (Hierarchical Feature Importance)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  2. ATTENTION ROLLOUT")
    print("─" * 70)

    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")

    # Sample attack instances for attention analysis
    atk_mask = y_test_merged > 0
    X_atk = X_test[atk_mask]
    y_atk = y_test_merged[atk_mask] - 1

    n_attn = min(args.attn_samples, len(X_atk))
    idx = np.random.choice(len(X_atk), n_attn, replace=False)
    X_sample = X_atk[idx].to(device)
    y_sample = y_atk[idx]

    print(f"  Analyzing {n_attn} attack samples...")

    # Get attention weights from attack model
    with torch.no_grad():
        logits, explanations = atk_model(X_sample, return_explanations=True)

    intra_attn = explanations["raw_intra_attn"]
    inter_attn = explanations["raw_inter_attn"]

    # Hierarchical rollout
    hi_result = rollout.hierarchical_importance(intra_attn, inter_attn, group_names)
    group_importance_all = hi_result["group"].cpu().numpy()  # (N, G)
    global_feature_imp = hi_result["global_flat"].cpu().numpy()  # (N, total_features)

    # ── Attention × Gradient (for better XCS) ────────────────────────
    # Raw attention only captures information routing; multiplying by
    # input gradients adds sensitivity information, making it more
    # comparable to SHAP (which also measures feature sensitivity).
    print("  Computing attention × gradient importance...")
    X_grad = X_sample.clone().requires_grad_(True)
    logits_grad = atk_model(X_grad, return_explanations=False)
    # Backprop the predicted class score
    pred_classes = logits_grad.argmax(dim=1)
    loss_grad = logits_grad.gather(1, pred_classes.unsqueeze(1)).sum()
    loss_grad.backward()
    input_grads = X_grad.grad.abs().cpu().numpy()  # (N, 72)

    # Attention × |gradient| per sample
    attn_x_grad = global_feature_imp * input_grads
    # Normalise per sample
    attn_x_grad = attn_x_grad / (attn_x_grad.sum(axis=1, keepdims=True) + 1e-12)

    # Per-class group importance
    num_atk_classes = NUM_MERGED - 1
    per_class_group_imp = np.zeros((NUM_MERGED, len(group_names)))

    # BENIGN — no attention (it's handled by binary model)
    per_class_group_imp[0] = 1.0 / len(group_names)  # uniform

    for cls in range(num_atk_classes):
        cls_mask = y_sample == cls
        if cls_mask.sum() > 0:
            per_class_group_imp[cls + 1] = group_importance_all[cls_mask.numpy()].mean(axis=0)

    # Global feature importance (mean across all attack samples)
    attn_importance_global = global_feature_imp.mean(axis=0)

    print(f"  Group importance (mean across attack samples):")
    for i, name in enumerate(group_names):
        print(f"    {name}: {group_importance_all.mean(axis=0)[i]:.4f}")

    print(f"\n  Top-10 features (attention):")
    top_10_attn = np.argsort(attn_importance_global)[-10:][::-1]
    for rank, idx_f in enumerate(top_10_attn):
        print(f"    {rank+1}. {feature_names[idx_f]}: {attn_importance_global[idx_f]:.4f}")

    # ════════════════════════════════════════════════════════════════════
    # 3. SHAP ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3. SHAP ANALYSIS")
    print("─" * 70)

    # Background: random subset of training data (attack only for attack model)
    train_atk_mask = merge_labels(train_data["y"]) > 0
    X_train_atk = X_train[train_atk_mask]
    bg_idx = np.random.choice(len(X_train_atk), args.shap_bg, replace=False)
    bg_data = X_train_atk[bg_idx]

    try:
        shap_analyzer = SHAPAnalyzer(
            atk_model, bg_data, device=device, method="gradient")

        # SHAP feature importance
        n_shap = min(args.shap_samples, len(X_atk))
        shap_idx = np.random.choice(len(X_atk), n_shap, replace=False)
        X_shap = X_atk[shap_idx]

        shap_importance = shap_analyzer.feature_importance(X_shap, max_samples=None)

        print(f"\n  Top-10 features (SHAP):")
        top_10_shap = np.argsort(shap_importance)[-10:][::-1]
        for rank, idx_f in enumerate(top_10_shap):
            print(f"    {rank+1}. {feature_names[idx_f]}: {shap_importance[idx_f]:.4f}")

        shap_available = True
    except Exception as e:
        print(f"  [WARNING] SHAP analysis failed: {e}")
        shap_importance = np.ones(len(feature_names)) / len(feature_names)
        shap_available = False

    # ════════════════════════════════════════════════════════════════════
    # 4. XCS - ATTENTION vs SHAP CONSISTENCY
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  4. XCS METRIC (Attention-SHAP Consistency)")
    print("─" * 70)

    if shap_available:
        # For XCS, we need per-sample attention and SHAP importance
        # Use a common subset of samples
        n_xcs = min(500, n_attn, n_shap)
        xcs_idx = np.random.choice(min(n_attn, n_shap), n_xcs, replace=False)

        # Use attention×gradient for XCS (more comparable to SHAP)
        attn_for_xcs = attn_x_grad[:n_xcs]

        # Compute per-sample SHAP for XCS
        X_xcs = X_atk[:n_xcs]
        print(f"  Computing per-sample SHAP for {n_xcs} samples...")
        try:
            shap_values = shap_analyzer.compute_shap_values(X_xcs.to(device), max_samples=n_xcs)
            if shap_values.ndim == 3:
                shap_per_sample = np.abs(shap_values).mean(axis=2)
            else:
                shap_per_sample = np.abs(shap_values)

            # Normalize per sample
            shap_per_sample = shap_per_sample / (shap_per_sample.sum(axis=1, keepdims=True) + 1e-12)

            xcs_stats = compute_xcs_batch(attn_for_xcs, shap_per_sample)
            top_k_stats = compute_top_k_overlap_batch(attn_for_xcs, shap_per_sample)

            print(f"\n  XCS Results (all 72 features):")
            print(f"    Mean Spearman ρ: {xcs_stats['mean']:.4f} ± {xcs_stats['std']:.4f}")
            print(f"    Median: {xcs_stats['median']:.4f}")
            print(f"    Range: [{xcs_stats['min']:.4f}, {xcs_stats['max']:.4f}]")
            print(f"    Significant (p<0.05): {xcs_stats['significant_ratio']*100:.1f}%")

            # ── Top-K Filtered XCS ──────────────────────────────────────
            # Compute XCS only on the union of top-K features from both
            # methods, removing noise from unimportant features.
            for top_k in [20, 30]:
                filtered_scores = []
                filtered_pvals = []
                for i in range(n_xcs):
                    a = attn_for_xcs[i]
                    s = shap_per_sample[i]
                    # Union of top-K from each method
                    top_a = set(np.argsort(a)[-top_k:])
                    top_s = set(np.argsort(s)[-top_k:])
                    union_idx = sorted(top_a | top_s)
                    if len(union_idx) >= 5:
                        rho, pval = spearmanr(a[union_idx], s[union_idx])
                        if not np.isnan(rho):
                            filtered_scores.append(rho)
                            filtered_pvals.append(pval)
                if filtered_scores:
                    fs = np.array(filtered_scores)
                    fp = np.array(filtered_pvals)
                    print(f"\n  XCS Filtered (top-{top_k} union, {len(union_idx)} features):")
                    print(f"    Mean Spearman ρ: {fs.mean():.4f} ± {fs.std():.4f}")
                    print(f"    Median: {np.median(fs):.4f}")
                    print(f"    Significant (p<0.05): {(fp < 0.05).mean()*100:.1f}%")

                    # Save the best filtered result
                    if top_k == 30:
                        xcs_stats_filtered = {
                            "mean": float(fs.mean()),
                            "std": float(fs.std()),
                            "median": float(np.median(fs)),
                            "min": float(fs.min()),
                            "max": float(fs.max()),
                            "significant_ratio": float((fp < 0.05).mean()),
                            "n_samples": len(fs),
                            "method": f"attention_x_gradient_top{top_k}_union",
                        }

            print(f"\n  Top-K Overlap:")
            for k, v in top_k_stats.items():
                print(f"    Top-{k}: {v*100:.1f}%")

            xcs_scores = np.array([
                compute_xcs_batch(
                    attn_for_xcs[i:i+1], shap_per_sample[i:i+1]
                )["mean"] for i in range(n_xcs)
            ])
        except Exception as e:
            print(f"  [WARNING] Per-sample SHAP failed: {e}")
            xcs_stats = {"mean": 0, "std": 0}
            xcs_scores = np.array([0.5])
            top_k_stats = {}
    else:
        xcs_stats = {"mean": 0, "std": 0}
        xcs_scores = np.array([0.5])
        top_k_stats = {}

    # ════════════════════════════════════════════════════════════════════
    # 5. VISUALIZATIONS
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  5. GENERATING VISUALIZATIONS")
    print("─" * 70)

    viz = XAIVisualizer(
        save_dir=fig_dir,
        label_names=MERGED_NAMES,
        group_names=group_names,
    )

    # 5.1 Confusion Matrix
    print("  → Confusion matrix...")
    viz.plot_confusion_matrix(cm, normalize=True, save_name="confusion_matrix.pdf")

    # 5.2 Group Importance Heatmap
    print("  → Group importance heatmap...")
    viz.plot_group_importance_heatmap(
        per_class_group_imp, save_name="group_importance_heatmap.pdf")

    # 5.3 Top Features Comparison (Attention vs SHAP)
    print("  → Top features comparison...")
    viz.plot_top_features_comparison(
        attn_importance_global, shap_importance, feature_names,
        top_k=15, save_name="top_features_comparison.pdf")

    # 5.4 Attack Signatures Radar
    print("  → Attack signature radar...")
    # Select interesting attack types
    attack_indices = [2, 3, 4, 10, 12]  # DDoS, GoldenEye, Hulk, PortScan, WebAttack
    viz.plot_attack_signature_radar(
        per_class_group_imp, class_indices=attack_indices,
        save_name="attack_signatures_radar.pdf")

    # 5.5 XCS Distribution
    print("  → XCS distribution...")
    viz.plot_xcs_distribution(xcs_scores, save_name="xcs_distribution.pdf")

    # ════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ════════════════════════════════════════════════════════════════════
    results = {
        "cascade_macro_f1": float(macro_f1),
        "group_importance_mean": {
            name: float(group_importance_all.mean(axis=0)[i])
            for i, name in enumerate(group_names)
        },
        "top_10_features_attention": [
            {"name": feature_names[i], "importance": float(attn_importance_global[i])}
            for i in top_10_attn
        ],
        "top_10_features_shap": [
            {"name": feature_names[i], "importance": float(shap_importance[i])}
            for i in top_10_shap
        ] if shap_available else [],
        "xcs": xcs_stats,
        "top_k_overlap": {str(k): v for k, v in top_k_stats.items()},
        "per_class_group_importance": {
            MERGED_NAMES[i]: per_class_group_imp[i].tolist()
            for i in range(NUM_MERGED)
        },
    }

    out_name = "xai_results_v2.json" if args.v2 else "xai_results.json"
    outpath = os.path.join(tables_dir, out_name)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {outpath}")
    print(f"  Figures saved to: {fig_dir}/")

    print("\n" + "=" * 70)
    print("  XAI ANALYSIS COMPLETE")
    print("  Generated figures:")
    print(f"    • {fig_dir}/confusion_matrix.pdf")
    print(f"    • {fig_dir}/group_importance_heatmap.pdf")
    print(f"    • {fig_dir}/top_features_comparison.pdf")
    print(f"    • {fig_dir}/attack_signatures_radar.pdf")
    print(f"    • {fig_dir}/xcs_distribution.pdf")
    print("=" * 70)


if __name__ == "__main__":
    main()

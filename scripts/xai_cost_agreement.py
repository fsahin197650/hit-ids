#!/usr/bin/env python3
"""
Explainability Cost & Per-Sample Agreement Analysis.

1. SHAP computational cost per model (ms/sample)
2. Per-sample cross-model SHAP agreement (pairwise Spearman ρ)
3. HiT-IDS attention vs all models' SHAP per-sample agreement

Usage:
    python scripts/xai_cost_agreement.py --config configs/config.yaml
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
from scipy.stats import spearmanr
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baselines import BiLSTMClassifier, VanillaTransformer
from src.models.hit_ids import HiTIDS
from src.xai.attention_rollout import AttentionRollout

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


def get_groups(config):
    fg = config["feature_groups"]
    groups = {}
    for g in fg:
        groups[g] = fg[g]
    return groups


# ═════════════════════════════════════════════════════════════════════════
# Per-sample SHAP helpers
# ═════════════════════════════════════════════════════════════════════════

def shap_tree_per_sample(model, X_samples):
    """Return per-sample SHAP importance matrix (n_samples, n_features)."""
    import shap
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_samples)
    except Exception:
        # XGBoost fallback: use feature_importances_ replicated per sample
        imp = model.feature_importances_.copy()
        imp = imp / (imp.sum() + 1e-12)
        return np.tile(imp, (len(X_samples), 1))

    if isinstance(sv, list):
        shap_abs = np.mean([np.abs(s) for s in sv], axis=0)
    elif sv.ndim == 3:
        shap_abs = np.abs(sv).mean(axis=2)
    else:
        shap_abs = np.abs(sv)
    # Normalize per sample
    row_sums = shap_abs.sum(axis=1, keepdims=True) + 1e-12
    return shap_abs / row_sums


def shap_gradient_per_sample(model, bg_tensor, X_tensor):
    """Return per-sample SHAP importance matrix."""
    import shap
    explainer = shap.GradientExplainer(model, bg_tensor)
    sv = explainer.shap_values(X_tensor)

    if isinstance(sv, list):
        shap_abs = np.mean([np.abs(s) for s in sv], axis=0)
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_abs = np.abs(sv).mean(axis=-1)
    else:
        shap_abs = np.abs(sv) if isinstance(sv, np.ndarray) else np.abs(sv.cpu().numpy())

    row_sums = shap_abs.sum(axis=1, keepdims=True) + 1e-12
    return shap_abs / row_sums


def time_shap(func, *args, n_runs=3):
    """Time a SHAP function and return (result, mean_time_seconds)."""
    times = []
    result = None
    for _ in range(n_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        result = func(*args)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)
    return result, np.mean(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of attack samples for per-sample analysis")
    parser.add_argument("--shap-bg", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)
    torch.manual_seed(42)

    splits_dir = config["paths"]["splits"]
    ckpt_dir = config["paths"]["checkpoints"]
    fig_dir = config["paths"].get("figures", "results/figures")
    tables_dir = config["paths"]["tables"]
    os.makedirs(fig_dir, exist_ok=True)

    feature_names = get_feature_names(config)
    groups = get_groups(config)
    num_features = len(feature_names)

    print("=" * 70)
    print("  XAI COST & PER-SAMPLE AGREEMENT ANALYSIS")
    print("=" * 70)

    # ── Load Data ──────────────────────────────────────────────────────
    print("\n  Loading data...")
    test_data = torch.load(f"{splits_dir}/test.pt", weights_only=False)
    train_data = torch.load(f"{splits_dir}/train.pt", weights_only=False)

    X_test = test_data["X"].numpy()
    y_test = merge_labels(test_data["y"])
    X_train = train_data["X"].numpy()

    attack_mask = y_test > 0
    X_attack = X_test[attack_mask]

    # Select samples
    n = min(args.n_samples, len(X_attack))
    idx = np.random.choice(len(X_attack), n, replace=False)
    X_samples = X_attack[idx]

    bg_idx = np.random.choice(len(X_train), min(args.shap_bg, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]
    bg_tensor = torch.tensor(X_bg, dtype=torch.float32).to(device)
    sample_tensor = torch.tensor(X_samples, dtype=torch.float32).to(device)

    print(f"  Samples: {n}  |  Background: {len(X_bg)}")

    # Dictionary: model_name -> (per_sample_importance_matrix, time_seconds)
    model_shap = {}
    cost_table = {}

    # ══════════════════════════════════════════════════════════════════
    # 1. HiT-IDS ATTENTION (intrinsic — no extra cost)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  1. HiT-IDS — Attention Rollout (intrinsic)")
    print("─" * 70)

    # Load HiT-IDS attack model
    atk_path = os.path.join(ckpt_dir, "hit_ids_attack_merged.pt")
    if not os.path.exists(atk_path):
        atk_path = os.path.join(ckpt_dir, "hit_ids_attack.pt")
    atk_ckpt = torch.load(atk_path, map_location=device, weights_only=False)

    model_cfg = config["model"]
    group_names = list(groups.keys())
    group_sizes = [len(groups[g]) for g in group_names]

    num_classes = atk_ckpt["model_state_dict"]["classification_head.head.3.bias"].shape[0]
    hit_model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=0.0, num_classes=num_classes,
    )
    hit_model.load_state_dict(atk_ckpt["model_state_dict"])
    hit_model.to(device).eval()

    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")

    # Time attention rollout (includes forward pass)
    def get_attention(samples_t):
        with torch.no_grad():
            logits, explanations = hit_model(samples_t, return_explanations=True)
        intra_attn = explanations["raw_intra_attn"]
        inter_attn = explanations["raw_inter_attn"]
        hi_result = rollout.hierarchical_importance(intra_attn, inter_attn, group_names)
        global_imp = hi_result["global_flat"].cpu().numpy()  # (N, 72)
        # Normalize per sample
        row_sums = global_imp.sum(axis=1, keepdims=True) + 1e-12
        return global_imp / row_sums

    attn_imp, attn_time = time_shap(get_attention, sample_tensor)
    ms_per_sample = (attn_time / n) * 1000
    print(f"  Time: {attn_time:.2f}s total, {ms_per_sample:.2f} ms/sample")
    model_shap["HiT-IDS (Attention)"] = attn_imp
    cost_table["HiT-IDS (Attention)"] = {"total_s": round(attn_time, 2),
                                          "ms_per_sample": round(ms_per_sample, 2),
                                          "method": "Attention Rollout (intrinsic)"}

    # ══════════════════════════════════════════════════════════════════
    # 2. HiT-IDS SHAP (post-hoc)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  2. HiT-IDS — GradientSHAP (post-hoc)")
    print("─" * 70)

    import shap

    class FlatWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x, return_explanations=False)

    wrapper = FlatWrapper(hit_model).to(device)
    wrapper.eval()

    def get_hit_shap(bg_t, samp_t):
        explainer = shap.GradientExplainer(wrapper, bg_t)
        return shap_gradient_per_sample.__wrapped__(explainer, samp_t) if hasattr(shap_gradient_per_sample, '__wrapped__') else None

    # Direct computation with timing
    t0 = time.perf_counter()
    hit_explainer = shap.GradientExplainer(wrapper, bg_tensor)
    hit_sv = hit_explainer.shap_values(sample_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    hit_shap_time = time.perf_counter() - t0

    if isinstance(hit_sv, list):
        hit_shap_abs = np.mean([np.abs(s) for s in hit_sv], axis=0)
    elif isinstance(hit_sv, np.ndarray) and hit_sv.ndim == 3:
        hit_shap_abs = np.abs(hit_sv).mean(axis=-1)
    else:
        hit_shap_abs = np.abs(hit_sv) if isinstance(hit_sv, np.ndarray) else np.abs(hit_sv.cpu().numpy())
    row_sums = hit_shap_abs.sum(axis=1, keepdims=True) + 1e-12
    hit_shap_imp = hit_shap_abs / row_sums

    ms_per_sample = (hit_shap_time / n) * 1000
    print(f"  Time: {hit_shap_time:.2f}s total, {ms_per_sample:.2f} ms/sample")
    model_shap["HiT-IDS (SHAP)"] = hit_shap_imp
    cost_table["HiT-IDS (SHAP)"] = {"total_s": round(hit_shap_time, 2),
                                      "ms_per_sample": round(ms_per_sample, 2),
                                      "method": "GradientSHAP (post-hoc)"}

    # ══════════════════════════════════════════════════════════════════
    # 3. RANDOM FOREST — TreeSHAP
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3. Random Forest — TreeSHAP")
    print("─" * 70)

    with open(os.path.join(ckpt_dir, "baseline_rf.pkl"), "rb") as f:
        rf_model = pickle.load(f)

    rf_imp, rf_time = time_shap(shap_tree_per_sample, rf_model, X_samples)
    ms_per_sample = (rf_time / n) * 1000
    print(f"  Time: {rf_time:.2f}s total, {ms_per_sample:.2f} ms/sample")
    model_shap["Random Forest"] = rf_imp
    cost_table["Random Forest"] = {"total_s": round(rf_time, 2),
                                    "ms_per_sample": round(ms_per_sample, 2),
                                    "method": "TreeSHAP"}

    # ══════════════════════════════════════════════════════════════════
    # 4. XGBOOST — native importance (TreeSHAP buggy)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  4. XGBoost — Feature Importance")
    print("─" * 70)

    with open(os.path.join(ckpt_dir, "baseline_xgb.pkl"), "rb") as f:
        xgb_model = pickle.load(f)

    t0 = time.perf_counter()
    xgb_imp_vec = xgb_model.feature_importances_.copy()
    xgb_imp_vec = xgb_imp_vec / (xgb_imp_vec.sum() + 1e-12)
    xgb_imp = np.tile(xgb_imp_vec, (n, 1))
    xgb_time = time.perf_counter() - t0
    ms_per_sample = (xgb_time / n) * 1000
    print(f"  Time: {xgb_time:.4f}s total, {ms_per_sample:.4f} ms/sample (global importance)")
    model_shap["XGBoost"] = xgb_imp
    cost_table["XGBoost"] = {"total_s": round(xgb_time, 4),
                              "ms_per_sample": round(ms_per_sample, 4),
                              "method": "Gain importance (global, not per-sample)"}

    # ══════════════════════════════════════════════════════════════════
    # 5. BiLSTM — GradientSHAP
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  5. BiLSTM — GradientSHAP")
    print("─" * 70)

    bilstm_ckpt = torch.load(os.path.join(ckpt_dir, "baseline_bilstm.pt"),
                              map_location=device, weights_only=False)
    bilstm = BiLSTMClassifier(num_features=bilstm_ckpt["num_features"],
                               num_classes=bilstm_ckpt["num_classes"])
    bilstm.load_state_dict(bilstm_ckpt["model_state_dict"])
    bilstm.to(device)

    cudnn_was = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    bilstm.train()

    t0 = time.perf_counter()
    bilstm_explainer = shap.GradientExplainer(bilstm, bg_tensor)
    bilstm_sv = bilstm_explainer.shap_values(sample_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    bilstm_time = time.perf_counter() - t0

    if isinstance(bilstm_sv, list):
        bilstm_abs = np.mean([np.abs(s) for s in bilstm_sv], axis=0)
    elif isinstance(bilstm_sv, np.ndarray) and bilstm_sv.ndim == 3:
        bilstm_abs = np.abs(bilstm_sv).mean(axis=-1)
    else:
        bilstm_abs = np.abs(bilstm_sv) if isinstance(bilstm_sv, np.ndarray) else np.abs(bilstm_sv.cpu().numpy())
    row_sums = bilstm_abs.sum(axis=1, keepdims=True) + 1e-12
    bilstm_imp = bilstm_abs / row_sums

    torch.backends.cudnn.enabled = cudnn_was

    ms_per_sample = (bilstm_time / n) * 1000
    print(f"  Time: {bilstm_time:.2f}s total, {ms_per_sample:.2f} ms/sample")
    model_shap["BiLSTM"] = bilstm_imp
    cost_table["BiLSTM"] = {"total_s": round(bilstm_time, 2),
                             "ms_per_sample": round(ms_per_sample, 2),
                             "method": "GradientSHAP"}

    # ══════════════════════════════════════════════════════════════════
    # 6. VANILLA TRANSFORMER — GradientSHAP
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  6. Vanilla Transformer — GradientSHAP")
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

    t0 = time.perf_counter()
    vt_explainer = shap.GradientExplainer(vt, bg_tensor)
    vt_sv = vt_explainer.shap_values(sample_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    vt_time = time.perf_counter() - t0

    if isinstance(vt_sv, list):
        vt_abs = np.mean([np.abs(s) for s in vt_sv], axis=0)
    elif isinstance(vt_sv, np.ndarray) and vt_sv.ndim == 3:
        vt_abs = np.abs(vt_sv).mean(axis=-1)
    else:
        vt_abs = np.abs(vt_sv) if isinstance(vt_sv, np.ndarray) else np.abs(vt_sv.cpu().numpy())
    row_sums = vt_abs.sum(axis=1, keepdims=True) + 1e-12
    vt_imp = vt_abs / row_sums

    ms_per_sample = (vt_time / n) * 1000
    print(f"  Time: {vt_time:.2f}s total, {ms_per_sample:.2f} ms/sample")
    model_shap["Vanilla Transformer"] = vt_imp
    cost_table["Vanilla Transformer"] = {"total_s": round(vt_time, 2),
                                          "ms_per_sample": round(ms_per_sample, 2),
                                          "method": "GradientSHAP"}

    # ══════════════════════════════════════════════════════════════════
    # 7. PER-SAMPLE PAIRWISE AGREEMENT
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PER-SAMPLE PAIRWISE AGREEMENT")
    print("=" * 70)

    mnames = list(model_shap.keys())
    pair_results = {}

    for i, m1 in enumerate(mnames):
        for j, m2 in enumerate(mnames):
            if i >= j:
                continue
            rhos = []
            for s in range(n):
                rho, _ = spearmanr(model_shap[m1][s], model_shap[m2][s])
                if not np.isnan(rho):
                    rhos.append(rho)
            mean_rho = np.mean(rhos) if rhos else 0.0
            std_rho = np.std(rhos) if rhos else 0.0
            sig_pct = 100 * sum(1 for r in rhos if abs(r) > 0.3) / len(rhos) if rhos else 0
            pair_results[f"{m1} vs {m2}"] = {
                "mean_rho": round(mean_rho, 4),
                "std_rho": round(std_rho, 4),
                "n_valid": len(rhos),
            }
            print(f"  {m1:28s} vs {m2:28s}: ρ = {mean_rho:.4f} ± {std_rho:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # 8. COST TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EXPLAINABILITY COST COMPARISON")
    print("=" * 70)
    print(f"\n  {'Model':<28s} {'Method':<32s} {'ms/sample':>10s} {'Intrinsic':>10s}")
    print("  " + "─" * 82)
    for name in mnames:
        ct = cost_table[name]
        intrinsic = "✓" if "intrinsic" in ct["method"].lower() else "✗"
        print(f"  {name:<28s} {ct['method']:<32s} {ct['ms_per_sample']:>10.2f} {intrinsic:>10s}")

    # Speedup relative to slowest SHAP
    shap_costs = {k: v["ms_per_sample"] for k, v in cost_table.items()
                  if "intrinsic" not in v["method"].lower() and v["ms_per_sample"] > 0.01}
    if shap_costs:
        attn_ms = cost_table["HiT-IDS (Attention)"]["ms_per_sample"]
        avg_shap_ms = np.mean(list(shap_costs.values()))
        max_shap_ms = max(shap_costs.values())
        print(f"\n  HiT-IDS attention: {attn_ms:.2f} ms/sample")
        print(f"  Average SHAP cost: {avg_shap_ms:.2f} ms/sample")
        print(f"  Speedup vs avg SHAP: {avg_shap_ms/attn_ms:.1f}×" if attn_ms > 0 else "")
        print(f"  Speedup vs max SHAP: {max_shap_ms/attn_ms:.1f}×" if attn_ms > 0 else "")

    # ══════════════════════════════════════════════════════════════════
    # 9. FIGURES
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  GENERATING FIGURES")
    print("─" * 70)

    # Cost bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names_plot = list(cost_table.keys())
    costs = [cost_table[n]["ms_per_sample"] for n in names_plot]
    colors = ["#2ecc71" if "intrinsic" in cost_table[n]["method"].lower() else "#e74c3c"
              for n in names_plot]
    bars = ax.barh(range(len(names_plot)), costs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names_plot)))
    ax.set_yticklabels(names_plot, fontsize=10)
    ax.set_xlabel("Explanation Cost (ms/sample)", fontsize=11)
    ax.set_title("Explainability Computational Cost", fontsize=13, fontweight="bold")
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_width() + max(costs)*0.02, bar.get_y() + bar.get_height()/2,
                f"{cost:.1f} ms", va="center", fontsize=9)
    ax.legend(handles=[plt.Rectangle((0,0),1,1, color="#2ecc71"), plt.Rectangle((0,0),1,1, color="#e74c3c")],
              labels=["Intrinsic (no extra cost)", "Post-hoc SHAP"], loc="lower right")
    plt.tight_layout()
    cost_path = os.path.join(fig_dir, "xai_cost_comparison.pdf")
    plt.savefig(cost_path, dpi=300, bbox_inches="tight")
    plt.savefig(cost_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {cost_path}")
    plt.close()

    # Per-sample agreement heatmap
    n_models = len(mnames)
    rho_mat = np.eye(n_models)
    for i, m1 in enumerate(mnames):
        for j, m2 in enumerate(mnames):
            if i < j:
                key = f"{m1} vs {m2}"
                rho_mat[i, j] = pair_results[key]["mean_rho"]
                rho_mat[j, i] = pair_results[key]["mean_rho"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(rho_mat, cmap="RdYlGn", vmin=-0.1, vmax=1)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(mnames, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(mnames, fontsize=9)
    for i in range(n_models):
        for j in range(n_models):
            ax.text(j, i, f"{rho_mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if rho_mat[i,j] < 0.3 else "black")
    plt.colorbar(im, label="Mean Per-Sample Spearman ρ")
    ax.set_title("Per-Sample Explainability Agreement", fontsize=13, fontweight="bold")
    plt.tight_layout()
    agree_path = os.path.join(fig_dir, "per_sample_agreement.pdf")
    plt.savefig(agree_path, dpi=300, bbox_inches="tight")
    plt.savefig(agree_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {agree_path}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════
    # 10. SAVE JSON
    # ══════════════════════════════════════════════════════════════════
    out = {
        "cost_table": cost_table,
        "per_sample_agreement": pair_results,
        "n_samples": n,
    }
    out_path = os.path.join(tables_dir, "xai_cost_agreement.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

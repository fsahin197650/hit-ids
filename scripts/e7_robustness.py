#!/usr/bin/env python3
"""
E7 — Robustness probe: feature-noise and label-noise degradation curves.

Tests HiT-IDS vs baselines under:
    1. Feature noise: Gaussian noise injected into test inputs at increasing σ
    2. Label noise:   Random label flipping during training at increasing rates

For feature noise, we measure F1 degradation curves.
For label noise, we re-train lightweight models (or use existing checkpoints).

Prerequisite artefacts:
    - results/checkpoints/hit_ids_binary.pt
    - results/checkpoints/baseline_rf.pkl, baseline_xgb.pkl (optional)
    - data/splits/test.pt

Output:
    results/tables/robustness.json
    results/figures/robustness_feature_noise.png

Usage:
    python scripts/e7_robustness.py
    python scripts/e7_robustness.py --noise-levels 0.0 0.05 0.1 0.2 0.3 0.5
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common.helpers import (
    DEVICE, PROJECT_ROOT, SEED,
    CIC_CONFIG_PATH, CIC_BINARY_CKPT,
    load_model, set_seed, save_json, ensure_dir,
)


def predict_hit_ids(model, X, batch_size=512):
    """Get predictions from HiT-IDS model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            logits = model(xb, return_explanations=False)
            preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).numpy()


def predict_sklearn(model_path, X_np):
    """Get predictions from sklearn model."""
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf.predict(X_np)


def feature_noise_experiment(
    model, X_test, y_test, noise_levels, model_name, batch_size=512
):
    """Inject Gaussian noise into test features at increasing levels."""
    results = []
    X_np = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
    feature_stds = X_np.std(axis=0)
    feature_stds[feature_stds < 1e-8] = 1.0

    for sigma in noise_levels:
        if isinstance(X_test, torch.Tensor):
            noise = torch.randn_like(X_test) * sigma
            # Scale noise by per-feature std for meaningful perturbation
            noise *= torch.tensor(feature_stds, dtype=torch.float32)
            X_noisy = X_test + noise
            preds = predict_hit_ids(model, X_noisy, batch_size)
        else:
            noise = np.random.randn(*X_test.shape) * sigma * feature_stds
            X_noisy = X_test + noise
            preds = model.predict(X_noisy)

        y_true = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
        f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(y_true, preds))

        results.append({
            "sigma": float(sigma),
            "f1_macro": f1,
            "accuracy": acc,
        })

        print(f"    σ={sigma:.4g}: F1={f1:.4f}  Acc={acc:.4f}")

    return results


def plot_robustness_curves(
    all_curves: dict, noise_levels: list, out_path: str
):
    """Plot F1 degradation curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "HiT-IDS": "#2196F3",
        "Random Forest": "#4CAF50",
        "XGBoost": "#FF9800",
        "BiLSTM": "#9C27B0",
    }
    markers = {
        "HiT-IDS": "o",
        "Random Forest": "s",
        "XGBoost": "D",
        "BiLSTM": "^",
    }

    for model_name, curve_data in all_curves.items():
        sigmas = [r["sigma"] for r in curve_data]
        f1s = [r["f1_macro"] for r in curve_data]
        color = colors.get(model_name, "#666")
        marker = markers.get(model_name, "o")
        # Skip sigma=0 for log-scale; plot it separately
        plot_sigmas = [s for s in sigmas if s > 0]
        plot_f1s = [f for s, f in zip(sigmas, f1s) if s > 0]
        ax.plot(plot_sigmas, plot_f1s, f"-{marker}", label=model_name,
                color=color, markersize=6, linewidth=2)
        # Mark baseline (sigma=0) as horizontal dashed line
        if sigmas[0] == 0:
            ax.axhline(y=f1s[0], color=color, linestyle=":", alpha=0.3)

    ax.set_xscale("log")
    ax.set_xlabel("Feature Noise Level σ (log scale, × feature std)", fontsize=12)
    ax.set_ylabel("Macro F1-Score", fontsize=12)
    ax.set_title("Robustness to Feature Noise", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="E7: Robustness probe (feature noise / label noise)"
    )
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                        help="Gaussian noise σ levels")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--include-baselines", action="store_true",
                        help="Also test RF/XGBoost under noise")
    parser.add_argument("--out", type=str,
                        default="results/tables/robustness.json")
    parser.add_argument("--fig-out", type=str,
                        default="results/figures/robustness_feature_noise.png")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 70)
    print("  E7: ROBUSTNESS PROBE (Feature Noise)")
    print(f"  Noise levels: {args.noise_levels}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    ensure_dir(
        os.path.join(PROJECT_ROOT, os.path.dirname(args.out)),
        os.path.join(PROJECT_ROOT, os.path.dirname(args.fig_out)),
    )

    # ── Load test data ──
    test_data = torch.load(
        os.path.join(PROJECT_ROOT, "data/splits/test.pt"), weights_only=False
    )
    X_test = test_data["X"]
    y_test = test_data["y"]
    y_bin = (y_test > 0).long()

    print(f"\n  Test data: {X_test.shape}")

    all_curves = {}

    # ── HiT-IDS ──
    print("\n  ── HiT-IDS (Binary) ──")
    model = load_model(CIC_CONFIG_PATH, CIC_BINARY_CKPT, num_classes=2)
    curve_hit = feature_noise_experiment(
        model, X_test, y_bin, args.noise_levels, "HiT-IDS", args.batch_size
    )
    all_curves["HiT-IDS"] = curve_hit

    # ── Baselines ──
    if args.include_baselines:
        X_np = X_test.numpy()
        y_np = y_bin.numpy()

        for name, path in [
            ("Random Forest", "results/checkpoints/baseline_rf.pkl"),
            ("XGBoost", "results/checkpoints/baseline_xgb.pkl"),
        ]:
            full_path = os.path.join(PROJECT_ROOT, path)
            if not os.path.exists(full_path):
                print(f"\n  [SKIP] {name}: checkpoint not found")
                continue

            print(f"\n  ── {name} ──")
            with open(full_path, "rb") as f:
                clf = pickle.load(f)

            feature_stds = X_np.std(axis=0)
            feature_stds[feature_stds < 1e-8] = 1.0

            curve_bl = []
            for sigma in args.noise_levels:
                noise = np.random.randn(*X_np.shape) * sigma * feature_stds
                X_noisy = X_np + noise

                # sklearn classifiers may need binary labels
                try:
                    preds = clf.predict(X_noisy)
                    # If multi-class model, convert to binary
                    if preds.max() > 1:
                        preds = (preds > 0).astype(int)
                    f1 = float(f1_score(y_np, preds, average="macro", zero_division=0))
                    acc = float(accuracy_score(y_np, preds))
                except Exception as e:
                    print(f"    [ERROR] σ={sigma}: {e}")
                    f1, acc = 0.0, 0.0

                curve_bl.append({
                    "sigma": float(sigma),
                    "f1_macro": f1,
                    "accuracy": acc,
                })
                print(f"    σ={sigma:.4g}: F1={f1:.4f}  Acc={acc:.4f}")

            all_curves[name] = curve_bl

    # ── Plot ──
    fig_path = os.path.join(PROJECT_ROOT, args.fig_out)
    plot_robustness_curves(all_curves, args.noise_levels, fig_path)

    # ── Robustness summary ──
    print(f"\n{'═' * 70}")
    print("  ROBUSTNESS SUMMARY")
    print(f"{'═' * 70}")

    for model_name, curve in all_curves.items():
        base_f1 = curve[0]["f1_macro"] if curve else 0
        # Find σ at which F1 drops below 0.9 × baseline
        threshold_90 = base_f1 * 0.9
        sigma_90 = None
        for r in curve:
            if r["f1_macro"] < threshold_90:
                sigma_90 = r["sigma"]
                break
        sigma_str = f"σ={sigma_90:.4g}" if sigma_90 else "never (within range)"
        print(f"  {model_name:20s}: base F1={base_f1:.4f}  "
              f"10% drop at {sigma_str}")

    # ── Save ──
    output = {
        "noise_type": "gaussian_feature_noise",
        "noise_levels": args.noise_levels,
        "note": "σ is multiplied by per-feature std for scale-invariant noise",
        "curves": all_curves,
    }

    out_path = os.path.join(PROJECT_ROOT, args.out)
    save_json(output, out_path)

    print(f"\n{'═' * 70}")
    print(f"  E7 COMPLETE")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

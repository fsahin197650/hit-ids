#!/usr/bin/env python3
"""
E4 — Calibration analysis: Expected Calibration Error (ECE) + reliability diagram.

Computes ECE for HiT-IDS (binary) on CIC-IDS2017 and optionally UNSW-NB15.
Also produces reliability diagrams (predicted confidence vs actual accuracy).
Optionally compares with RF/XGBoost baselines.

Prerequisite artefacts:
    - results/checkpoints/hit_ids_binary.pt
    - data/splits/test.pt
    - (optional) results/unsw_nb15/checkpoints/hit_ids_binary_unsw.pt
    - (optional) results/checkpoints/baseline_rf.pkl, baseline_xgb.pkl

Output:
    results/tables/calibration.json
    results/figures/reliability_diagram.png

Usage:
    python scripts/e4_calibration.py
    python scripts/e4_calibration.py --n-bins 15 --include-baselines
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common.helpers import (
    DEVICE, PROJECT_ROOT, SEED, CIC_CONFIG_PATH, CIC_BINARY_CKPT,
    UNSW_CONFIG_PATH, UNSW_BINARY_CKPT,
    load_model, set_seed, save_json, ensure_dir,
)


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """
    Expected Calibration Error (ECE).

    ECE = Σ (|B_m| / N) × |acc(B_m) − conf(B_m)|

    Returns:
        Dict with ECE, per-bin stats, and MCE (max calibration error).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()

        if n_in_bin == 0:
            bins.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "count": 0, "avg_confidence": 0.0, "avg_accuracy": 0.0,
                "gap": 0.0,
            })
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        gap = abs(avg_acc - avg_conf)

        ece += (n_in_bin / len(labels)) * gap

        bins.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "count": int(n_in_bin),
            "avg_confidence": float(avg_conf),
            "avg_accuracy": float(avg_acc),
            "gap": float(gap),
        })

    mce = max(b["gap"] for b in bins if b["count"] > 0) if bins else 0.0

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "n_samples": len(labels),
        "overall_accuracy": float(accuracies.mean()),
        "avg_confidence": float(confidences.mean()),
        "bins": bins,
    }


def get_hit_ids_probs(model, X, batch_size=512):
    """Get softmax probabilities from HiT-IDS."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            logits = model(xb, return_explanations=False)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


def get_baseline_probs(model_path: str, X_np: np.ndarray) -> np.ndarray | None:
    """Get predicted probabilities from sklearn baseline."""
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        # sklearn predict_proba returns (N, n_classes)
        probs = clf.predict_proba(X_np)
        return probs
    except Exception as e:
        print(f"    [WARN] Could not load {model_path}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# RELIABILITY DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def plot_reliability_diagram(
    cal_results: dict, out_path: str, title: str = "Reliability Diagram"
):
    """Plot reliability diagram with gap bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, cal in cal_results.items():
        bins_data = cal["bins"]
        confs = [b["avg_confidence"] for b in bins_data if b["count"] > 0]
        accs = [b["avg_accuracy"] for b in bins_data if b["count"] > 0]

        ax1.plot(confs, accs, "o-", label=f"{model_name} (ECE={cal['ece']:.4f})", markersize=5)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax1.set_ylabel("Observed accuracy in bin", fontsize=11)
    ax1.set_title(title, fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Bar chart of bin counts (first model only)
    first_model = list(cal_results.keys())[0]
    bins_data = cal_results[first_model]["bins"]
    bin_mids = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins_data]
    counts = [b["count"] for b in bins_data]
    width = 1.0 / len(bins_data) * 0.8
    ax2.bar(bin_mids, counts, width=width, alpha=0.7, color="steelblue")
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title(f"Confidence distribution of {first_model}", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="E4: Calibration analysis (ECE + reliability diagram)"
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--include-baselines", action="store_true",
                        help="Include RF/XGBoost calibration comparison")
    parser.add_argument("--out", type=str,
                        default="results/tables/calibration.json")
    parser.add_argument("--fig-out", type=str,
                        default="results/figures/reliability_diagram.png")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 70)
    print("  E4: CALIBRATION ANALYSIS (ECE + Reliability Diagram)")
    print(f"  Bins: {args.n_bins}  |  Device: {DEVICE}")
    print("=" * 70)

    ensure_dir(
        os.path.join(PROJECT_ROOT, os.path.dirname(args.out)),
        os.path.join(PROJECT_ROOT, os.path.dirname(args.fig_out)),
    )

    all_results = {}
    cal_for_plot = {}

    # ── CIC-IDS2017 ──
    print("\n  ── CIC-IDS2017 ──")
    test_data = torch.load(
        os.path.join(PROJECT_ROOT, "data/splits/test.pt"), weights_only=False
    )
    X_test = test_data["X"]
    y_test = test_data["y"]
    y_bin = (y_test > 0).long().numpy()

    model_cic = load_model(CIC_CONFIG_PATH, CIC_BINARY_CKPT, num_classes=2)
    probs_cic = get_hit_ids_probs(model_cic, X_test)
    cal_cic = compute_ece(probs_cic, y_bin, args.n_bins)
    all_results["CIC-IDS2017_HiT-IDS"] = cal_cic
    cal_for_plot["HiT-IDS (CIC-2017)"] = cal_cic
    print(f"    ECE = {cal_cic['ece']:.4f}  |  MCE = {cal_cic['mce']:.4f}  |  "
          f"Acc = {cal_cic['overall_accuracy']:.4f}")

    # Baselines
    if args.include_baselines:
        X_np = X_test.numpy()
        for name, path in [
            ("RF", "results/checkpoints/baseline_rf.pkl"),
            ("XGBoost", "results/checkpoints/baseline_xgb.pkl"),
        ]:
            full_path = os.path.join(PROJECT_ROOT, path)
            probs_bl = get_baseline_probs(full_path, X_np)
            if probs_bl is not None:
                # Convert multi-class probs to binary
                if probs_bl.shape[1] > 2:
                    # Sum non-normal classes as "attack"
                    p_attack = probs_bl[:, 1:].sum(axis=1)
                    probs_bl_bin = np.column_stack([1 - p_attack, p_attack])
                else:
                    probs_bl_bin = probs_bl
                cal_bl = compute_ece(probs_bl_bin, y_bin, args.n_bins)
                all_results[f"CIC-IDS2017_{name}"] = cal_bl
                cal_for_plot[f"{name} (CIC-2017)"] = cal_bl
                print(f"    {name}: ECE = {cal_bl['ece']:.4f}")

    # ── UNSW-NB15 (if available) ──
    unsw_ckpt = os.path.join(PROJECT_ROOT, UNSW_BINARY_CKPT)
    unsw_test_path = os.path.join(PROJECT_ROOT, "data/unsw_nb15/splits/test.pt")

    if os.path.exists(unsw_ckpt):
        print("\n  ── UNSW-NB15 ──")
        model_unsw = load_model(UNSW_CONFIG_PATH, unsw_ckpt, num_classes=2)

        # Try to load UNSW test data
        if os.path.exists(unsw_test_path):
            unsw_data = torch.load(unsw_test_path, weights_only=False)
            X_unsw = unsw_data["X"]
            y_unsw = unsw_data.get("y_bin", unsw_data.get("y"))
            if y_unsw.max() > 1:
                y_unsw = (y_unsw > 0).long()
        else:
            # Fallback: use training data tail
            train_path = os.path.join(PROJECT_ROOT, "data/unsw_nb15/splits/train.pt")
            if os.path.exists(train_path):
                unsw_data = torch.load(train_path, weights_only=False)
                n_test = int(len(unsw_data["X"]) * 0.2)
                X_unsw = unsw_data["X"][-n_test:]
                y_unsw = unsw_data.get("y_bin", unsw_data.get("y"))[-n_test:]
                if y_unsw.max() > 1:
                    y_unsw = (y_unsw > 0).long()
            else:
                X_unsw = None

        if X_unsw is not None:
            probs_unsw = get_hit_ids_probs(model_unsw, X_unsw)
            cal_unsw = compute_ece(probs_unsw, y_unsw.numpy(), args.n_bins)
            all_results["UNSW-NB15_HiT-IDS"] = cal_unsw
            cal_for_plot["HiT-IDS (UNSW-NB15)"] = cal_unsw
            print(f"    ECE = {cal_unsw['ece']:.4f}  |  MCE = {cal_unsw['mce']:.4f}")

    # ── Plot reliability diagram ──
    if cal_for_plot:
        fig_path = os.path.join(PROJECT_ROOT, args.fig_out)
        plot_reliability_diagram(cal_for_plot, fig_path, "Reliability diagrams (10 bins): Stage-1 HiT-IDS detector and tree ensembles")

    # ── Save ──
    out_path = os.path.join(PROJECT_ROOT, args.out)
    # Strip bin details for clean JSON (keep in nested structure)
    save_json(all_results, out_path)

    print(f"\n{'═' * 70}")
    print(f"  E4 COMPLETE")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
E8 — GMM adaptive threshold benefit-cost sweep.

Analyzes when the adaptive GMM threshold helps vs hurts by measuring the
distribution shift between source and target softmax outputs and correlating
it with ΔF1 (adaptive - default).

Uses existing cross-dataset transfer results (Exp A, B, E, F) plus the E1
rerun results.

Prerequisite artefacts:
    - results/cross_dataset/exp_*.json (existing transfer results)
    - results/cross_dataset/transfer_rerun_nb15binary.json (from E1)
    - Model checkpoints + aligned data (for fresh softmax distribution analysis)

Output:
    results/tables/threshold_sweep.json
    results/figures/threshold_benefit_scatter.png

Usage:
    python scripts/e8_threshold_sweep.py
    python scripts/e8_threshold_sweep.py --recompute-distributions
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common.helpers import (
    DEVICE, PROJECT_ROOT, SEED,
    CIC_CONFIG_PATH, CIC_BINARY_CKPT,
    UNSW_CONFIG_PATH, UNSW_BINARY_CKPT,
    ALIGNED_NB15_FOR_CIC_MODEL, ALIGNED_CIC_FOR_NB15_MODEL,
    load_model, set_seed, save_json, ensure_dir,
)


# ═══════════════════════════════════════════════════════════════════════════
# DISTRIBUTION DISTANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def js_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 50) -> float:
    """Jensen-Shannon divergence between two 1D distributions."""
    # Histogram-based
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    # Add small epsilon for numerical stability
    eps = 1e-12
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalize
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    m = 0.5 * (p_hist + q_hist)

    kl_pm = np.sum(p_hist * np.log(p_hist / m))
    kl_qm = np.sum(q_hist * np.log(q_hist / m))

    return float(0.5 * kl_pm + 0.5 * kl_qm)


def wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    """1-Wasserstein (Earth Mover's) distance."""
    return float(scipy_stats.wasserstein_distance(p, q))


def ks_statistic(p: np.ndarray, q: np.ndarray) -> tuple:
    """Kolmogorov-Smirnov statistic and p-value."""
    stat, pval = scipy_stats.ks_2samp(p, q)
    return float(stat), float(pval)


def compute_softmax_distribution(model, X, batch_size=512):
    """Get softmax attack-class probabilities for a dataset."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            logits = model(xb, return_explanations=False)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E8: GMM adaptive threshold benefit-cost sweep"
    )
    parser.add_argument("--recompute-distributions", action="store_true",
                        help="Recompute softmax distributions from models")
    parser.add_argument("--out", type=str,
                        default="results/tables/threshold_sweep.json")
    parser.add_argument("--fig-out", type=str,
                        default="results/figures/threshold_benefit_scatter.png")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 70)
    print("  E8: GMM ADAPTIVE THRESHOLD BENEFIT-COST SWEEP")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    ensure_dir(
        os.path.join(PROJECT_ROOT, os.path.dirname(args.out)),
        os.path.join(PROJECT_ROOT, os.path.dirname(args.fig_out)),
    )

    # ── Collect existing transfer results ──
    print("\n  ── Loading existing transfer results ──")
    cross_dir = os.path.join(PROJECT_ROOT, "results/cross_dataset")
    experiments = []

    for fpath in sorted(glob.glob(os.path.join(cross_dir, "exp_*.json"))):
        with open(fpath) as f:
            r = json.load(f)

        exp_name = r.get("experiment", os.path.basename(fpath))
        default_f1 = r.get("f1_macro", 0)
        adaptive_f1 = r.get("adaptive_f1_macro", 0)
        adaptive_thr = r.get("adaptive_threshold", 0.5)
        delta_f1 = adaptive_f1 - default_f1

        exp_entry = {
            "experiment": exp_name,
            "description": r.get("description", ""),
            "train_dataset": r.get("train_dataset", "?"),
            "test_dataset": r.get("test_dataset", "?"),
            "default_f1_macro": default_f1,
            "adaptive_f1_macro": adaptive_f1,
            "adaptive_threshold": adaptive_thr,
            "delta_f1": delta_f1,
            "benefit": delta_f1 > 0,
            "prob_stats": {
                "mean": r.get("prob_attack_mean"),
                "max": r.get("prob_attack_max"),
                "p95": r.get("prob_attack_p95"),
            },
        }
        experiments.append(exp_entry)
        status = "BENEFIT ↑" if delta_f1 > 0 else "COST ↓" if delta_f1 < 0 else "NEUTRAL"
        print(f"    {exp_name}: ΔF1={delta_f1:+.4f} ({status})  "
              f"thr={adaptive_thr:.3f}")

    # ── Load E1 rerun results if available ──
    e1_path = os.path.join(cross_dir, "transfer_rerun_nb15binary.json")
    if os.path.exists(e1_path):
        print("\n  ── Loading E1 rerun results ──")
        with open(e1_path) as f:
            e1 = json.load(f)

        for exp_key in ["exp_A_rerun", "exp_E_rerun"]:
            if exp_key in e1:
                r = e1[exp_key]
                default_f1 = r.get("default", {}).get("f1_macro", 0)
                adaptive_f1 = r.get("adaptive", {}).get("f1_macro", 0)
                adaptive_thr = r.get("adaptive_threshold", 0.5)
                delta_f1 = adaptive_f1 - default_f1

                exp_entry = {
                    "experiment": exp_key,
                    "description": r.get("direction", ""),
                    "train_dataset": r.get("source_model", "?"),
                    "test_dataset": "target",
                    "default_f1_macro": default_f1,
                    "adaptive_f1_macro": adaptive_f1,
                    "adaptive_threshold": adaptive_thr,
                    "delta_f1": delta_f1,
                    "benefit": delta_f1 > 0,
                    "prob_stats": r.get("prob_stats", {}),
                }
                experiments.append(exp_entry)
                status = "BENEFIT ↑" if delta_f1 > 0 else "COST ↓"
                print(f"    {exp_key}: ΔF1={delta_f1:+.4f} ({status})")

    # ── Compute distribution distances (if --recompute-distributions) ──
    if args.recompute_distributions:
        print("\n  ── Computing softmax distribution distances ──")

        # Source distributions (in-domain)
        source_dists = {}

        # CIC-IDS2017 source distribution
        cic_test = torch.load(
            os.path.join(PROJECT_ROOT, "data/splits/test.pt"), weights_only=False
        )
        model_cic = load_model(CIC_CONFIG_PATH, CIC_BINARY_CKPT, num_classes=2)
        source_dists["CIC-IDS2017"] = compute_softmax_distribution(
            model_cic, cic_test["X"]
        )
        print(f"    CIC source dist: N={len(source_dists['CIC-IDS2017'])}")

        # NB15 source distribution (if available)
        unsw_ckpt = os.path.join(PROJECT_ROOT, UNSW_BINARY_CKPT)
        if os.path.exists(unsw_ckpt):
            train_path = os.path.join(PROJECT_ROOT, "data/unsw_nb15/splits/train.pt")
            if os.path.exists(train_path):
                unsw_data = torch.load(train_path, weights_only=False)
                n_test = int(len(unsw_data["X"]) * 0.2)
                model_unsw = load_model(UNSW_CONFIG_PATH, unsw_ckpt, num_classes=2)
                source_dists["UNSW-NB15"] = compute_softmax_distribution(
                    model_unsw, unsw_data["X"][-n_test:]
                )
                print(f"    NB15 source dist: N={len(source_dists['UNSW-NB15'])}")

        # Target distributions for each transfer
        transfer_configs = []

        # Exp A: CIC→NB15
        nb15_aligned = os.path.join(PROJECT_ROOT, ALIGNED_NB15_FOR_CIC_MODEL)
        if os.path.exists(nb15_aligned):
            data = torch.load(nb15_aligned, weights_only=False)
            target_probs = compute_softmax_distribution(model_cic, data["X"])
            js = js_divergence(source_dists["CIC-IDS2017"], target_probs)
            wd = wasserstein_1d(source_dists["CIC-IDS2017"], target_probs)
            ks, ks_p = ks_statistic(source_dists["CIC-IDS2017"], target_probs)
            transfer_configs.append({
                "experiment": "A",
                "js_divergence": js,
                "wasserstein": wd,
                "ks_statistic": ks,
            })
            print(f"    Exp A: JSD={js:.4f}  W1={wd:.4f}  KS={ks:.4f}")

        # Exp E: NB15→CIC
        cic_aligned = os.path.join(PROJECT_ROOT, ALIGNED_CIC_FOR_NB15_MODEL)
        if os.path.exists(cic_aligned) and "UNSW-NB15" in source_dists:
            data = torch.load(cic_aligned, weights_only=False)
            model_unsw = load_model(UNSW_CONFIG_PATH, unsw_ckpt, num_classes=2)
            target_probs = compute_softmax_distribution(model_unsw, data["X"])
            js = js_divergence(source_dists["UNSW-NB15"], target_probs)
            wd = wasserstein_1d(source_dists["UNSW-NB15"], target_probs)
            ks, ks_p = ks_statistic(source_dists["UNSW-NB15"], target_probs)
            transfer_configs.append({
                "experiment": "E",
                "js_divergence": js,
                "wasserstein": wd,
                "ks_statistic": ks,
            })
            print(f"    Exp E: JSD={js:.4f}  W1={wd:.4f}  KS={ks:.4f}")

        # Merge distance info into experiments
        for tc in transfer_configs:
            for exp in experiments:
                if exp["experiment"] == tc["experiment"]:
                    exp["distribution_distance"] = tc
                    break
    else:
        print("\n  [INFO] Skipping distribution recomputation. "
              "Use --recompute-distributions to enable.")

    # ── Analysis: benefit vs cost framework ──
    print(f"\n{'═' * 70}")
    print("  ADAPTIVE THRESHOLD BENEFIT-COST ANALYSIS")
    print(f"{'═' * 70}")

    benefits = [e for e in experiments if e["delta_f1"] > 0]
    costs = [e for e in experiments if e["delta_f1"] < 0]
    neutral = [e for e in experiments if abs(e["delta_f1"]) < 0.001]

    print(f"\n  Benefits:  {len(benefits)}")
    for b in benefits:
        print(f"    {b['experiment']}: ΔF1={b['delta_f1']:+.4f}  thr={b['adaptive_threshold']:.3f}")

    print(f"\n  Costs:     {len(costs)}")
    for c in costs:
        print(f"    {c['experiment']}: ΔF1={c['delta_f1']:+.4f}  thr={c['adaptive_threshold']:.3f}")

    print(f"\n  Neutral:   {len(neutral)}")

    # Three prerequisites framework
    print(f"\n  ── Three Prerequisites for Adaptive Threshold Benefit ──")
    print("    1. Significant domain shift (source ≠ target distribution)")
    print("    2. Sufficient in-domain model quality (F1 > 0.85)")
    print("    3. Moderate threshold deviation from 0.5 (not too extreme)")

    # ── Plot scatter (ΔF1 vs threshold) ──
    if experiments:
        fig, ax = plt.subplots(figsize=(8, 5.5))

        # Filter out duplicate rerun entries (same data as originals)
        SKIP_DUPLICATES = {"exp_A_rerun", "exp_E_rerun"}
        plot_exps = [e for e in experiments if e["experiment"] not in SKIP_DUPLICATES]

        # Clean labels for publication
        LABELS = {
            "A": "Exp A\nCIC→NB15",
            "B": "Exp B\nCIC→CIC2018",
            "E": "Exp E\nNB15→CIC",
            "F": "Exp F\nCIC2018→CIC",
        }
        # Offset positions to avoid label overlap
        OFFSETS = {
            "A": (8, -18),
            "B": (8, -5),
            "E": (8, 8),
            "F": (8, 8),
        }

        for exp in plot_exps:
            name = exp["experiment"]
            delta = exp["delta_f1"]
            color = "#2E7D32" if delta > 0 else "#C62828"
            marker = "^" if delta > 0 else "v"
            ax.scatter(
                exp["adaptive_threshold"], delta,
                color=color, s=140, zorder=5, marker=marker,
                edgecolors="white", linewidths=1.0
            )
            label = LABELS.get(name, name)
            offset = OFFSETS.get(name, (5, 5))
            ax.annotate(
                label,
                (exp["adaptive_threshold"], delta),
                textcoords="offset points", xytext=offset, fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8),
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4),
            )

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.3, label="Default (0.5)")

        # Shade benefit/cost regions
        xlim = ax.get_xlim()
        ax.axhspan(0, ax.get_ylim()[1], alpha=0.03, color="#4CAF50")
        ax.axhspan(ax.get_ylim()[0], 0, alpha=0.03, color="#F44336")

        ax.set_xlabel("GMM Adaptive Threshold", fontsize=12)
        ax.set_ylabel("ΔF1 (Adaptive − Default)", fontsize=12)
        ax.set_title("Adaptive Threshold Benefit–Cost Analysis", fontsize=13)
        ax.grid(True, alpha=0.2)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#2E7D32",
                   markersize=10, label="Benefit (ΔF1 > 0)"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#C62828",
                   markersize=10, label="Cost (ΔF1 < 0)"),
            Line2D([0], [0], color="gray", linestyle=":", alpha=0.5,
                   label="Default threshold (0.5)"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

        plt.tight_layout()
        fig_path = os.path.join(PROJECT_ROOT, args.fig_out)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n  ✓ Figure saved: {fig_path}")

    # ── Save ──
    output = {
        "n_experiments": len(experiments),
        "n_benefits": len(benefits),
        "n_costs": len(costs),
        "experiments": experiments,
        "framework": {
            "name": "Three Prerequisites for Adaptive Threshold",
            "prerequisites": [
                "Significant domain shift",
                "Sufficient in-domain model quality (F1 > 0.85)",
                "Moderate threshold deviation from 0.5",
            ],
        },
    }

    out_path = os.path.join(PROJECT_ROOT, args.out)
    save_json(output, out_path)

    print(f"\n{'═' * 70}")
    print(f"  E8 COMPLETE")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

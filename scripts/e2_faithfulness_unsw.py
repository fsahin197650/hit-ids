#!/usr/bin/env python3
"""
E2 — Faithfulness evaluation on the second dataset (UNSW-NB15).

Replicates the P2 faithfulness protocol (5 metrics × 3 attribution methods)
on the UNSW-NB15 binary model. This enables the claim:
    "Across two datasets, attention is significantly faithful vs random."

Prerequisite artefacts:
    - results/unsw_nb15/checkpoints/hit_ids_binary_unsw.pt
    - data/unsw_nb15/splits/test.pt  (or uses full aligned data)

Output:
    results/unsw_nb15/tables/faithfulness_nb15.json
    results/unsw_nb15/tables/faithfulness_nb15_per_sample.npz

Usage:
    python scripts/e2_faithfulness_unsw.py
    python scripts/e2_faithfulness_unsw.py --n-samples 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common.helpers import (
    DEVICE, PROJECT_ROOT, SEED, UNSW_CONFIG_PATH, UNSW_BINARY_CKPT,
    load_model, load_config, get_group_info, set_seed, save_json, ensure_dir,
    compute_attention_importance, compute_gradient_importance,
    compute_random_importance,
)
from src.xai.faithfulness import ComprehensiveFaithfulness


def load_unsw_test_data() -> tuple:
    """Load UNSW-NB15 test data.

    Tries CV fold-0 test split first, then full aligned data.
    """
    # Try fold-specific test split
    fold_test = os.path.join(PROJECT_ROOT, "data/unsw_nb15/splits/test.pt")
    if os.path.exists(fold_test):
        data = torch.load(fold_test, weights_only=False)
        X = data["X"]
        y = data.get("y", data.get("y_bin"))
        if y is None:
            # Binary: label column
            y = data.get("labels")
        print(f"  Loaded NB15 test split: {X.shape}")
        return X, y

    # Fallback: use full NB15 training data test portion
    train_path = os.path.join(PROJECT_ROOT, "data/unsw_nb15/splits/train.pt")
    if os.path.exists(train_path):
        # Use a held-out portion of train data for faithfulness
        data = torch.load(train_path, weights_only=False)
        X = data["X"]
        y = data.get("y_bin", data.get("y"))
        # Take last 20% as "test"
        n_test = int(len(X) * 0.2)
        X = X[-n_test:]
        y = y[-n_test:]
        print(f"  Using last 20% of NB15 train as test: {X.shape}")
        return X, y

    raise FileNotFoundError(
        "No UNSW-NB15 test data found. Expected data/unsw_nb15/splits/test.pt"
    )


def paired_test(metric_name, vals_a, vals_b, direction="greater"):
    """Paired t-test for per-sample values."""
    t_stat, p_two = stats.ttest_rel(vals_a, vals_b)
    if direction == "greater":
        p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:
        p = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    return {
        "metric": metric_name,
        "a_mean": float(np.mean(vals_a)),
        "b_mean": float(np.mean(vals_b)),
        "delta": float(np.mean(vals_a) - np.mean(vals_b)),
        "t_statistic": float(t_stat),
        "p_value": float(p),
        "significant_005": bool(p < 0.05),
        "significant_001": bool(p < 0.001),
    }


def main():
    parser = argparse.ArgumentParser(
        description="E2: Faithfulness evaluation on UNSW-NB15"
    )
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of correctly-classified samples to evaluate")
    parser.add_argument("--steps", type=int, default=20,
                        help="Steps for deletion/insertion curves")
    parser.add_argument("--out-dir", type=str,
                        default="results/unsw_nb15/tables")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 70)
    print("  E2: FAITHFULNESS ON UNSW-NB15 (Second Dataset)")
    print(f"  Samples: {args.n_samples}  |  Steps: {args.steps}  |  Device: {DEVICE}")
    print("=" * 70)

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)

    # ── Load model ──
    config = load_config(UNSW_CONFIG_PATH)
    group_names, group_sizes = get_group_info(config)
    model = load_model(UNSW_CONFIG_PATH, UNSW_BINARY_CKPT, num_classes=2)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Load data ──
    X_all, y_all = load_unsw_test_data()

    # For binary model, y should be 0/1
    if y_all.max() > 1:
        y_all = (y_all > 0).long()

    # Subsample
    N = min(args.n_samples * 2, len(X_all))  # oversample, then filter correct
    idx = np.random.choice(len(X_all), N, replace=False)
    X_eval = X_all[idx]
    y_eval = y_all[idx]

    # Keep only correctly classified
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_eval), 512):
            logits = model(X_eval[i:i+512].to(DEVICE), return_explanations=False)
            preds.append(logits.argmax(1).cpu())
        preds = torch.cat(preds)

    correct_mask = preds == y_eval
    X_eval = X_eval[correct_mask]
    y_eval = y_eval[correct_mask]

    # Trim to target sample count
    if len(X_eval) > args.n_samples:
        X_eval = X_eval[:args.n_samples]
        y_eval = y_eval[:args.n_samples]

    print(f"  Using {len(X_eval)} correctly classified samples")

    # ── Compute importances ──
    print("\n  Computing attention importance...")
    t0 = time.time()
    attn_imp = compute_attention_importance(model, X_eval, group_names)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("  Computing gradient importance...")
    t0 = time.time()
    grad_imp = compute_gradient_importance(model, X_eval, y_eval)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("  Computing random importance...")
    random_imp = compute_random_importance(len(X_eval), X_eval.shape[1])

    # ── Faithfulness metrics ──
    evaluator = ComprehensiveFaithfulness(model, device=DEVICE, masking_value=0.0)
    k_values = [5, 10, 15, 20]

    methods = {
        "attention": attn_imp,
        "gradient": grad_imp,
        "random": random_imp,
    }

    all_results = {}
    for method_name, importances in methods.items():
        print(f"\n{'─' * 50}")
        print(f"  Method: {method_name}")
        print(f"{'─' * 50}")
        result = evaluator.evaluate_all(
            X_eval, importances, y_eval,
            k_values=k_values, steps=args.steps
        )
        all_results[method_name] = result

        print(f"    Deletion AUC:   {result['deletion_auc']['mean']:.4f}")
        print(f"    Insertion AUC:  {result['insertion_auc']['mean']:.4f}")
        print(f"    AOPC:           {result['aopc']['mean']:.4f}")
        for k in k_values:
            print(f"    Comp(k={k:2d}):     {result[f'comprehensiveness_k{k}']['mean']:.4f}")
            print(f"    Suff(k={k:2d}):     {result[f'sufficiency_k{k}']['mean']:.4f}")

    # ── Statistical tests ──
    print(f"\n{'═' * 70}")
    print("  STATISTICAL TESTS")
    print(f"{'═' * 70}")

    attn = all_results["attention"]
    grad = all_results["gradient"]
    rand = all_results["random"]

    test_configs = [
        ("deletion_auc", "less",    "Deletion AUC ↓"),
        ("insertion_auc", "greater", "Insertion AUC ↑"),
        ("aopc",          "greater", "AOPC ↑"),
    ]
    for k in k_values:
        test_configs.append((f"comprehensiveness_k{k}", "greater", f"Comp(k={k}) ↑"))
        test_configs.append((f"sufficiency_k{k}",       "less",    f"Suff(k={k}) ↓"))

    attn_vs_rand_tests = []
    attn_vs_grad_tests = []

    for metric_key, direction, desc in test_configs:
        a_vals = np.array(attn[metric_key]["per_sample"])
        r_vals = np.array(rand[metric_key]["per_sample"])
        g_vals = np.array(grad[metric_key]["per_sample"])

        t_ar = paired_test(metric_key, a_vals, r_vals, direction)
        t_ag = paired_test(metric_key, a_vals, g_vals, direction)

        attn_vs_rand_tests.append(t_ar)
        attn_vs_grad_tests.append(t_ag)

        ar_sig = "✓" if t_ar["significant_001"] else ("~" if t_ar["significant_005"] else "✗")
        ag_sig = "✓" if t_ag["significant_001"] else ("~" if t_ag["significant_005"] else "✗")
        print(f"  {desc:18s}: attn={t_ar['a_mean']:.4f}  "
              f"rand={t_ar['b_mean']:.4f} p={t_ar['p_value']:.4f}{ar_sig}  "
              f"grad={t_ag['b_mean']:.4f} p={t_ag['p_value']:.4f}{ag_sig}")

    n_sig_ar = sum(1 for t in attn_vs_rand_tests if t["significant_001"])
    n_sig_ag = sum(1 for t in attn_vs_grad_tests if t["significant_005"])
    n_total = len(attn_vs_rand_tests)

    print(f"\n  Attention vs Random (p<0.001): {n_sig_ar}/{n_total}")
    print(f"  Attention vs Gradient (p<0.05): {n_sig_ag}/{n_total}")

    # ── Save summary ──
    summary_results = {}
    for method_name, result in all_results.items():
        summary_results[method_name] = {}
        for key, val in result.items():
            if isinstance(val, dict) and "per_sample" in val:
                summary_results[method_name][key] = {
                    "mean": val["mean"], "std": val["std"],
                }
            else:
                summary_results[method_name][key] = val

    output = {
        "dataset": "UNSW-NB15",
        "model": "HiT-IDS binary (F1=0.935)",
        "n_samples": len(X_eval),
        "n_features": int(X_eval.shape[1]),
        "steps": args.steps,
        "k_values": k_values,
        "results": summary_results,
        "attn_vs_random_tests": attn_vs_rand_tests,
        "attn_vs_gradient_tests": attn_vs_grad_tests,
        "n_sig_attn_vs_random_001": n_sig_ar,
        "n_sig_attn_vs_gradient_005": n_sig_ag,
        "scenario": "A" if n_sig_ar >= 3 else "B",
    }

    save_json(output, os.path.join(out_dir, "faithfulness_nb15.json"))

    # Save per-sample for downstream E3 stats hardening
    per_sample_data = {}
    for method_name, result in all_results.items():
        for key, val in result.items():
            if isinstance(val, dict) and "per_sample" in val:
                per_sample_data[f"{method_name}_{key}"] = np.array(val["per_sample"])

    npz_path = os.path.join(out_dir, "faithfulness_nb15_per_sample.npz")
    np.savez(npz_path, **per_sample_data)
    print(f"  ✓ Per-sample data: {npz_path}")

    print(f"\n{'═' * 70}")
    print(f"  E2 COMPLETE — Scenario {'A' if n_sig_ar >= 3 else 'B'}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

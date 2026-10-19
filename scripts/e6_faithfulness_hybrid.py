#!/usr/bin/env python3
"""
E6 — Hybrid attribution (attention × gradient) faithfulness evaluation.

Investigates the 3 metrics where gradient outperforms attention
(Insertion AUC, Suff k=5, Suff k=10) and tests whether a simple
hybrid attribution (attention × |gradient|) can close the gap.

Prerequisite artefacts:
    - CIC-IDS2017 attack checkpoint
    - data/splits/test.pt
    - results/tables/faithfulness_per_sample.npz (for comparison)

Output:
    results/tables/faithfulness_hybrid.json

Usage:
    python scripts/e6_faithfulness_hybrid.py
    python scripts/e6_faithfulness_hybrid.py --n-samples 500
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
    DEVICE, PROJECT_ROOT, SEED,
    CIC_CONFIG_PATH,
    load_config, get_group_info, set_seed, save_json, ensure_dir,
    compute_attention_importance, compute_gradient_importance,
    compute_random_importance,
)
from src.models.hit_ids import HiTIDS
from src.xai.faithfulness import ComprehensiveFaithfulness


def compute_hybrid_importance(
    attn_imp: np.ndarray, grad_imp: np.ndarray, mode: str = "product"
) -> np.ndarray:
    """
    Compute hybrid attribution.

    Modes:
        - 'product':    normalize(attention) × normalize(|gradient|)
        - 'geometric':  sqrt(normalize(attention) × normalize(|gradient|))
        - 'weighted':   0.6 × attention + 0.4 × gradient (tunable)
    """
    # Already normalized; re-normalize to ensure
    a = attn_imp / (attn_imp.sum(axis=1, keepdims=True) + 1e-12)
    g = grad_imp / (grad_imp.sum(axis=1, keepdims=True) + 1e-12)

    if mode == "product":
        hybrid = a * g
    elif mode == "geometric":
        hybrid = np.sqrt(a * g + 1e-12)
    elif mode == "weighted":
        hybrid = 0.6 * a + 0.4 * g
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Re-normalize
    row_sums = hybrid.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return hybrid / row_sums


def paired_test(vals_a, vals_b, direction):
    """One-sided paired t-test."""
    t_stat, p_two = stats.ttest_rel(vals_a, vals_b)
    if direction == "greater":
        p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:
        p = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    return float(p), float(t_stat)


def main():
    parser = argparse.ArgumentParser(
        description="E6: Hybrid attribution (attention × gradient) faithfulness"
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", type=str,
                        default="results/tables/faithfulness_hybrid.json")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 70)
    print("  E6: HYBRID ATTRIBUTION FAITHFULNESS")
    print(f"  Samples: {args.n_samples}  |  Steps: {args.steps}  |  Device: {DEVICE}")
    print("=" * 70)

    config = load_config(CIC_CONFIG_PATH)
    group_names, group_sizes = get_group_info(config)
    mcfg = config["model"]

    # ── Load attack model ──
    ckpt_candidates = [
        os.path.join(PROJECT_ROOT, "results/checkpoints/hit_ids_attack_merged_v2.pt"),
        os.path.join(PROJECT_ROOT, "results/checkpoints/hit_ids_attack_merged.pt"),
        os.path.join(PROJECT_ROOT, "results/checkpoints/hit_ids_attack.pt"),
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break
    if not ckpt_path:
        raise FileNotFoundError("No attack checkpoint found")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    num_classes = ckpt["model_state_dict"]["classification_head.head.3.bias"].shape[0]

    model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=mcfg["d_emb"], nhead=mcfg["nhead"],
        intra_layers=mcfg["intra_group_layers"],
        inter_layers=mcfg["inter_group_layers"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=0.0, num_classes=num_classes,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model loaded: {num_classes} classes")

    # ── Load test data ──
    test_data = torch.load(
        os.path.join(PROJECT_ROOT, "data/splits/test.pt"), weights_only=False
    )
    atk_mask = test_data["y"] > 0
    X_atk = test_data["X"][atk_mask]
    y_atk = test_data["y"][atk_mask] - 1

    N = min(args.n_samples * 2, len(X_atk))
    idx = np.random.choice(len(X_atk), N, replace=False)
    X_eval = X_atk[idx]
    y_eval = y_atk[idx]

    # Correct predictions only
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_eval), 512):
            logits = model(X_eval[i:i+512].to(DEVICE), return_explanations=False)
            preds.append(logits.argmax(1).cpu())
        preds = torch.cat(preds)

    correct_mask = preds == y_eval
    X_eval = X_eval[correct_mask][:args.n_samples]
    y_eval = y_eval[correct_mask][:args.n_samples]
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

    # ── Compute hybrid importances ──
    hybrid_modes = ["product", "geometric", "weighted"]
    hybrid_imps = {}
    for mode in hybrid_modes:
        print(f"  Computing hybrid ({mode}) importance...")
        hybrid_imps[mode] = compute_hybrid_importance(attn_imp, grad_imp, mode)

    # ── Evaluate all methods ──
    evaluator = ComprehensiveFaithfulness(model, device=DEVICE, masking_value=0.0)
    k_values = [5, 10, 15, 20]

    methods = {
        "attention": attn_imp,
        "gradient": grad_imp,
        "random": random_imp,
    }
    for mode in hybrid_modes:
        methods[f"hybrid_{mode}"] = hybrid_imps[mode]

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

        print(f"    Del AUC: {result['deletion_auc']['mean']:.4f}  "
              f"Ins AUC: {result['insertion_auc']['mean']:.4f}  "
              f"AOPC: {result['aopc']['mean']:.4f}")
        for k in k_values:
            print(f"    Comp(k={k:2d}): {result[f'comprehensiveness_k{k}']['mean']:.4f}  "
                  f"Suff(k={k:2d}): {result[f'sufficiency_k{k}']['mean']:.4f}")

    # ── Compare: which method wins on which metric ──
    print(f"\n{'═' * 70}")
    print("  METHOD COMPARISON (per metric)")
    print(f"{'═' * 70}")

    metric_keys = ["deletion_auc", "insertion_auc", "aopc"]
    metric_dirs = ["less",         "greater",       "greater"]
    for k in k_values:
        metric_keys.append(f"comprehensiveness_k{k}")
        metric_dirs.append("greater")
        metric_keys.append(f"sufficiency_k{k}")
        metric_dirs.append("less")

    comparison_table = []
    for mk, md in zip(metric_keys, metric_dirs):
        row = {"metric": mk, "direction": md}
        best_val = None
        best_method = None

        for method_name, result in all_results.items():
            val = result[mk]["mean"]
            row[method_name] = val

            if best_val is None:
                best_val = val
                best_method = method_name
            elif md == "greater" and val > best_val:
                best_val = val
                best_method = method_name
            elif md == "less" and val < best_val:
                best_val = val
                best_method = method_name

        row["winner"] = best_method
        comparison_table.append(row)
        print(f"  {mk:25s}: winner = {best_method}")

    # Count wins per method
    win_counts = {}
    for row in comparison_table:
        w = row["winner"]
        win_counts[w] = win_counts.get(w, 0) + 1

    print(f"\n  Win counts:")
    for m, c in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"    {m}: {c}/{len(comparison_table)}")

    # ── Specific focus: does hybrid close the gradient gap? ──
    # Gradient wins on: Insertion AUC, Suff(k=5), Suff(k=10)
    gap_metrics = ["insertion_auc", "sufficiency_k5", "sufficiency_k10"]
    print(f"\n  ── GRADIENT-GAP ANALYSIS ──")
    for mk in gap_metrics:
        if mk not in all_results["attention"]:
            continue
        a = all_results["attention"][mk]["mean"]
        g = all_results["gradient"][mk]["mean"]
        direction = "less" if "suff" in mk else "greater"

        best_hybrid_name = None
        best_hybrid_val = None
        for mode in hybrid_modes:
            hk = f"hybrid_{mode}"
            h = all_results[hk][mk]["mean"]
            if best_hybrid_val is None:
                best_hybrid_val = h
                best_hybrid_name = hk
            elif direction == "greater" and h > best_hybrid_val:
                best_hybrid_val = h
                best_hybrid_name = hk
            elif direction == "less" and h < best_hybrid_val:
                best_hybrid_val = h
                best_hybrid_name = hk

        if direction == "greater":
            gap_closed = best_hybrid_val >= g
        else:
            gap_closed = best_hybrid_val <= g

        print(f"  {mk}: attn={a:.4f}  grad={g:.4f}  "
              f"best_hybrid({best_hybrid_name})={best_hybrid_val:.4f}  "
              f"{'CLOSED ✓' if gap_closed else 'OPEN ✗'}")

    # ── Save ──
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
        "n_samples": len(X_eval),
        "n_features": int(X_eval.shape[1]),
        "hybrid_modes": hybrid_modes,
        "results": summary_results,
        "comparison_table": comparison_table,
        "win_counts": win_counts,
    }

    out_path = os.path.join(PROJECT_ROOT, args.out)
    save_json(output, out_path)

    print(f"\n{'═' * 70}")
    print(f"  E6 COMPLETE")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

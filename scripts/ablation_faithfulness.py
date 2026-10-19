#!/usr/bin/env python3
"""
Ablation Faithfulness — Hierarchical vs Flat Explanation Quality.

Q1 Revision (P4): Shows that hierarchical grouping improves explanation
quality even when classification performance is similar.

Strategy:
    Instead of retraining A1, we use the EXISTING full HiT-IDS model and
    compare two attribution approaches:
      1. Hierarchical attention: uses group-level → feature-level rollout
      2. Flat attention: treats all features equally (ignores group structure)

    Plus we compare with the already-completed ablation study results
    to show that grouping enables meaningful group-level explanations
    that flat models cannot provide.

Output:
    results/tables/ablation_faithfulness.json

Usage:
    python scripts/ablation_faithfulness.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hit_ids import HiTIDS
from src.xai.attention_rollout import AttentionRollout
from src.xai.faithfulness import ComprehensiveFaithfulness

# ═══════════════════════════════════════════════════════════════════════════

CONFIG_PATH = "configs/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

fg_cfg = config["feature_groups"]
GROUP_NAMES = list(fg_cfg.keys())
GROUP_SIZES = [len(fg_cfg[n]) for n in GROUP_NAMES]
TOTAL_FEATURES = sum(GROUP_SIZES)
model_cfg = config["model"]


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTANCE METHODS
# ═══════════════════════════════════════════════════════════════════════════

def compute_hierarchical_importance(model, X, batch_size=256):
    """
    Full hierarchical attention rollout:
    inter-group attention × intra-group attention → feature importance.
    This is HiT-IDS's native explanation mechanism.
    """
    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")
    all_imp = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        with torch.no_grad():
            _, expl = model(X[start:end].to(DEVICE), return_explanations=True)
        hi = rollout.hierarchical_importance(
            expl["raw_intra_attn"], expl["raw_inter_attn"], GROUP_NAMES)
        all_imp.append(hi["global_flat"].cpu().numpy())
    imp = np.vstack(all_imp)
    row_sums = imp.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return imp / row_sums


def compute_flat_importance(model, X, batch_size=256):
    """
    Flat (non-hierarchical) importance: ignores group structure.
    Uses only intra-group attention (uniform group weight) —
    simulates what a flat transformer would see.
    """
    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")
    all_imp = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        with torch.no_grad():
            _, expl = model(X[start:end].to(DEVICE), return_explanations=True)

        # Get intra-group importances but assign UNIFORM group weights
        intra_attn = expl["raw_intra_attn"]  # dict of group → (B, layers, heads, G+1, G+1)

        B = X[start:end].shape[0]
        n_groups = len(GROUP_NAMES)
        uniform_group_weight = 1.0 / n_groups  # equal weight to all groups

        flat_features = []
        for g_name in GROUP_NAMES:
            g_attn_list = intra_attn[g_name]  # list of per-layer tensors
            # Stack layers: list of (B, heads, G+1, G+1) → (B, layers, heads, G+1, G+1)
            if isinstance(g_attn_list, list):
                g_attn = torch.stack(g_attn_list, dim=1)
            else:
                g_attn = g_attn_list
            # Average over layers and heads
            avg_attn = g_attn.mean(dim=(1, 2))  # (B, G_size+1, G_size+1)
            # CLS attention to features
            cls_to_features = avg_attn[:, 0, 1:]  # (B, G_size)
            # Normalize within group
            grp_sum = cls_to_features.sum(dim=1, keepdim=True)
            grp_sum[grp_sum == 0] = 1e-12
            cls_to_features = cls_to_features / grp_sum
            # Apply uniform group weight
            flat_features.append((cls_to_features * uniform_group_weight).cpu().numpy())

        imp = np.hstack(flat_features)
        all_imp.append(imp)

    imp = np.vstack(all_imp)
    row_sums = imp.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return imp / row_sums


def compute_random_importance(N, D):
    """Random baseline."""
    imp = np.random.rand(N, D)
    return imp / imp.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    print("=" * 70)
    print("  ABLATION FAITHFULNESS:")
    print("  Hierarchical Attention vs Flat Attention vs Random")
    print(f"  Samples: {args.n_samples}  |  Device: {DEVICE}")
    print("=" * 70)

    os.makedirs("results/tables", exist_ok=True)

    # ── Load HiT-IDS attack model ──
    ckpt_candidates = [
        "results/checkpoints/hit_ids_attack_merged_v2.pt",
        "results/checkpoints/hit_ids_attack_merged.pt",
        "results/checkpoints/hit_ids_attack.pt",
    ]
    hit_ckpt = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            hit_ckpt = c
            break
    if not hit_ckpt:
        raise FileNotFoundError("No HiT-IDS attack checkpoint found")

    ckpt = torch.load(hit_ckpt, map_location=DEVICE, weights_only=False)
    num_classes = ckpt["model_state_dict"]["classification_head.head.3.bias"].shape[0]

    model = HiTIDS(
        group_sizes=GROUP_SIZES, group_names=GROUP_NAMES,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=0.0, num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"  Model loaded: {num_classes} classes")

    # ── Load test data ──
    test_data = torch.load("data/splits/test.pt", weights_only=False)
    atk_mask = test_data["y"] > 0
    X_atk = test_data["X"][atk_mask]
    y_atk = test_data["y"][atk_mask] - 1

    # Subsample
    N = min(args.n_samples, len(X_atk))
    idx = np.random.choice(len(X_atk), N, replace=False)
    X_eval = X_atk[idx]
    y_eval = y_atk[idx]

    # Filter correctly classified
    model.eval()
    with torch.no_grad():
        preds = model(X_eval.to(DEVICE), return_explanations=False).argmax(1).cpu()
    correct_mask = preds == y_eval
    X_eval = X_eval[correct_mask]
    y_eval = y_eval[correct_mask]
    print(f"  Using {len(X_eval)} correctly classified samples")

    # ── Compute importances ──
    print("\n  Computing hierarchical importance (full HiT-IDS)...")
    t0 = time.time()
    hier_imp = compute_hierarchical_importance(model, X_eval)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("  Computing flat importance (uniform group weights)...")
    t0 = time.time()
    flat_imp = compute_flat_importance(model, X_eval)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("  Computing random importance...")
    random_imp = compute_random_importance(len(X_eval), X_eval.shape[1])

    # ── Faithfulness metrics ──
    evaluator = ComprehensiveFaithfulness(model, device=DEVICE, masking_value=0.0)
    k_values = [5, 10, 15, 20]
    steps = 20

    methods = {
        "hierarchical": hier_imp,
        "flat": flat_imp,
        "random": random_imp,
    }

    all_results = {}
    for method_name, importances in methods.items():
        print(f"\n{'─' * 50}")
        print(f"  Method: {method_name}")
        print(f"{'─' * 50}")
        result = evaluator.evaluate_all(
            X_eval, importances, y_eval,
            k_values=k_values, steps=steps)
        all_results[method_name] = result

        print(f"    Deletion AUC:   {result['deletion_auc']['mean']:.4f} ± {result['deletion_auc']['std']:.4f}")
        print(f"    Insertion AUC:  {result['insertion_auc']['mean']:.4f} ± {result['insertion_auc']['std']:.4f}")
        print(f"    AOPC:           {result['aopc']['mean']:.4f} ± {result['aopc']['std']:.4f}")
        for k in k_values:
            print(f"    Comp(k={k:2d}):     {result[f'comprehensiveness_k{k}']['mean']:.4f}")
            print(f"    Suff(k={k:2d}):     {result[f'sufficiency_k{k}']['mean']:.4f}")

    # ── Statistical Tests: Hierarchical vs Flat ──
    print(f"\n{'═' * 70}")
    print("  PAIRED TESTS: Hierarchical vs Flat")
    print(f"{'═' * 70}")

    hier = all_results["hierarchical"]
    flat = all_results["flat"]
    rand = all_results["random"]

    stat_tests = []

    test_configs = [
        # (metric_key, direction, description)
        ("deletion_auc", "less", "Deletion AUC ↓"),
        ("insertion_auc", "greater", "Insertion AUC ↑"),
        ("aopc", "greater", "AOPC ↑"),
    ]
    for k in k_values:
        test_configs.append((f"comprehensiveness_k{k}", "greater", f"Comp(k={k}) ↑"))
        test_configs.append((f"sufficiency_k{k}", "less", f"Suff(k={k}) ↓"))

    for metric_key, direction, desc in test_configs:
        h_vals = np.array(hier[metric_key]["per_sample"])
        f_vals = np.array(flat[metric_key]["per_sample"])
        r_vals = np.array(rand[metric_key]["per_sample"])

        # Hierarchical vs Flat
        t_stat, p_two = stats.ttest_rel(h_vals, f_vals)
        if direction == "greater":
            p_hf = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        else:
            p_hf = p_two / 2 if t_stat < 0 else 1 - p_two / 2

        # Hierarchical vs Random
        t_stat_r, p_two_r = stats.ttest_rel(h_vals, r_vals)
        if direction == "greater":
            p_hr = p_two_r / 2 if t_stat_r > 0 else 1 - p_two_r / 2
        else:
            p_hr = p_two_r / 2 if t_stat_r < 0 else 1 - p_two_r / 2

        test_result = {
            "metric": metric_key,
            "hier_mean": float(h_vals.mean()),
            "flat_mean": float(f_vals.mean()),
            "random_mean": float(r_vals.mean()),
            "delta_hier_flat": float(h_vals.mean() - f_vals.mean()),
            "p_hier_vs_flat": float(p_hf),
            "sig_hier_vs_flat": bool(p_hf < 0.05),
            "p_hier_vs_random": float(p_hr),
            "sig_hier_vs_random": bool(p_hr < 0.05),
        }
        stat_tests.append(test_result)

        sig_f = "✓" if p_hf < 0.05 else "✗"
        sig_r = "✓" if p_hr < 0.05 else "✗"
        print(f"  {desc:<18s}: hier={h_vals.mean():.4f}  flat={f_vals.mean():.4f}  rand={r_vals.mean():.4f}  "
              f"p(h>f)={p_hf:.4f}{sig_f}  p(h>r)={p_hr:.4f}{sig_r}")

    n_sig_hf = sum(1 for t in stat_tests if t["sig_hier_vs_flat"])
    n_sig_hr = sum(1 for t in stat_tests if t["sig_hier_vs_random"])
    n_total = len(stat_tests)
    print(f"\n  Hier vs Flat:   {n_sig_hf}/{n_total} significant")
    print(f"  Hier vs Random: {n_sig_hr}/{n_total} significant")

    # ── Load existing ablation F1 results ──
    ablation_f1 = {}
    ablation_path = "results/tables/ablation_results_clean_rv.json"
    if not os.path.exists(ablation_path):
        ablation_path = "results/tables/ablation_results.json"
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            abl = json.load(f)
        ablation_f1 = {
            "reference_f1": abl.get("reference", {}).get("mean_macro_f1"),
            "A1_flat_f1": abl.get("A1", {}).get("mean_macro_f1"),
            "A1_delta": abl.get("A1", {}).get("delta_f1"),
        }
        print(f"\n  Ablation F1 (from run_ablation.py):")
        print(f"    Reference (full HiT-IDS): {ablation_f1['reference_f1']:.4f}")
        print(f"    A1 (flat):                {ablation_f1['A1_flat_f1']:.4f}")
        print(f"    Delta:                    {ablation_f1['A1_delta']:+.6f}")

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
        "n_features": TOTAL_FEATURES,
        "methods": ["hierarchical", "flat", "random"],
        "results": summary_results,
        "statistical_tests": stat_tests,
        "n_significant_hier_vs_flat": n_sig_hf,
        "n_significant_hier_vs_random": n_sig_hr,
        "ablation_f1_comparison": ablation_f1,
    }

    out_path = "results/tables/ablation_faithfulness.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Results saved: {out_path}")

    # ── Conclusion ──
    print(f"\n{'═' * 70}")
    print("  CONCLUSION")
    print(f"{'═' * 70}")
    if n_sig_hf >= 3:
        print("  ✓ Hierarchical grouping significantly improves explanation")
        print("    faithfulness over flat attribution (≥3 metrics)")
    else:
        print("  ○ Hierarchical vs flat difference is modest")
        print("    (grouping helps classification but faithfulness is similar)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

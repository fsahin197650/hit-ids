#!/usr/bin/env python3
"""
Comprehensive Faithfulness Evaluation — Q1 Revision (P2).

Compares three attribution methods on CIC-IDS2017 attack model:
  1. Attention Rollout (intrinsic — HiT-IDS's built-in)
  2. Input Gradient (gradient-based baseline)
  3. Random (null baseline)

Metrics: Deletion AUC, Insertion AUC, Comprehensiveness, Sufficiency, AOPC
Statistical tests: Paired t-test for attention vs random (per-sample)

Output:
    results/tables/faithfulness_results.json

Usage:
    python scripts/run_faithfulness.py
    python scripts/run_faithfulness.py --n-samples 200   # quick test
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
# CONFIG
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
model_cfg = config["model"]


# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL & DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_attack_model(explicit=None):
    """Load the best CIC-IDS2017 attack model."""
    # Try v2 merged first, then fallback
    ckpt_candidates = ([explicit] if explicit else []) + [
        "results/checkpoints/hit_ids_attack_merged_v2.pt",
        "results/checkpoints/hit_ids_attack_merged.pt",
        "results/checkpoints/hit_ids_attack_v2.pt",
        "results/checkpoints/hit_ids_attack.pt",
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No attack checkpoint found in {ckpt_candidates}")

    print(f"  Loading model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
    print(f"  Model loaded: {num_classes} classes, {sum(p.numel() for p in model.parameters()):,} params")
    return model, num_classes


def load_test_data():
    """Load CIC-IDS2017 attack test samples."""
    test_data = torch.load("data/splits/test.pt", weights_only=False)
    X_test = test_data["X"]
    y_test = test_data["y"]

    # Filter attack only
    atk_mask = y_test > 0
    X_atk = X_test[atk_mask]
    y_atk = y_test[atk_mask] - 1  # 0-indexed for attack model

    print(f"  Test data: {len(X_test)} total, {len(X_atk)} attack samples")
    return X_atk, y_atk


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_attention_importance(model, X, batch_size=256):
    """Compute attention rollout importance for all samples."""
    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")
    all_imp = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = X[start:end].to(DEVICE)
        with torch.no_grad():
            _, expl = model(xb, return_explanations=True)
        hi = rollout.hierarchical_importance(
            expl["raw_intra_attn"], expl["raw_inter_attn"], GROUP_NAMES)
        imp = hi["global_flat"].cpu().numpy()
        all_imp.append(imp)

    imp_all = np.vstack(all_imp)
    # Normalize per sample
    row_sums = imp_all.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return imp_all / row_sums


def compute_gradient_importance(model, X, y, batch_size=128):
    """Compute input gradient (|∂L/∂x|) importance for all samples."""
    all_imp = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = X[start:end].clone().to(DEVICE).requires_grad_(True)
        yb = y[start:end].to(DEVICE)

        logits = model(xb, return_explanations=False)
        # Gradient of correct class logit w.r.t. input
        scores = logits.gather(1, yb.unsqueeze(1)).sum()
        scores.backward()

        imp = xb.grad.abs().cpu().numpy()
        all_imp.append(imp)

        model.zero_grad()

    imp_all = np.vstack(all_imp)
    row_sums = imp_all.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return imp_all / row_sums


def compute_shap_importance(model, X, n_bg=100, seed=42):
    """GradientSHAP |value| of the predicted class, normalised per sample."""
    from src.xai.shap_analysis import SHAPAnalyzer
    tr = torch.load("data/splits/train.pt", weights_only=False)
    ytr = tr["y"].numpy()
    rng = np.random.RandomState(seed)
    bg = tr["X"][rng.choice(np.where(ytr > 0)[0], n_bg, replace=False)]
    analyzer = SHAPAnalyzer(model, bg, device=DEVICE, method="gradient")
    sv = analyzer.compute_shap_values(X.to(DEVICE), max_samples=None)
    if sv.ndim == 2:
        sv = sv[:, :, None]
    with torch.no_grad():
        yhat = model(X.to(DEVICE), return_explanations=False).argmax(1).cpu().numpy()
    imp = np.abs(sv[np.arange(len(yhat)), :, yhat])
    row = imp.sum(axis=1, keepdims=True)
    row[row == 0] = 1e-12
    return imp / row


def compute_random_importance(N, D):
    """Random baseline importance."""
    imp = np.random.rand(N, D)
    return imp / imp.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def paired_test(metric_name, attn_vals, random_vals, direction="greater"):
    """
    Paired t-test for per-sample faithfulness values.
    direction: "greater" (attn > random) or "less" (attn < random)
    """
    t_stat, p_two = stats.ttest_rel(attn_vals, random_vals)
    if direction == "greater":
        p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:  # less
        p = p_two / 2 if t_stat < 0 else 1 - p_two / 2

    delta = np.mean(attn_vals) - np.mean(random_vals)
    return {
        "metric": metric_name,
        "attn_mean": float(np.mean(attn_vals)),
        "random_mean": float(np.mean(random_vals)),
        "delta": float(delta),
        "t_statistic": float(t_stat),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "direction": direction,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Faithfulness Evaluation")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--steps", type=int, default=20,
                        help="Steps for deletion/insertion curves")
    parser.add_argument("--attack-ckpt", default=None, help="explicit attack checkpoint")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. _phase1")
    parser.add_argument("--with-shap", action="store_true", help="also evaluate GradientSHAP attribution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("  COMPREHENSIVE FAITHFULNESS EVALUATION")
    print(f"  Samples: {args.n_samples}  |  Steps: {args.steps}  |  Device: {DEVICE}")
    print("=" * 70)

    os.makedirs("results/tables", exist_ok=True)

    # Load
    model, num_classes = load_attack_model(args.attack_ckpt)
    X_atk, y_atk = load_test_data()

    # Subsample
    N = min(args.n_samples, len(X_atk))
    idx = np.random.choice(len(X_atk), N, replace=False)
    X_eval = X_atk[idx]
    y_eval = y_atk[idx]

    # Verify predictions are correct
    model.eval()
    with torch.no_grad():
        logits = model(X_eval.to(DEVICE), return_explanations=False)
        preds = logits.argmax(1).cpu()
    correct = (preds == y_eval).float().mean().item()
    print(f"\n  Model accuracy on eval subset: {correct:.4f}")

    # Use only correctly classified samples for faithfulness
    correct_mask = preds == y_eval
    X_eval = X_eval[correct_mask]
    y_eval = y_eval[correct_mask]
    print(f"  Using {len(X_eval)} correctly classified samples")

    # ── Compute Importances ──
    print("\n  Computing attention importance...")
    t0 = time.time()
    attn_imp = compute_attention_importance(model, X_eval)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("  Computing gradient importance...")
    t0 = time.time()
    grad_imp = compute_gradient_importance(model, X_eval, y_eval)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("  Computing random importance...")
    random_imp = compute_random_importance(len(X_eval), X_eval.shape[1])

    # ── Evaluate All Methods ──
    evaluator = ComprehensiveFaithfulness(model, device=DEVICE, masking_value=0.0)
    k_values = [5, 10, 15, 20]

    methods = {
        "attention": attn_imp,
        "gradient": grad_imp,
        "random": random_imp,
    }
    if args.with_shap:
        print("  Computing GradientSHAP importance...")
        t0 = time.time()
        methods["shap"] = compute_shap_importance(model, X_eval)
        print(f"    Done ({time.time()-t0:.1f}s)")

    all_results = {}
    for method_name, importances in methods.items():
        print(f"\n{'─' * 50}")
        print(f"  Method: {method_name}")
        print(f"{'─' * 50}")
        result = evaluator.evaluate_all(
            X_eval, importances, y_eval,
            k_values=k_values, steps=args.steps)
        all_results[method_name] = result

        print(f"    Deletion AUC:   {result['deletion_auc']['mean']:.4f} ± {result['deletion_auc']['std']:.4f}")
        print(f"    Insertion AUC:  {result['insertion_auc']['mean']:.4f} ± {result['insertion_auc']['std']:.4f}")
        print(f"    AOPC:           {result['aopc']['mean']:.4f} ± {result['aopc']['std']:.4f}")
        for k in k_values:
            print(f"    Comp(k={k:2d}):     {result[f'comprehensiveness_k{k}']['mean']:.4f}")
            print(f"    Suff(k={k:2d}):     {result[f'sufficiency_k{k}']['mean']:.4f}")

    # ── Statistical Tests ──
    print(f"\n{'═' * 70}")
    print("  STATISTICAL TESTS (Attention vs Random)")
    print(f"{'═' * 70}")

    stat_tests = []
    attn = all_results["attention"]
    rand = all_results["random"]

    # Deletion AUC: lower = better → attention should be LESS
    test = paired_test("deletion_auc",
                       attn["deletion_auc"]["per_sample"],
                       rand["deletion_auc"]["per_sample"],
                       direction="less")
    stat_tests.append(test)
    print(f"  Deletion AUC:  attn={test['attn_mean']:.4f}  rand={test['random_mean']:.4f}  p={test['p_value']:.4f}  {'✓' if test['significant'] else '✗'}")

    # Insertion AUC: higher = better → attention should be GREATER
    test = paired_test("insertion_auc",
                       attn["insertion_auc"]["per_sample"],
                       rand["insertion_auc"]["per_sample"],
                       direction="greater")
    stat_tests.append(test)
    print(f"  Insertion AUC: attn={test['attn_mean']:.4f}  rand={test['random_mean']:.4f}  p={test['p_value']:.4f}  {'✓' if test['significant'] else '✗'}")

    # AOPC: higher = better
    test = paired_test("aopc",
                       attn["aopc"]["per_sample"],
                       rand["aopc"]["per_sample"],
                       direction="greater")
    stat_tests.append(test)
    print(f"  AOPC:          attn={test['attn_mean']:.4f}  rand={test['random_mean']:.4f}  p={test['p_value']:.4f}  {'✓' if test['significant'] else '✗'}")

    # Comprehensiveness: higher = better
    for k in k_values:
        key = f"comprehensiveness_k{k}"
        test = paired_test(key,
                           attn[key]["per_sample"],
                           rand[key]["per_sample"],
                           direction="greater")
        stat_tests.append(test)
        print(f"  Comp(k={k:2d}):    attn={test['attn_mean']:.4f}  rand={test['random_mean']:.4f}  p={test['p_value']:.4f}  {'✓' if test['significant'] else '✗'}")

    # Sufficiency: lower = better → attention should be LESS
    for k in k_values:
        key = f"sufficiency_k{k}"
        test = paired_test(key,
                           attn[key]["per_sample"],
                           rand[key]["per_sample"],
                           direction="less")
        stat_tests.append(test)
        print(f"  Suff(k={k:2d}):    attn={test['attn_mean']:.4f}  rand={test['random_mean']:.4f}  p={test['p_value']:.4f}  {'✓' if test['significant'] else '✗'}")

    # Count significant
    n_sig = sum(1 for t in stat_tests if t["significant"])
    n_total = len(stat_tests)
    print(f"\n  ════════════════════════════════════")
    print(f"  SIGNIFICANT: {n_sig}/{n_total} metrics")
    if n_sig >= 3:
        print(f"  → SCENARIO A: Strong faithfulness (≥3 significant)")
    else:
        print(f"  → SCENARIO B: Weak/mixed faithfulness (<3 significant)")
    print(f"  ════════════════════════════════════")

    # ── Save ──
    # Strip per_sample for smaller JSON (keep in separate file)
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
        "steps": args.steps,
        "k_values": k_values,
        "results": summary_results,
        "statistical_tests": stat_tests,
        "scenario": "A" if n_sig >= 3 else "B",
    }

    out_path = f"results/tables/faithfulness_results{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Summary saved: {out_path}")

    # Save per-sample for downstream paired tests
    per_sample_path = f"results/tables/faithfulness_per_sample{args.tag}.npz"
    per_sample_data = {}
    for method_name, result in all_results.items():
        for key, val in result.items():
            if isinstance(val, dict) and "per_sample" in val:
                per_sample_data[f"{method_name}_{key}"] = np.array(val["per_sample"])
    np.savez(per_sample_path, **per_sample_data)
    print(f"  ✓ Per-sample data saved: {per_sample_path}")

    print("\n" + "=" * 70)
    print("  FAITHFULNESS EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

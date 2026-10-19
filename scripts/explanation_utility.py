#!/usr/bin/env python3
"""
Explanation Utility — Attention-Guided vs Random Feature Ablation.

Q1 Revision (P1-utility): Demonstrates that HiT-IDS attention identifies
decision-relevant features by showing that masking attention-highlighted
features degrades performance MORE than masking random features.

Protocol:
    For each k in [5, 10, 15, 20]:
        - Mask top-k attention features → measure F1 drop
        - Mask k random features (10 trials) → measure F1 drop
        - Paired t-test: attention-guided drop > random drop?

Output:
    results/tables/explanation_utility.json

Usage:
    python scripts/explanation_utility.py
    python scripts/explanation_utility.py --n-samples 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from scipy import stats
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hit_ids import HiTIDS
from src.xai.attention_rollout import AttentionRollout

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


def load_model_and_data():
    """Load attack model and test data."""
    ckpt_candidates = [
        "results/checkpoints/hit_ids_attack_merged_v2.pt",
        "results/checkpoints/hit_ids_attack_merged.pt",
        "results/checkpoints/hit_ids_attack.pt",
    ]
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break
    else:
        raise FileNotFoundError("No attack checkpoint found")

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

    test_data = torch.load("data/splits/test.pt", weights_only=False)
    atk_mask = test_data["y"] > 0
    X = test_data["X"][atk_mask]
    y = test_data["y"][atk_mask] - 1
    return model, X, y, num_classes


def predict_f1(model, X, y, batch_size=512):
    """Get macro-F1 for a batch."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            logits = model(X[start:end].to(DEVICE), return_explanations=False)
            preds.append(logits.argmax(1).cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = y.numpy() if isinstance(y, torch.Tensor) else y
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_attention_importance(model, X, batch_size=256):
    """Get per-sample attention importance."""
    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")
    all_imp = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        with torch.no_grad():
            _, expl = model(X[start:end].to(DEVICE), return_explanations=True)
        hi = rollout.hierarchical_importance(
            expl["raw_intra_attn"], expl["raw_inter_attn"], GROUP_NAMES)
        all_imp.append(hi["global_flat"].cpu().numpy())
    return np.vstack(all_imp)


def main():
    parser = argparse.ArgumentParser(description="Explanation Utility Test")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-random-trials", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPLANATION UTILITY: Attention-Guided vs Random Ablation")
    print(f"  Samples: {args.n_samples}  |  Random trials: {args.n_random_trials}")
    print("=" * 70)

    model, X_all, y_all, num_classes = load_model_and_data()
    N = min(args.n_samples, len(X_all))
    idx = np.random.choice(len(X_all), N, replace=False)
    X = X_all[idx]
    y = y_all[idx]

    D = X.shape[1]
    k_values = [5, 10, 15, 20]

    # Baseline F1 (no masking)
    base_f1 = predict_f1(model, X, y)
    print(f"\n  Baseline F1 (no masking): {base_f1:.4f}")

    # Compute attention importance
    print("  Computing attention importance...")
    attn_imp = compute_attention_importance(model, X)
    # Global importance (mean across samples)
    global_imp = attn_imp.mean(axis=0)
    global_rank = np.argsort(global_imp)[::-1].copy()

    results = {}

    for k in k_values:
        print(f"\n  ── k = {k} ──")

        # Attention-guided: mask top-k features (by global importance)
        X_attn_masked = X.clone()
        X_attn_masked[:, global_rank[:k]] = 0.0
        attn_f1 = predict_f1(model, X_attn_masked, y)
        attn_drop = base_f1 - attn_f1

        # Per-sample attention masking (for paired test)
        per_sample_attn_ranks = np.argsort(-attn_imp, axis=1)
        X_attn_ps = X.clone()
        for i in range(len(X)):
            X_attn_ps[i, per_sample_attn_ranks[i, :k]] = 0.0
        attn_ps_f1 = predict_f1(model, X_attn_ps, y)
        attn_ps_drop = base_f1 - attn_ps_f1

        # Random: mask k random features (multiple trials)
        random_f1s = []
        for trial in range(args.n_random_trials):
            rng = np.random.RandomState(SEED + trial)
            random_features = rng.choice(D, k, replace=False)
            X_rand = X.clone()
            X_rand[:, random_features] = 0.0
            rf1 = predict_f1(model, X_rand, y)
            random_f1s.append(rf1)

        random_f1_mean = np.mean(random_f1s)
        random_f1_std = np.std(random_f1s)
        random_drop_mean = base_f1 - random_f1_mean

        # Statistical test
        random_drops = [base_f1 - rf for rf in random_f1s]
        # One-sample t-test: is attn_drop significantly > mean(random_drops)?
        t_stat, p_val = stats.ttest_1samp(random_drops, attn_drop)
        p_one = p_val / 2 if t_stat < 0 else 1 - p_val / 2  # one-sided: attn > random

        sig = "✓" if p_one < 0.05 else "✗"
        print(f"    Attention (global) F1: {attn_f1:.4f}  (drop: {attn_drop:+.4f})")
        print(f"    Attention (per-sample) F1: {attn_ps_f1:.4f}  (drop: {attn_ps_drop:+.4f})")
        print(f"    Random F1:  {random_f1_mean:.4f} ± {random_f1_std:.4f}  (drop: {random_drop_mean:+.4f})")
        print(f"    Δ(drop):    {attn_drop - random_drop_mean:+.4f}  p={p_one:.4f} {sig}")

        results[f"k={k}"] = {
            "base_f1": float(base_f1),
            "attn_f1_global": float(attn_f1),
            "attn_drop_global": float(attn_drop),
            "attn_f1_per_sample": float(attn_ps_f1),
            "attn_drop_per_sample": float(attn_ps_drop),
            "random_f1_mean": float(random_f1_mean),
            "random_f1_std": float(random_f1_std),
            "random_drop_mean": float(random_drop_mean),
            "delta_drop": float(attn_drop - random_drop_mean),
            "p_value": float(p_one),
            "significant": bool(p_one < 0.05),
            "random_per_trial": [float(f) for f in random_f1s],
        }

    # Save
    output = {
        "n_samples": N,
        "n_features": D,
        "n_random_trials": args.n_random_trials,
        "baseline_f1": float(base_f1),
        "results": results,
    }
    os.makedirs("results/tables", exist_ok=True)
    out_path = "results/tables/explanation_utility.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Results saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
03_stats.py  —  Paired statistical tests vs HiT-IDS.

Reads a result JSON + HiT-IDS per-fold macro-F1 values,
computes (matching paper Section 4.6/5.2):
    - One-sided paired t-test p-value
    - Cohen's d
    - 95% CI of the mean macro-F1 difference across 5 paired folds

Usage:
    python revision_baselines/03_stats.py --baseline revision_results/results_deepids.json
    python revision_baselines/03_stats.py --baseline revision_results/results_tranad.json --metric binary_f1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

# ── Pipeline hooks ────────────────────────────────────────────────────────
from pipeline_hooks import (
    get_hit_ids_per_fold_f1,
    RESULTS_DIR,
    save_json,
)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for paired data."""
    nx, ny = len(x), len(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else 0.0


def effect_label(d: float) -> str:
    """Human-readable effect size label."""
    d_abs = abs(d)
    if d_abs >= 0.8:
        return "large"
    elif d_abs >= 0.5:
        return "medium"
    elif d_abs >= 0.2:
        return "small"
    else:
        return "negligible"


def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical tests vs HiT-IDS"
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Path to baseline results JSON (e.g. revision_results/results_deepids.json)"
    )
    parser.add_argument(
        "--metric", default="macro_f1",
        help="Metric to compare (default: macro_f1; for TranAD use binary_f1)"
    )
    parser.add_argument(
        "--hit-f1", nargs=5, type=float, default=None,
        help="Override HiT-IDS per-fold F1 (5 values)"
    )
    args = parser.parse_args()

    # ── Load baseline results ────────────────────────────────────────────
    if not os.path.exists(args.baseline):
        print(f"  ❌ File not found: {args.baseline}")
        print("     Run the baseline experiment first.")
        sys.exit(1)

    with open(args.baseline) as f:
        baseline_data = json.load(f)

    baseline_name = baseline_data.get("model", os.path.basename(args.baseline))

    # Extract per-fold metric
    metric = args.metric
    if "per_fold" in baseline_data:
        per_fold = baseline_data["per_fold"]
        if isinstance(per_fold, list):
            if isinstance(per_fold[0], dict):
                bl_f1s = [r[metric] for r in per_fold]
            else:
                bl_f1s = per_fold  # already a list of floats
        else:
            print(f"  ❌ Cannot parse per_fold from {args.baseline}")
            sys.exit(1)
    elif "aggregate" in baseline_data and "per_fold" in baseline_data:
        bl_f1s = [r[metric] for r in baseline_data["per_fold"]]
    else:
        # Try to extract from nested structure
        print(f"  ❌ Cannot find per-fold results in {args.baseline}")
        print(f"     Expected key 'per_fold' with list of dicts containing '{metric}'")
        sys.exit(1)

    # ── HiT-IDS reference ───────────────────────────────────────────────
    if args.hit_f1:
        hit_f1s = np.array(args.hit_f1)
    elif args.metric == "binary_f1":
        # Compare like with like: HiT-IDS attack-vs-benign F1 per fold
        import json as _json
        from pipeline_hooks import CLEAN_CV_PATH
        with open(CLEAN_CV_PATH) as _f:
            _cv = _json.load(_f)
        hit_f1s = np.array([r["binary"]["f1"] for r in _cv["per_fold_results"]["HiT-IDS (Cascade)"]])
    else:
        hit_f1s = np.array(get_hit_ids_per_fold_f1())
    bl_f1s = np.array(bl_f1s)

    K = len(hit_f1s)
    assert len(bl_f1s) == K, (
        f"Fold count mismatch: HiT-IDS has {K} folds, "
        f"baseline has {len(bl_f1s)}"
    )

    # ── Paired statistics ────────────────────────────────────────────────
    delta = hit_f1s - bl_f1s  # positive = HiT-IDS is better

    # One-sided paired t-test (H_a: HiT-IDS > baseline)
    t_stat, p_two = stats.ttest_rel(hit_f1s, bl_f1s)
    # Convert to one-sided p-value
    if delta.mean() > 0:
        p_one = p_two / 2  # HiT-IDS > baseline
    else:
        p_one = 1.0 - p_two / 2  # baseline is better

    # Cohen's d
    d = cohens_d(hit_f1s, bl_f1s)

    # 95% CI of the mean difference
    se = delta.std(ddof=1) / np.sqrt(K)
    t_crit = stats.t.ppf(0.975, df=K - 1)
    ci_lo = delta.mean() - t_crit * se
    ci_hi = delta.mean() + t_crit * se

    # ── Results ──────────────────────────────────────────────────────────
    result = {
        "comparison": f"HiT-IDS vs {baseline_name}",
        "metric": metric,
        "k_folds": K,
        "hit_ids_per_fold": hit_f1s.tolist(),
        "baseline_per_fold": bl_f1s.tolist(),
        "hit_ids_mean": float(hit_f1s.mean()),
        "hit_ids_std": float(hit_f1s.std()),
        "baseline_mean": float(bl_f1s.mean()),
        "baseline_std": float(bl_f1s.std()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std(ddof=1)),
        "delta_95ci": [float(ci_lo), float(ci_hi)],
        "paired_t_test": {
            "t_statistic": float(t_stat),
            "p_value_two_sided": float(p_two),
            "p_value_one_sided": float(p_one),
            "significant_at_005": bool(p_one < 0.05),
        },
        "cohens_d": float(d),
        "effect_size": effect_label(d),
    }

    # Determine output filename from baseline name
    bl_tag = baseline_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    from pipeline_hooks import CV_TAG
    out_file = f"stats_{bl_tag}_{CV_TAG}.json"
    save_json(result, out_file)

    # ── Print summary ────────────────────────────────────────────────────
    direction = "BETTER" if delta.mean() > 0 else "WORSE"
    sig = "✓ (p<0.05)" if p_one < 0.05 else "✗ (ns)"

    print("\n" + "=" * 70)
    print(f"  PAIRED STATISTICS: HiT-IDS vs {baseline_name}")
    print("=" * 70)
    print(f"  Metric: {metric}")
    print(f"  HiT-IDS:  {hit_f1s.mean():.4f} ± {hit_f1s.std():.4f}")
    print(f"  Baseline: {bl_f1s.mean():.4f} ± {bl_f1s.std():.4f}")
    print(f"  Δ{metric}: {delta.mean():+.4f} (HiT-IDS {direction})")
    print(f"  95% CI:   [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Cohen's d: {d:.3f} ({effect_label(d)})")
    print(f"  Paired t-test (one-sided): t={t_stat:.3f}, p={p_one:.6f} {sig}")
    print("=" * 70)


if __name__ == "__main__":
    main()

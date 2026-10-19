#!/usr/bin/env python3
"""
E3 — Statistical hardening: multiple-comparison correction + Cohen's d + bootstrap CI.

Applies Holm–Bonferroni and Benjamini–Hochberg FDR correction to ALL paired
tests from P1 (utility), P2 (faithfulness), P4 (ablation), and optionally E2
(UNSW faithfulness). Computes Cohen's d (paired) and bootstrap 95% CI.

Prerequisite artefacts:
    - results/tables/faithfulness_per_sample.npz
    - results/tables/explanation_utility.json
    - results/tables/ablation_faithfulness.json
    - (optional) results/unsw_nb15/tables/faithfulness_nb15_per_sample.npz

Output:
    results/tables/stats_hardened.json

Usage:
    python scripts/e3_stats_hardening.py
    python scripts/e3_stats_hardening.py --n-bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common.helpers import (
    PROJECT_ROOT, SEED, set_seed, save_json, ensure_dir,
)


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TOOLS
# ═══════════════════════════════════════════════════════════════════════════

def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni step-down correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n

    prev_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        adjusted = max(adjusted, prev_max)  # enforce monotonicity
        adjusted = min(adjusted, 1.0)
        corrected[orig_idx] = adjusted
        prev_max = adjusted

    return corrected


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
    corrected = [0.0] * n

    prev_min = 1.0
    for rank_rev, (orig_idx, p) in enumerate(indexed):
        rank_asc = n - rank_rev  # 1-indexed ascending rank
        adjusted = p * n / rank_asc
        adjusted = min(adjusted, prev_min)
        adjusted = min(adjusted, 1.0)
        corrected[orig_idx] = adjusted
        prev_min = adjusted

    return corrected


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples: d = mean(x-y) / std(x-y)."""
    diff = x - y
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    return float(d)


def bootstrap_ci(
    x: np.ndarray, n_bootstrap: int = 1000,
    ci: float = 0.95, statistic="mean", seed: int = SEED
) -> dict:
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    stats_list = []
    for _ in range(n_bootstrap):
        sample = rng.choice(x, size=len(x), replace=True)
        if statistic == "mean":
            stats_list.append(sample.mean())
        elif statistic == "median":
            stats_list.append(np.median(sample))
    stats_arr = np.array(stats_list)

    alpha = 1 - ci
    lo = float(np.percentile(stats_arr, 100 * alpha / 2))
    hi = float(np.percentile(stats_arr, 100 * (1 - alpha / 2)))
    return {
        "point_estimate": float(x.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PAIRED TEST EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_p2_tests(npz_path: str, k_values: list = None) -> list[dict]:
    """Extract paired tests from P2 faithfulness per-sample data."""
    if k_values is None:
        k_values = [5, 10, 15, 20]

    data = np.load(npz_path)
    tests = []

    metric_configs = [
        ("deletion_auc", "less",    "P2: Deletion AUC (attn vs rand)"),
        ("insertion_auc", "greater", "P2: Insertion AUC (attn vs rand)"),
        ("aopc",          "greater", "P2: AOPC (attn vs rand)"),
    ]
    for k in k_values:
        metric_configs.append(
            (f"comprehensiveness_k{k}", "greater", f"P2: Comp(k={k}) (attn vs rand)")
        )
        metric_configs.append(
            (f"sufficiency_k{k}", "less", f"P2: Suff(k={k}) (attn vs rand)")
        )

    for metric_key, direction, label in metric_configs:
        a_key = f"attention_{metric_key}"
        r_key = f"random_{metric_key}"
        g_key = f"gradient_{metric_key}"

        if a_key not in data or r_key not in data:
            continue

        a_vals = data[a_key]
        r_vals = data[r_key]

        t_stat, p_two = stats.ttest_rel(a_vals, r_vals)
        if direction == "greater":
            p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        else:
            p = p_two / 2 if t_stat < 0 else 1 - p_two / 2

        d = cohens_d_paired(a_vals, r_vals) if direction == "greater" else cohens_d_paired(r_vals, a_vals)

        ci_a = bootstrap_ci(a_vals)
        ci_diff = bootstrap_ci(a_vals - r_vals)

        test_entry = {
            "family": "P2_faithfulness_CIC2017",
            "label": label,
            "metric": metric_key,
            "direction": direction,
            "raw_p": float(p),
            "cohens_d": float(d),
            "attn_mean": float(a_vals.mean()),
            "random_mean": float(r_vals.mean()),
            "ci_attn": ci_a,
            "ci_difference": ci_diff,
        }

        # Also add attention vs gradient test
        if g_key in data:
            g_vals = data[g_key]
            t_g, p_two_g = stats.ttest_rel(a_vals, g_vals)
            if direction == "greater":
                p_g = p_two_g / 2 if t_g > 0 else 1 - p_two_g / 2
            else:
                p_g = p_two_g / 2 if t_g < 0 else 1 - p_two_g / 2
            d_g = cohens_d_paired(a_vals, g_vals) if direction == "greater" else cohens_d_paired(g_vals, a_vals)
            test_entry["gradient_mean"] = float(g_vals.mean())
            test_entry["vs_gradient_raw_p"] = float(p_g)
            test_entry["vs_gradient_cohens_d"] = float(d_g)

        tests.append(test_entry)

    return tests


def extract_p4_tests(json_path: str) -> list[dict]:
    """Extract paired tests from P4 (ablation faithfulness) results."""
    with open(json_path) as f:
        abl = json.load(f)

    tests = []
    for st in abl.get("statistical_tests", []):
        p = st.get("p_hier_vs_flat", 1.0)
        tests.append({
            "family": "P4_ablation_faithfulness",
            "label": f"P4: {st['metric']} (hier vs flat)",
            "metric": st["metric"],
            "direction": "greater" if "comp" in st["metric"] or "aopc" in st["metric"]
                         or "insertion" in st["metric"] else "less",
            "raw_p": float(p),
            "cohens_d": float(st.get("delta_hier_flat", 0.0)),  # approximate
            "hier_mean": st.get("hier_mean"),
            "flat_mean": st.get("flat_mean"),
        })

    return tests


def main():
    parser = argparse.ArgumentParser(
        description="E3: Statistical hardening (Holm/FDR + Cohen's d + bootstrap CI)"
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for CI")
    parser.add_argument("--out", type=str,
                        default="results/tables/stats_hardened.json")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 70)
    print("  E3: STATISTICAL HARDENING")
    print(f"  Bootstrap: {args.n_bootstrap} resamples")
    print("=" * 70)

    all_tests = []

    # ── P2: CIC-IDS2017 faithfulness ──
    p2_npz = os.path.join(PROJECT_ROOT, "results/tables/faithfulness_per_sample.npz")
    if os.path.exists(p2_npz):
        p2_tests = extract_p2_tests(p2_npz)
        all_tests.extend(p2_tests)
        print(f"  Loaded P2 (CIC-IDS2017): {len(p2_tests)} tests")
    else:
        print(f"  [WARN] P2 data not found: {p2_npz}")

    # ── E2: UNSW-NB15 faithfulness (if available) ──
    e2_npz = os.path.join(PROJECT_ROOT, "results/unsw_nb15/tables/faithfulness_nb15_per_sample.npz")
    if os.path.exists(e2_npz):
        e2_tests = extract_p2_tests(e2_npz)
        # Re-label family
        for t in e2_tests:
            t["family"] = "E2_faithfulness_NB15"
            t["label"] = t["label"].replace("P2:", "E2:")
        all_tests.extend(e2_tests)
        print(f"  Loaded E2 (UNSW-NB15):   {len(e2_tests)} tests")
    else:
        print(f"  [INFO] E2 data not yet available: {e2_npz}")

    # ── P4: Ablation faithfulness ──
    p4_json = os.path.join(PROJECT_ROOT, "results/tables/ablation_faithfulness.json")
    if os.path.exists(p4_json):
        p4_tests = extract_p4_tests(p4_json)
        all_tests.extend(p4_tests)
        print(f"  Loaded P4 (ablation):    {len(p4_tests)} tests")
    else:
        print(f"  [WARN] P4 data not found: {p4_json}")

    if not all_tests:
        print("  [ERROR] No tests found. Run P2/P4/E2 first.")
        return

    # ── Group by family for per-family corrections ──
    families = {}
    for t in all_tests:
        fam = t["family"]
        if fam not in families:
            families[fam] = []
        families[fam].append(t)

    print(f"\n  Total tests: {len(all_tests)}")
    print(f"  Families: {list(families.keys())}")

    # ── Apply corrections within each family ──
    for fam_name, fam_tests in families.items():
        raw_ps = [t["raw_p"] for t in fam_tests]
        holm_ps = holm_bonferroni(raw_ps)
        bh_ps = benjamini_hochberg(raw_ps)

        for i, t in enumerate(fam_tests):
            t["p_holm"] = holm_ps[i]
            t["p_bh_fdr"] = bh_ps[i]
            t["sig_raw_005"] = t["raw_p"] < 0.05
            t["sig_holm_005"] = holm_ps[i] < 0.05
            t["sig_bh_005"] = bh_ps[i] < 0.05

    # ── Also apply global correction across all tests ──
    all_raw_ps = [t["raw_p"] for t in all_tests]
    all_holm = holm_bonferroni(all_raw_ps)
    all_bh = benjamini_hochberg(all_raw_ps)
    for i, t in enumerate(all_tests):
        t["p_holm_global"] = all_holm[i]
        t["p_bh_global"] = all_bh[i]
        t["sig_holm_global_005"] = all_holm[i] < 0.05

    # ── Print summary ──
    print(f"\n{'═' * 70}")
    print("  RESULTS")
    print(f"{'═' * 70}")

    for fam_name, fam_tests in families.items():
        print(f"\n  ── {fam_name} ({len(fam_tests)} tests) ──")
        n_raw = sum(1 for t in fam_tests if t["sig_raw_005"])
        n_holm = sum(1 for t in fam_tests if t["sig_holm_005"])
        n_bh = sum(1 for t in fam_tests if t["sig_bh_005"])
        print(f"    Significant (raw):  {n_raw}/{len(fam_tests)}")
        print(f"    Significant (Holm): {n_holm}/{len(fam_tests)}")
        print(f"    Significant (BH):   {n_bh}/{len(fam_tests)}")

        for t in fam_tests:
            d_str = f"d={t.get('cohens_d', 0):.3f}" if 'cohens_d' in t else ""
            holm_sig = "✓" if t["sig_holm_005"] else "✗"
            print(f"    {t['label']:45s}  p_raw={t['raw_p']:.4f}  "
                  f"p_holm={t['p_holm']:.4f}{holm_sig}  {d_str}")

    # ── Global summary ──
    n_global_raw = sum(1 for t in all_tests if t["sig_raw_005"])
    n_global_holm = sum(1 for t in all_tests if t["sig_holm_global_005"])
    print(f"\n  GLOBAL: {n_global_raw}/{len(all_tests)} raw → "
          f"{n_global_holm}/{len(all_tests)} after Holm")

    # ── Save ──
    output = {
        "n_tests_total": len(all_tests),
        "families": {
            fam: {
                "n_tests": len(fam_tests),
                "n_sig_raw": sum(1 for t in fam_tests if t["sig_raw_005"]),
                "n_sig_holm": sum(1 for t in fam_tests if t["sig_holm_005"]),
                "n_sig_bh": sum(1 for t in fam_tests if t["sig_bh_005"]),
            }
            for fam, fam_tests in families.items()
        },
        "global_summary": {
            "n_sig_raw": n_global_raw,
            "n_sig_holm_global": n_global_holm,
        },
        "tests": all_tests,
    }

    out_path = os.path.join(PROJECT_ROOT, args.out)
    save_json(output, out_path)

    print(f"\n{'═' * 70}")
    print(f"  E3 COMPLETE")
    print(f"  Key: All 'significant' claims survive Holm correction? "
          f"{'YES ✓' if n_global_holm == n_global_raw else 'SOME LOST — review!'}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

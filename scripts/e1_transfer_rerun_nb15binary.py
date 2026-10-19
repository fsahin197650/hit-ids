#!/usr/bin/env python3
"""
E1 — Cross-dataset transfer re-run with new strong NB15 binary model.

Re-runs Exp A (CIC2017→NB15) and Exp E (NB15→CIC2017) from Section 5.10
using the new binary NB15 model (F1=0.935) instead of the old weak 10-class
model (in-domain F1≈0.45). This resolves the inconsistency flagged in the
experiment plan.

Prerequisite artefacts:
    - results/unsw_nb15/checkpoints/hit_ids_binary_unsw.pt   (new NB15 binary)
    - results/checkpoints/hit_ids_binary.pt                   (CIC-IDS2017 binary)
    - data/unsw_nb15/aligned_for_2017_model.pt                (aligned NB15 data)
    - data/splits/aligned_for_nb15_model.pt                   (aligned CIC data)

Output:
    results/cross_dataset/transfer_rerun_nb15binary.json

Usage:
    python scripts/e1_transfer_rerun_nb15binary.py
    python scripts/e1_transfer_rerun_nb15binary.py --batch-size 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common.helpers import (
    DEVICE, PROJECT_ROOT, SEED,
    CIC_CONFIG_PATH, CIC_BINARY_CKPT,
    UNSW_CONFIG_PATH, UNSW_BINARY_CKPT,
    ALIGNED_NB15_FOR_CIC_MODEL, ALIGNED_CIC_FOR_NB15_MODEL,
    load_model, set_seed, save_json, ensure_dir,
)


def find_adaptive_threshold(probs: np.ndarray) -> float:
    """GMM-based adaptive threshold (same as evaluate_cross_dataset.py)."""
    probs_2d = probs.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(probs_2d)

    means = gmm.means_.flatten()
    lo_idx = int(np.argmin(means))
    hi_idx = int(np.argmax(means))
    stds = np.sqrt(gmm.covariances_.flatten())
    threshold = (means[lo_idx] * stds[hi_idx] + means[hi_idx] * stds[lo_idx]) / (
        stds[lo_idx] + stds[hi_idx]
    )
    return float(np.clip(threshold, 0.01, 0.95))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_binary": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "report": classification_report(
            y_true, y_pred, target_names=["Normal", "Attack"],
            digits=4, zero_division=0
        ),
    }


def evaluate_binary(model, X, y_bin, batch_size=512):
    """Evaluate with both default (0.5) and adaptive GMM threshold."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            logits = model(xb, return_explanations=False)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs)
    y_true = y_bin.numpy() if isinstance(y_bin, torch.Tensor) else y_bin

    # Default threshold (argmax / 0.5)
    y_default = (all_probs >= 0.5).astype(int)
    default_m = compute_metrics(y_true, y_default)

    # Adaptive GMM threshold
    adaptive_thr = find_adaptive_threshold(all_probs)
    y_adaptive = (all_probs >= adaptive_thr).astype(int)
    adaptive_m = compute_metrics(y_true, y_adaptive)

    return {
        "default": default_m,
        "adaptive": adaptive_m,
        "adaptive_threshold": adaptive_thr,
        "prob_stats": {
            "mean": float(all_probs.mean()),
            "max": float(all_probs.max()),
            "p95": float(np.percentile(all_probs, 95)),
        },
    }


def run_exp_A_rerun(batch_size: int) -> dict:
    """Exp A rerun: CIC-IDS2017 model → NB15 test data (same as before)."""
    print("\n" + "=" * 60)
    print("  Exp A (re-run): CIC-IDS2017 → UNSW-NB15 (Binary)")
    print("=" * 60)

    if not os.path.exists(ALIGNED_NB15_FOR_CIC_MODEL):
        print(f"  [ERROR] Aligned data not found: {ALIGNED_NB15_FOR_CIC_MODEL}")
        print("  Run: python scripts/cross_dataset_align.py --direction nb15_to_cic2017")
        return None

    model = load_model(CIC_CONFIG_PATH, CIC_BINARY_CKPT, num_classes=2)
    data = torch.load(ALIGNED_NB15_FOR_CIC_MODEL, weights_only=False)
    result = evaluate_binary(model, data["X"], data["y_bin"], batch_size)
    result["experiment"] = "A_rerun"
    result["direction"] = "CIC-IDS2017 → UNSW-NB15"
    result["source_model"] = "CIC-IDS2017 binary"
    result["test_samples"] = len(data["X"])

    d = result["default"]
    a = result["adaptive"]
    print(f"\n  Default (thr=0.50):  F1m={d['f1_macro']:.4f}  Acc={d['accuracy']:.4f}")
    print(f"  Adaptive (thr={result['adaptive_threshold']:.4f}): "
          f"F1m={a['f1_macro']:.4f}  Acc={a['accuracy']:.4f}")
    return result


def run_exp_E_rerun(batch_size: int) -> dict:
    """Exp E rerun: NEW NB15 binary model → CIC-IDS2017 test data."""
    print("\n" + "=" * 60)
    print("  Exp E (re-run): UNSW-NB15 (NEW binary) → CIC-IDS2017")
    print("=" * 60)

    if not os.path.exists(ALIGNED_CIC_FOR_NB15_MODEL):
        print(f"  [ERROR] Aligned data not found: {ALIGNED_CIC_FOR_NB15_MODEL}")
        print("  Run: python scripts/cross_dataset_align.py --direction cic2017_to_nb15")
        return None

    if not os.path.exists(UNSW_BINARY_CKPT):
        print(f"  [ERROR] NB15 binary model not found: {UNSW_BINARY_CKPT}")
        return None

    model = load_model(UNSW_CONFIG_PATH, UNSW_BINARY_CKPT, num_classes=2)
    data = torch.load(ALIGNED_CIC_FOR_NB15_MODEL, weights_only=False)
    result = evaluate_binary(model, data["X"], data["y_bin"], batch_size)
    result["experiment"] = "E_rerun"
    result["direction"] = "UNSW-NB15 (new binary) → CIC-IDS2017"
    result["source_model"] = "UNSW-NB15 binary (F1=0.935)"
    result["test_samples"] = len(data["X"])

    d = result["default"]
    a = result["adaptive"]
    print(f"\n  Default (thr=0.50):  F1m={d['f1_macro']:.4f}  Acc={d['accuracy']:.4f}")
    print(f"  Adaptive (thr={result['adaptive_threshold']:.4f}): "
          f"F1m={a['f1_macro']:.4f}  Acc={a['accuracy']:.4f}")
    return result


def compute_retention(indomain_f1: float, transfer_f1: float) -> float:
    """Retention ratio = transfer_f1 / indomain_f1."""
    if indomain_f1 == 0:
        return 0.0
    return transfer_f1 / indomain_f1


def main():
    parser = argparse.ArgumentParser(
        description="E1: Re-run cross-dataset transfer with new NB15 binary model"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=str,
                        default="results/cross_dataset/transfer_rerun_nb15binary.json")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 60)
    print("  E1: CROSS-DATASET TRANSFER RE-RUN (New NB15 Binary)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    results = {}
    t0 = time.time()

    # Exp A: CIC→NB15 (re-run for completeness)
    res_a = run_exp_A_rerun(args.batch_size)
    if res_a:
        results["exp_A_rerun"] = res_a

    # Exp E: NB15→CIC (the KEY re-run with new model)
    res_e = run_exp_E_rerun(args.batch_size)
    if res_e:
        results["exp_E_rerun"] = res_e

    # Load old results for comparison
    old_exp_e_path = os.path.join(PROJECT_ROOT, "results/cross_dataset/exp_E_nb15_to_cic2017.json")
    if os.path.exists(old_exp_e_path):
        with open(old_exp_e_path) as f:
            old_e = json.load(f)
        results["old_exp_E_comparison"] = {
            "old_f1_macro": old_e.get("f1_macro"),
            "old_accuracy": old_e.get("accuracy"),
            "old_adaptive_f1_macro": old_e.get("adaptive_f1_macro"),
            "note": "Old Exp E used weak 10-class NB15 model (in-domain F1≈0.45)"
        }

    # Retention calculations
    # In-domain baselines for retention
    INDOMAIN_CIC_F1 = 0.9891    # from Q1_REVIZYON_SONUCLARI.md
    INDOMAIN_NB15_F1 = 0.9354   # new binary model

    if res_a and res_e:
        best_a = max(
            res_a["default"]["f1_macro"],
            res_a["adaptive"]["f1_macro"]
        )
        best_e = max(
            res_e["default"]["f1_macro"],
            res_e["adaptive"]["f1_macro"]
        )
        results["retention"] = {
            "exp_A_CIC_to_NB15": {
                "indomain_f1": INDOMAIN_CIC_F1,
                "transfer_f1": best_a,
                "retention": compute_retention(INDOMAIN_CIC_F1, best_a),
            },
            "exp_E_NB15_to_CIC": {
                "indomain_f1": INDOMAIN_NB15_F1,
                "transfer_f1": best_e,
                "retention": compute_retention(INDOMAIN_NB15_F1, best_e),
            },
        }
        print("\n  RETENTION:")
        for k, v in results["retention"].items():
            print(f"    {k}: {v['retention']:.4f} "
                  f"(transfer={v['transfer_f1']:.4f} / indomain={v['indomain_f1']:.4f})")

    results["elapsed_seconds"] = time.time() - t0

    # Save
    out_path = os.path.join(PROJECT_ROOT, args.out)
    # Strip non-serializable report strings
    for exp_key in ["exp_A_rerun", "exp_E_rerun"]:
        if exp_key in results:
            for thr_key in ["default", "adaptive"]:
                results[exp_key][thr_key].pop("report", None)

    save_json(results, out_path)

    print(f"\n{'=' * 60}")
    print(f"  E1 COMPLETE — {results['elapsed_seconds']:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

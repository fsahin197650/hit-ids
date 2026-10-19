#!/usr/bin/env python3
"""
Cross-Dataset Evaluation Runner for HiT-IDS.

Runs all cross-dataset transfer experiments and collects results into
a summary table. This is the main orchestrator script.

Experiments:
  A: CIC-IDS2017 → UNSW-NB15 (binary transfer)
  B: CIC-IDS2017 → CIC-IDS2018 (same-family transfer)
  E: UNSW-NB15 → CIC-IDS2017 (reverse transfer, requires Exp C)
  F: CIC-IDS2018 → CIC-IDS2017 (reverse transfer, requires Exp D)

Usage:
    # Run all transfer experiments (A + B):
    python scripts/evaluate_cross_dataset.py --experiments A B

    # Run reverse transfers (after C and D training):
    python scripts/evaluate_cross_dataset.py --experiments E F

    # Run all and summarize:
    python scripts/evaluate_cross_dataset.py --experiments A B E F --summary
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
import yaml
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hit_ids import HiTIDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results/cross_dataset"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model(config_path: str, checkpoint_path: str,
               num_classes: int, device: str = DEVICE) -> HiTIDS:
    """Load a trained HiT-IDS model from config and checkpoint."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    fg = config["feature_groups"]
    group_names = list(fg.keys())
    group_sizes = [len(fg[n]) for n in group_names]
    mcfg = config["model"]

    # Try to infer actual group sizes from checkpoint (handles dynamic one-hot)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    for i, gn in enumerate(group_names):
        key = f"embeddings.{gn}.feature_weight"
        if key in state_dict:
            actual_size = state_dict[key].shape[0]
            if actual_size != group_sizes[i]:
                print(f"    [INFO] Group '{gn}' size adjusted: "
                      f"{group_sizes[i]} → {actual_size} (from checkpoint)")
                group_sizes[i] = actual_size

    model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=mcfg["d_emb"], nhead=mcfg["nhead"],
        intra_layers=mcfg["intra_group_layers"],
        inter_layers=mcfg["inter_group_layers"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=mcfg["dropout"], num_classes=num_classes,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def find_adaptive_threshold(probs: np.ndarray) -> float:
    """
    Find optimal binary decision threshold using GMM on softmax outputs.

    Fits a 2-component Gaussian Mixture Model to the attack-class
    probability distribution and places the threshold at the
    intersection of the two Gaussians. This is unsupervised —
    no target labels are used.
    """
    probs_2d = probs.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(probs_2d)

    # Identify which component has the higher mean (attack-like)
    means = gmm.means_.flatten()
    lo_idx = int(np.argmin(means))
    hi_idx = int(np.argmax(means))

    # Threshold = midpoint between two component means,
    # weighted by their standard deviations
    stds = np.sqrt(gmm.covariances_.flatten())
    # Weighted midpoint: closer to the low component if it is tighter
    threshold = (means[lo_idx] * stds[hi_idx] + means[hi_idx] * stds[lo_idx]) / (
        stds[lo_idx] + stds[hi_idx]
    )

    # Clamp to reasonable range
    threshold = float(np.clip(threshold, 0.01, 0.95))
    return threshold


def evaluate_binary(model: HiTIDS, X: torch.Tensor, y_bin: torch.Tensor,
                    batch_size: int = 512) -> dict:
    """
    Evaluate a binary model with both default (argmax) and adaptive threshold.

    Returns metrics for both approaches.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            logits = model(xb, return_explanations=False)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs)
    y_true = y_bin.numpy()

    # --- Default threshold (argmax, i.e. 0.5) ---
    y_pred_default = (all_probs >= 0.5).astype(int)
    default_metrics = _compute_metrics(y_true, y_pred_default)

    # --- Adaptive threshold (GMM) ---
    adaptive_thr = find_adaptive_threshold(all_probs)
    y_pred_adaptive = (all_probs >= adaptive_thr).astype(int)
    adaptive_metrics = _compute_metrics(y_true, y_pred_adaptive)

    return {
        # Default threshold results
        **default_metrics,
        # Adaptive threshold results
        "adaptive_threshold": adaptive_thr,
        "adaptive_accuracy": adaptive_metrics["accuracy"],
        "adaptive_f1_macro": adaptive_metrics["f1_macro"],
        "adaptive_f1_binary": adaptive_metrics["f1_binary"],
        "adaptive_precision": adaptive_metrics["precision"],
        "adaptive_recall": adaptive_metrics["recall"],
        # Reports
        "report": default_metrics["report"],
        "adaptive_report": adaptive_metrics["report"],
        # Distribution info
        "prob_attack_mean": float(all_probs.mean()),
        "prob_attack_max": float(all_probs.max()),
        "prob_attack_p95": float(np.percentile(all_probs, 95)),
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def run_experiment_A():
    """CIC-IDS2017 → UNSW-NB15 binary transfer."""
    print("\n" + "=" * 60)
    print("  Experiment A: CIC-IDS2017 → UNSW-NB15 (Binary)")
    print("=" * 60)

    aligned_path = "data/unsw_nb15/aligned_for_2017_model.pt"
    if not os.path.exists(aligned_path):
        print("  [ERROR] Aligned data not found. Run first:")
        print("    python scripts/cross_dataset_align.py --direction nb15_to_cic2017")
        return None

    # Load CIC-IDS2017 binary model
    model = load_model("configs/config.yaml",
                        "results/checkpoints/hit_ids_binary.pt",
                        num_classes=2)

    data = torch.load(aligned_path, weights_only=False)
    metrics = evaluate_binary(model, data["X"], data["y_bin"])

    result = {
        "experiment": "A",
        "description": "CIC-IDS2017 → UNSW-NB15 (Binary Transfer)",
        "train_dataset": "CIC-IDS2017",
        "test_dataset": "UNSW-NB15",
        "test_samples": len(data["X"]),
        **{k: v for k, v in metrics.items() if k not in ("report", "adaptive_report")},
    }

    print(f"\n  Default (argmax, thr=0.50):")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    F1-macro:  {metrics['f1_macro']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"\n  Adaptive (thr={metrics['adaptive_threshold']:.4f}):")
    print(f"    Accuracy:  {metrics['adaptive_accuracy']:.4f}")
    print(f"    F1-macro:  {metrics['adaptive_f1_macro']:.4f}")
    print(f"    Precision: {metrics['adaptive_precision']:.4f}")
    print(f"    Recall:    {metrics['adaptive_recall']:.4f}")
    print(f"\n{metrics['adaptive_report']}")

    save_path = os.path.join(RESULTS_DIR, "exp_A_cic2017_to_nb15.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Saved to {save_path}")

    return result


def run_experiment_B():
    """CIC-IDS2017 → CIC-IDS2018 binary + cascade transfer."""
    print("\n" + "=" * 60)
    print("  Experiment B: CIC-IDS2017 → CIC-IDS2018")
    print("=" * 60)

    aligned_path = "data/cicids2018/aligned_for_2017_model.pt"
    if not os.path.exists(aligned_path):
        print("  [ERROR] Aligned data not found. Run first:")
        print("    python scripts/preprocess_cicids2018.py --mode align")
        return None

    data = torch.load(aligned_path, weights_only=False)

    # Binary evaluation
    bin_model = load_model("configs/config.yaml",
                            "results/checkpoints/hit_ids_binary.pt",
                            num_classes=2)

    bin_metrics = evaluate_binary(bin_model, data["X"], data["y_bin"])

    print(f"\n  Binary Transfer:")
    print(f"    Accuracy:  {bin_metrics['accuracy']:.4f}")
    print(f"    F1-macro:  {bin_metrics['f1_macro']:.4f}")
    print(f"    Precision: {bin_metrics['precision']:.4f}")
    print(f"    Recall:    {bin_metrics['recall']:.4f}")
    print(f"\n{bin_metrics['report']}")

    # Cascade evaluation (multi-class)
    atk_ckpt = "results/checkpoints/hit_ids_attack_merged.pt"
    cascade_f1 = None

    if os.path.exists(atk_ckpt) and "y_multi" in data:
        atk_model = load_model("configs/config.yaml", atk_ckpt, num_classes=12)

        all_preds = []
        with torch.no_grad():
            for i in range(0, len(data["X"]), 256):
                xb = data["X"][i:i+256].to(DEVICE)
                bp = bin_model(xb, return_explanations=False).argmax(1)
                final = torch.zeros(len(xb), dtype=torch.long)
                mask = bp == 1
                if mask.any():
                    ap = atk_model(xb[mask], return_explanations=False).argmax(1).cpu() + 1
                    final[mask] = ap
                all_preds.extend(final.tolist())

        y_multi_pred = np.array(all_preds)
        y_multi_true = data["y_multi"].numpy()

        # Only evaluate on samples with known labels
        valid = y_multi_true >= 0
        if valid.sum() > 0:
            cascade_f1 = float(f1_score(
                y_multi_true[valid], y_multi_pred[valid],
                average="macro", zero_division=0
            ))
            print(f"\n  Cascade Multi-class F1 (common classes): {cascade_f1:.4f}")

    result = {
        "experiment": "B",
        "description": "CIC-IDS2017 → CIC-IDS2018 (Same-Family Transfer)",
        "train_dataset": "CIC-IDS2017",
        "test_dataset": "CIC-IDS2018",
        "test_samples": len(data["X"]),
        **{k: v for k, v in bin_metrics.items() if k not in ("report", "adaptive_report")},
    }
    if cascade_f1 is not None:
        result["cascade_f1_macro"] = cascade_f1

    print(f"\n  Default (argmax, thr=0.50):")
    print(f"    Accuracy:  {bin_metrics['accuracy']:.4f}")
    print(f"    F1-macro:  {bin_metrics['f1_macro']:.4f}")
    print(f"    Precision: {bin_metrics['precision']:.4f}")
    print(f"    Recall:    {bin_metrics['recall']:.4f}")
    print(f"\n  Adaptive (thr={bin_metrics['adaptive_threshold']:.4f}):")
    print(f"    Accuracy:  {bin_metrics['adaptive_accuracy']:.4f}")
    print(f"    F1-macro:  {bin_metrics['adaptive_f1_macro']:.4f}")
    print(f"    Precision: {bin_metrics['adaptive_precision']:.4f}")
    print(f"    Recall:    {bin_metrics['adaptive_recall']:.4f}")
    print(f"\n{bin_metrics['adaptive_report']}")

    save_path = os.path.join(RESULTS_DIR, "exp_B_cic2017_to_cic2018.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Saved to {save_path}")

    return result


def run_experiment_E():
    """UNSW-NB15 → CIC-IDS2017 reverse binary transfer."""
    print("\n" + "=" * 60)
    print("  Experiment E: UNSW-NB15 → CIC-IDS2017 (Binary)")
    print("=" * 60)

    aligned_path = "data/splits/aligned_for_nb15_model.pt"
    ckpt_path = "results/unsw_nb15/checkpoints/hit_ids_binary_unsw.pt"

    if not os.path.exists(aligned_path):
        print("  [ERROR] Aligned data not found. Run first:")
        print("    python scripts/cross_dataset_align.py --direction cic2017_to_nb15")
        return None

    if not os.path.exists(ckpt_path):
        print("  [ERROR] NB15 model not found. Run Experiment C first:")
        print("    python scripts/train_unsw_nb15.py")
        return None

    # Load NB15 binary model
    model = load_model("configs/config_unsw.yaml", ckpt_path, num_classes=2)

    data = torch.load(aligned_path, weights_only=False)
    metrics = evaluate_binary(model, data["X"], data["y_bin"])

    print(f"\n  Default (argmax, thr=0.50):")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    F1-macro:  {metrics['f1_macro']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"\n  Adaptive (thr={metrics['adaptive_threshold']:.4f}):")
    print(f"    Accuracy:  {metrics['adaptive_accuracy']:.4f}")
    print(f"    F1-macro:  {metrics['adaptive_f1_macro']:.4f}")
    print(f"    Precision: {metrics['adaptive_precision']:.4f}")
    print(f"    Recall:    {metrics['adaptive_recall']:.4f}")
    print(f"\n{metrics['adaptive_report']}")

    result = {
        "experiment": "E",
        "description": "UNSW-NB15 → CIC-IDS2017 (Reverse Binary Transfer)",
        "train_dataset": "UNSW-NB15",
        "test_dataset": "CIC-IDS2017",
        "test_samples": len(data["X"]),
        **{k: v for k, v in metrics.items() if k not in ("report", "adaptive_report")},
    }

    save_path = os.path.join(RESULTS_DIR, "exp_E_nb15_to_cic2017.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Saved to {save_path}")

    return result


def run_experiment_F():
    """CIC-IDS2018 → CIC-IDS2017 reverse binary transfer."""
    print("\n" + "=" * 60)
    print("  Experiment F: CIC-IDS2018 → CIC-IDS2017 (Binary)")
    print("=" * 60)

    ckpt_path = "results/cicids2018/checkpoints/hit_ids_binary_2018.pt"
    if not os.path.exists(ckpt_path):
        print("  [ERROR] CIC-IDS2018 model not found. Run Experiment D first:")
        print("    python scripts/train_cicids2018.py")
        return None

    # Load CIC-IDS2018 binary model (same architecture as CIC-IDS2017)
    model = load_model("configs/config.yaml", ckpt_path, num_classes=2)

    # Load CIC-IDS2017 test data directly (same feature space)
    test_data = torch.load("data/splits/test.pt", weights_only=False)
    X_test = test_data["X"]
    y_test = test_data["y"]
    y_bin = (y_test > 0).long()

    metrics = evaluate_binary(model, X_test, y_bin)

    print(f"\n  Default (argmax, thr=0.50):")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    F1-macro:  {metrics['f1_macro']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"\n  Adaptive (thr={metrics['adaptive_threshold']:.4f}):")
    print(f"    Accuracy:  {metrics['adaptive_accuracy']:.4f}")
    print(f"    F1-macro:  {metrics['adaptive_f1_macro']:.4f}")
    print(f"    Precision: {metrics['adaptive_precision']:.4f}")
    print(f"    Recall:    {metrics['adaptive_recall']:.4f}")
    print(f"\n{metrics['adaptive_report']}")

    result = {
        "experiment": "F",
        "description": "CIC-IDS2018 → CIC-IDS2017 (Reverse Same-Family Transfer)",
        "train_dataset": "CIC-IDS2018",
        "test_dataset": "CIC-IDS2017",
        "test_samples": len(X_test),
        **{k: v for k, v in metrics.items() if k not in ("report", "adaptive_report")},
    }

    save_path = os.path.join(RESULTS_DIR, "exp_F_cic2018_to_cic2017.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ✓ Saved to {save_path}")

    return result


def print_summary():
    """Print summary table of all experiments."""
    import glob

    print("\n" + "=" * 80)
    print("  CROSS-DATASET VALIDATION SUMMARY")
    print("=" * 80)

    header = f"{'Exp':<5s} {'Train':<13s} {'Test':<13s} " \
             f"{'Thr':<6s} {'Acc':<8s} {'F1m':<8s} {'F1b':<8s} {'Prec':<8s} {'Rec':<8s}"
    print(f"\n  {header}")
    print("  " + "-" * 85)

    results = []
    for f_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "exp_*.json"))):
        with open(f_path) as fp:
            r = json.load(fp)
        results.append(r)

        exp = r.get("experiment", "?")
        train = r.get("train_dataset", "?")[:12]
        test = r.get("test_dataset", "?")[:12]

        # Default row
        acc = r.get("accuracy", 0)
        f1m = r.get("f1_macro", 0)
        f1b = r.get("f1_binary", 0)
        prec = r.get("precision", 0)
        rec = r.get("recall", 0)
        print(f"  {exp:<5s} {train:<13s} {test:<13s} "
              f"{'0.50':<6s} {acc:<8.4f} {f1m:<8.4f} {f1b:<8.4f} {prec:<8.4f} {rec:<8.4f}")

        # Adaptive row (if available)
        if "adaptive_threshold" in r:
            athr = r["adaptive_threshold"]
            aacc = r.get("adaptive_accuracy", 0)
            af1m = r.get("adaptive_f1_macro", 0)
            af1b = r.get("adaptive_f1_binary", 0)
            aprec = r.get("adaptive_precision", 0)
            arec = r.get("adaptive_recall", 0)
            print(f"  {'↳':<5s} {'':13s} {'':13s} "
                  f"{athr:<6.3f} {aacc:<8.4f} {af1m:<8.4f} {af1b:<8.4f} {aprec:<8.4f} {arec:<8.4f}")

    print("  " + "-" * 85)
    print("=" * 80)

    # Also check in-domain results
    indomain_files = [
        ("CIC-IDS2017", "results/tables/comparison_results.json"),
        ("UNSW-NB15", "results/unsw_nb15/tables/unsw_results.json"),
        ("CIC-IDS2018", "results/cicids2018/tables/cicids2018_results.json"),
    ]

    print("\n  IN-DOMAIN BASELINES (for comparison):")
    print("  " + "-" * 50)
    for name, path in indomain_files:
        if os.path.exists(path):
            with open(path) as fp:
                r = json.load(fp)
            if isinstance(r, dict):
                if "HiT-IDS (Cascade)" in r:
                    r = r["HiT-IDS (Cascade)"]
                acc = r.get("accuracy", "?")
                f1 = r.get("f1_macro", "?")
                print(f"  {name:<15s}  Acc={acc}, F1-macro={f1}")

    # Save combined summary
    summary = {"experiments": results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    summary_path = os.path.join(RESULTS_DIR, "cross_dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Summary saved to {summary_path}")


EXPERIMENTS = {
    "A": run_experiment_A,
    "B": run_experiment_B,
    "E": run_experiment_E,
    "F": run_experiment_F,
}


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Evaluation")
    parser.add_argument("--experiments", nargs="+",
                        choices=["A", "B", "E", "F", "all"],
                        default=["A", "B"],
                        help="Which experiments to run")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table after experiments")
    args = parser.parse_args()

    print("=" * 60)
    print("  HiT-IDS Cross-Dataset Evaluation")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    if "all" in args.experiments:
        exps = ["A", "B", "E", "F"]
    else:
        exps = args.experiments

    print(f"  Experiments: {exps}")
    start = time.time()

    for exp in exps:
        if exp in EXPERIMENTS:
            try:
                EXPERIMENTS[exp]()
            except Exception as e:
                print(f"\n  [ERROR] Experiment {exp} failed: {e}")
                import traceback
                traceback.print_exc()

    if args.summary or len(exps) > 1:
        print_summary()

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()

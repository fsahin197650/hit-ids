"""
Evaluation metrics for HiT-IDS.

Computes classification metrics, generates confusion matrices,
and produces formatted comparison tables.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsCalculator:
    """
    Computes all evaluation metrics for IDS models.

    Metrics:
        - Accuracy
        - Precision (macro & weighted)
        - Recall (macro & weighted)
        - F1-Score (macro & weighted)
        - Per-class F1
        - Confusion Matrix

    Usage:
        >>> calc = MetricsCalculator(num_classes=15, label_names=label_names)
        >>> metrics = calc.compute(y_true, y_pred)
        >>> calc.print_report(metrics)
    """

    def __init__(
        self,
        num_classes: int = 15,
        label_names: Optional[List[str]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.label_names = label_names or [f"Class_{i}" for i in range(num_classes)]

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute all metrics.

        Args:
            y_true: Ground truth labels, shape (N,).
            y_pred: Predicted labels, shape (N,).
            y_prob: Optional predicted probabilities, shape (N, num_classes).

        Returns:
            Dict with all computed metrics.
        """
        metrics: Dict[str, Any] = {}

        # Overall metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # Per-class F1
        per_class_f1 = f1_score(
            y_true, y_pred, average=None, zero_division=0,
            labels=list(range(self.num_classes)),
        )
        metrics["per_class_f1"] = {
            self.label_names[i]: float(per_class_f1[i])
            for i in range(min(len(per_class_f1), len(self.label_names)))
        }

        # Confusion matrix
        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(self.num_classes))
        )
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report (string)
        metrics["classification_report"] = classification_report(
            y_true, y_pred,
            target_names=self.label_names[:self.num_classes],
            zero_division=0,
            labels=list(range(self.num_classes)),
        )

        return metrics

    def print_report(self, metrics: Dict[str, Any]) -> None:
        """Print a formatted metrics report."""
        print("\n" + "=" * 60)
        print("  Classification Results")
        print("=" * 60)
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
        print("\n  Per-Class F1:")
        for name, f1 in metrics["per_class_f1"].items():
            bar = "█" * int(f1 * 30)
            print(f"    {name:30s}: {f1:.4f}  {bar}")
        print("\n" + metrics["classification_report"])

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        save_path: str,
        model_name: str = "HiT-IDS",
    ) -> None:
        """Save metrics to JSON file."""
        # Remove non-serializable items
        save_data = {k: v for k, v in metrics.items() if k != "classification_report"}
        save_data["model_name"] = model_name

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Metrics saved to: {save_path}")


def create_comparison_table(
    results: Dict[str, Dict[str, Any]],
    metrics_to_show: Optional[List[str]] = None,
) -> str:
    """
    Create a formatted comparison table for multiple models.

    Args:
        results: Dict mapping model_name → metrics dict.
        metrics_to_show: Which metrics to include in the table.

    Returns:
        Formatted table string (also prints it).
    """
    if metrics_to_show is None:
        metrics_to_show = [
            "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_weighted",
        ]

    # Header
    header = f"{'Model':<25s}"
    for metric in metrics_to_show:
        short_name = metric.replace("_macro", "(M)").replace("_weighted", "(W)")
        header += f" {short_name:>12s}"
    separator = "─" * len(header)

    lines = [separator, header, separator]

    # Data rows
    best_values = {m: -1.0 for m in metrics_to_show}
    for model_name, metrics in results.items():
        for m in metrics_to_show:
            best_values[m] = max(best_values[m], metrics.get(m, 0.0))

    for model_name, metrics in results.items():
        row = f"{model_name:<25s}"
        for metric in metrics_to_show:
            val = metrics.get(metric, 0.0)
            is_best = abs(val - best_values[metric]) < 1e-6
            marker = " *" if is_best else "  "
            row += f" {val:>10.4f}{marker}"
        lines.append(row)

    lines.append(separator)
    lines.append("  * = best")

    table = "\n".join(lines)
    print(table)
    return table


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing MetricsCalculator...")

    np.random.seed(42)
    num_classes = 5
    N = 200

    y_true = np.random.randint(0, num_classes, N)
    y_pred = y_true.copy()
    # Introduce some errors
    errors = np.random.choice(N, size=30, replace=False)
    y_pred[errors] = np.random.randint(0, num_classes, len(errors))

    label_names = ["Benign", "DDoS", "PortScan", "Bot", "DoS"]

    calc = MetricsCalculator(num_classes=num_classes, label_names=label_names)
    metrics = calc.compute(y_true, y_pred)
    calc.print_report(metrics)

    # Test comparison table
    print("\n--- Comparison Table ---")
    results = {
        "HiT-IDS": metrics,
        "Random Forest": calc.compute(y_true, np.random.randint(0, num_classes, N)),
        "XGBoost": calc.compute(y_true, np.random.randint(0, num_classes, N)),
    }
    create_comparison_table(results)

    print("\n✓ MetricsCalculator test passed!")

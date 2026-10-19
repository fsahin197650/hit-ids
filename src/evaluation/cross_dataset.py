"""
Cross-dataset evaluation for HiT-IDS.

Tests generalization capability by evaluating on a different
IDS dataset (e.g., CSE-CIC-IDS2018) without retraining.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.hit_ids import HiTIDS
from .metrics import MetricsCalculator


class CrossDatasetEvaluator:
    """
    Evaluates a trained HiT-IDS model on a different dataset.

    The model is trained on CIC-IDS2017 and tested directly on
    CSE-CIC-IDS2018 (or other datasets) to test transferability.

    Key challenge: Label spaces may differ between datasets.
    We evaluate on the common label subset.

    Args:
        model: Trained HiT-IDS model.
        config: Full config dict.
        device: Computation device.
    """

    def __init__(
        self,
        model: HiTIDS,
        config: dict,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

    def evaluate_on_dataset(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        label_names: Optional[List[str]] = None,
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a new dataset.

        Args:
            X: Feature tensor, shape (N, total_features).
            y: Labels, shape (N,).
            label_names: Optional class label names.
            batch_size: Batch size for inference.

        Returns:
            Dict with classification metrics.
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch, return_explanations=False)
                preds = logits.argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_targets.extend(y_batch.tolist())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        num_classes = self.config["dataset"]["num_classes"]
        calc = MetricsCalculator(
            num_classes=num_classes,
            label_names=label_names,
        )

        metrics = calc.compute(y_true, y_pred)
        return metrics

    def evaluate_with_label_mapping(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        source_label_map: Dict[str, int],
        target_label_map: Dict[str, int],
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """
        Evaluate with explicit label mapping between datasets.

        Only evaluates on classes that exist in both datasets.

        Args:
            X: Feature tensor from target dataset.
            y: Labels from target dataset.
            source_label_map: Label name → ID mapping for source (training) dataset.
            target_label_map: Label name → ID mapping for target (test) dataset.
            batch_size: Batch size.

        Returns:
            Dict with metrics (only on common classes).
        """
        # Find common labels
        source_labels = set(source_label_map.keys())
        target_labels = set(target_label_map.keys())
        common_labels = source_labels & target_labels

        if not common_labels:
            raise ValueError("No common labels between source and target datasets!")

        print(f"  Common labels ({len(common_labels)}): {sorted(common_labels)}")

        # Create target → source label mapping
        target_to_source = {}
        for label_name in common_labels:
            target_to_source[target_label_map[label_name]] = source_label_map[label_name]

        # Filter data to common labels only
        mask = torch.tensor([
            int(y[i].item()) in target_to_source
            for i in range(len(y))
        ])
        X_common = X[mask]
        y_common = y[mask]

        # Remap labels to source label space
        y_remapped = torch.tensor([
            target_to_source[int(label.item())]
            for label in y_common
        ], dtype=torch.long)

        common_label_names = sorted(common_labels)
        return self.evaluate_on_dataset(
            X_common, y_remapped,
            label_names=common_label_names,
        )


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing CrossDatasetEvaluator...")

    model = HiTIDS(
        group_sizes=[4, 5, 3, 3], d_emb=16, nhead=2,
        intra_layers=1, inter_layers=1, num_classes=5,
    )

    config = {
        "dataset": {"num_classes": 5},
    }

    evaluator = CrossDatasetEvaluator(model, config, device="cpu")

    X = torch.randn(100, 15)
    y = torch.randint(0, 5, (100,))

    metrics = evaluator.evaluate_on_dataset(X, y)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")

    print("\n✓ CrossDatasetEvaluator test passed!")

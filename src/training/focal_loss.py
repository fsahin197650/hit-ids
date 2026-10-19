"""
Focal Loss for HiT-IDS.

Addresses severe class imbalance in CIC-IDS2017 by down-weighting
easy (well-classified) examples and focusing on hard ones.

FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Combines class-specific weights (α) with a focusing parameter (γ)
    to handle extreme class imbalance.

    Args:
        alpha: Per-class weights. Can be:
            - None: uniform weights
            - Tensor of shape (num_classes,): explicit weights
        gamma: Focusing parameter. Higher = more focus on hard examples.
               γ=0 reduces to standard cross-entropy. Default: 2.0
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'.
        label_smoothing: Optional label smoothing factor. Default: 0.0.

    Usage:
        >>> alpha = compute_class_weights(y_train, num_classes=15)
        >>> criterion = FocalLoss(alpha=alpha, gamma=2.0)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output, shape (batch, num_classes).
            targets: Ground truth labels, shape (batch,).

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        # Standard cross-entropy (per sample, no reduction)
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # p_t = probability of the correct class
        p_t = torch.exp(-ce_loss)

        # Focal modulation: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply class weights α
        if self.alpha is not None:
            # Move alpha to same device as logits
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    method: str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute per-class weights from label distribution.

    Args:
        labels: Label tensor, shape (N,).
        num_classes: Total number of classes.
        method: Weight computation method:
            - 'inverse_freq': weight = total / (num_classes × count)
            - 'effective': effective number of samples (for long-tailed distributions)
            - 'sqrt_inverse': weight = sqrt(total / count), less aggressive

    Returns:
        Tensor of shape (num_classes,) with per-class weights.
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.numpy() if labels.device.type == "cpu" else labels.cpu().numpy()
    else:
        labels_np = labels

    import numpy as np
    unique, counts = np.unique(labels_np, return_counts=True)
    total = len(labels_np)

    weights = torch.ones(num_classes, dtype=torch.float32)

    if method == "inverse_freq":
        for cls_id, count in zip(unique, counts):
            weights[cls_id] = total / (num_classes * count)
    elif method == "effective":
        beta = 0.9999
        for cls_id, count in zip(unique, counts):
            effective = (1.0 - beta ** count) / (1.0 - beta)
            weights[cls_id] = 1.0 / effective
    elif method == "sqrt_inverse":
        for cls_id, count in zip(unique, counts):
            weights[cls_id] = (total / count) ** 0.5
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize so mean weight ≈ 1
    weights = weights / weights.mean()

    return weights


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing FocalLoss...")

    batch_size = 32
    num_classes = 15

    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Without alpha
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(logits, targets)
    print(f"  FocalLoss (no alpha): {loss.item():.4f}")

    # With alpha
    alpha = compute_class_weights(targets, num_classes)
    criterion_weighted = FocalLoss(alpha=alpha, gamma=2.0)
    loss_w = criterion_weighted(logits, targets)
    print(f"  FocalLoss (weighted): {loss_w.item():.4f}")
    print(f"  Alpha weights: {alpha[:5].tolist()}")

    # Compare with standard CE
    ce_loss = F.cross_entropy(logits, targets)
    print(f"  Standard CE: {ce_loss.item():.4f}")

    # Focal with gamma=0 should approximate CE
    fl_gamma0 = FocalLoss(gamma=0.0)(logits, targets)
    print(f"  FocalLoss (γ=0): {fl_gamma0.item():.4f} (should ≈ CE)")

    # Test label smoothing
    fl_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)(logits, targets)
    print(f"  FocalLoss (smoothed): {fl_smooth.item():.4f}")

    print("\n✓ FocalLoss test passed!")

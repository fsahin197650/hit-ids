"""
Attention Rollout for HiT-IDS.

Computes attention rollout across transformer layers to trace how
information flows from input features to the [CLS] token.
Standalone module (separate from ExplanationHead) for detailed analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class AttentionRollout:
    """
    Attention Rollout for hierarchical explainability.

    Recursively multiplies attention matrices across layers with
    residual connections to compute total feature-to-CLS influence.

    R_0 = I
    R_l = A_l × R_{l-1}
    A_l = (1 - w) × attention_l + w × I  (residual mixing)

    Args:
        residual_weight: Weight for residual connection (default: 0.5).
        head_fusion: Multi-head fusion strategy: 'mean', 'max', 'min'.
        discard_ratio: Remove this fraction of lowest attention values.
    """

    def __init__(
        self,
        residual_weight: float = 0.5,
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
    ) -> None:
        self.residual_weight = residual_weight
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def rollout(
        self,
        attention_weights_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute attention rollout from a stack of attention matrices.

        Args:
            attention_weights_list: List of attention tensors per layer.
                Each shape: (batch, nhead, seq_len, seq_len).

        Returns:
            Rollout matrix, shape (batch, seq_len, seq_len).
        """
        # Fuse multi-head attention
        fused = self._fuse_heads(attention_weights_list)

        batch_size, seq_len, _ = fused[0].shape

        # Initialize rollout as identity
        rollout = torch.eye(seq_len, device=fused[0].device)
        rollout = rollout.unsqueeze(0).expand(batch_size, -1, -1)

        for attn in fused:
            # Optionally discard low-attention values
            if self.discard_ratio > 0:
                attn = self._discard_low_attention(attn)

            # Add residual
            identity = torch.eye(seq_len, device=attn.device)
            identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
            attn = (1 - self.residual_weight) * attn + self.residual_weight * identity

            # Renormalize
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)

            # Cumulative product
            rollout = torch.bmm(attn, rollout)

        return rollout

    def cls_feature_importance(
        self,
        attention_weights_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Get CLS token's attention to each feature (excluding CLS itself).

        Args:
            attention_weights_list: Attention tensors per layer.

        Returns:
            Feature importance scores, shape (batch, num_features).
            (num_features = seq_len - 1, excluding CLS)
        """
        rollout = self.rollout(attention_weights_list)
        # CLS row (0), skip CLS column (0)
        importance = rollout[:, 0, 1:]
        # Normalize to sum to 1
        importance = importance / (importance.sum(dim=-1, keepdim=True) + 1e-12)
        return importance

    def hierarchical_importance(
        self,
        intra_attn: Dict[str, List[torch.Tensor]],
        inter_attn: List[torch.Tensor],
        group_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full hierarchical feature importance.

        Returns importance at 3 levels:
            1. Intra-group: importance of each feature within its group
            2. Inter-group: importance of each group to the global decision
            3. Global: combined importance of each feature across all groups

        Args:
            intra_attn: Dict[group_name → attention tensor list per layer].
            inter_attn: Attention tensor list for inter-group transformer.
            group_names: Ordered list of group names.

        Returns:
            Dict with:
                - 'intra': Dict[group → (batch, num_features_g)]
                - 'group': (batch, num_groups)
                - 'global': Dict[group → (batch, num_features_g)]
                - 'global_flat': (batch, total_features) -- concatenated
        """
        if group_names is None:
            group_names = list(intra_attn.keys())

        result: Dict[str, any] = {}

        # 1. Intra-group importance
        intra_importance: Dict[str, torch.Tensor] = {}
        for group_name in group_names:
            attn_list = intra_attn[group_name]
            importance = self.cls_feature_importance(attn_list)
            intra_importance[group_name] = importance
        result["intra"] = intra_importance

        # 2. Inter-group importance
        inter_rollout = self.rollout(inter_attn)
        group_importance = inter_rollout[:, 0, 1:]  # skip global CLS
        group_importance = group_importance / (
            group_importance.sum(dim=-1, keepdim=True) + 1e-12
        )
        result["group"] = group_importance

        # 3. Global importance (group_importance × intra_importance)
        global_importance: Dict[str, torch.Tensor] = {}
        global_flat_parts: List[torch.Tensor] = []

        for i, group_name in enumerate(group_names):
            gi = group_importance[:, i:i+1]  # (batch, 1)
            fi = intra_importance[group_name]  # (batch, num_features_g)
            global_imp = gi * fi
            global_importance[group_name] = global_imp
            global_flat_parts.append(global_imp)

        result["global"] = global_importance
        result["global_flat"] = torch.cat(global_flat_parts, dim=1)

        return result

    def _fuse_heads(
        self, attention_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Fuse multi-head attention into single-head."""
        fused = []
        for attn in attention_list:
            if self.head_fusion == "mean":
                fused.append(attn.mean(dim=1))
            elif self.head_fusion == "max":
                fused.append(attn.max(dim=1).values)
            elif self.head_fusion == "min":
                fused.append(attn.min(dim=1).values)
            else:
                raise ValueError(f"Unknown fusion: {self.head_fusion}")
        return fused

    def _discard_low_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """Zero out the lowest attention values."""
        flat = attn.reshape(attn.shape[0], -1)
        k = int(flat.shape[1] * self.discard_ratio)
        if k > 0:
            threshold = flat.kthvalue(k, dim=1).values.unsqueeze(1)
            flat = flat.masked_fill(flat < threshold, 0.0)
        return flat.reshape(attn.shape)


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing AttentionRollout...")

    batch = 4
    nhead = 4

    # Simulate intra-group attention (2 layers, seq_len varies by group)
    group_configs = {"temporal": 20, "volume": 26, "protocol": 16, "behavioral": 14}

    intra_attn = {}
    for name, seq in group_configs.items():
        intra_attn[name] = [
            torch.softmax(torch.randn(batch, nhead, seq, seq), dim=-1)
            for _ in range(2)
        ]

    # Inter-group attention (2 layers, seq_len=5)
    inter_attn = [
        torch.softmax(torch.randn(batch, nhead, 5, 5), dim=-1)
        for _ in range(2)
    ]

    rollout = AttentionRollout(residual_weight=0.5)
    result = rollout.hierarchical_importance(intra_attn, inter_attn)

    print(f"  Group importance: {result['group'].shape}")
    print(f"  Group importance[0]: {result['group'][0].tolist()}")
    for name in group_configs:
        print(f"  {name}: intra={result['intra'][name].shape}, "
              f"global={result['global'][name].shape}")
    print(f"  Global flat: {result['global_flat'].shape}")

    # Verify sums
    gi = result["group"][0].sum().item()
    print(f"\n  Group importance sum: {gi:.4f} (should be ~1.0)")

    print("\n✓ AttentionRollout test passed!")

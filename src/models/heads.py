"""
Classification Head and Explanation Head for HiT-IDS.

ClassificationHead: Projects global CLS token → class logits.
ExplanationHead: Computes hierarchical feature importance from attention weights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Two-layer MLP classification head.

    Maps global CLS token (d_emb) → class logits (num_classes).

    Architecture:
        Linear(d_emb, d_emb) → ReLU → Dropout → Linear(d_emb, num_classes)

    Args:
        d_emb: Input embedding dimension (default: 64).
        num_classes: Number of output classes (default: 15).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        d_emb: int = 64,
        num_classes: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            cls_token: Global CLS representation, shape (batch, d_emb).

        Returns:
            Logits, shape (batch, num_classes).
        """
        return self.head(cls_token)


class ExplanationHead(nn.Module):
    """
    Computes hierarchical feature importance from attention weights.

    Uses attention rollout to trace information flow from input features
    through intra-group and inter-group transformers to the final prediction.

    This is the INTRINSIC XAI component — explanations are derived directly
    from the model's own attention mechanism, not from an external method.

    Args:
        group_names: List of group names (default: 4 groups).
        residual_weight: Weight for residual connection in rollout (default: 0.5).
        head_fusion: How to fuse multi-head attention: 'mean', 'max', 'min'.
    """

    def __init__(
        self,
        group_names: Optional[List[str]] = None,
        residual_weight: float = 0.5,
        head_fusion: str = "mean",
    ) -> None:
        super().__init__()

        self.group_names = group_names or ["temporal", "volume", "protocol", "behavioral"]
        self.residual_weight = residual_weight
        self.head_fusion = head_fusion

    def compute_attention_rollout(
        self,
        attention_weights_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute attention rollout across multiple transformer layers.

        Attention rollout recursively combines attention matrices:
            R_0 = I (identity)
            R_l = A_l × R_{l-1}
        where A_l = (1 - w) * attention_l + w * I  (residual connection)

        Args:
            attention_weights_list: List of attention weight tensors per layer,
                each shape (batch, nhead, seq_len, seq_len).

        Returns:
            Rollout matrix, shape (batch, seq_len, seq_len).
            Row 0 gives the importance of each token to [CLS].
        """
        # Fuse multi-head attention
        fused_attentions = []
        for attn in attention_weights_list:
            if self.head_fusion == "mean":
                fused = attn.mean(dim=1)  # (batch, seq, seq)
            elif self.head_fusion == "max":
                fused = attn.max(dim=1).values
            elif self.head_fusion == "min":
                fused = attn.min(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion: {self.head_fusion}")
            fused_attentions.append(fused)

        batch_size, seq_len, _ = fused_attentions[0].shape

        # Initialize rollout as identity matrix
        rollout = torch.eye(
            seq_len, device=fused_attentions[0].device
        ).unsqueeze(0).expand(batch_size, -1, -1)

        # Multiply through layers with residual connection
        for attn in fused_attentions:
            # Add residual: A' = (1 - w) * A + w * I
            identity = torch.eye(
                seq_len, device=attn.device
            ).unsqueeze(0).expand(batch_size, -1, -1)
            attn_with_residual = (
                (1 - self.residual_weight) * attn
                + self.residual_weight * identity
            )

            # Re-normalize rows to sum to 1
            attn_with_residual = attn_with_residual / (
                attn_with_residual.sum(dim=-1, keepdim=True) + 1e-12
            )

            # Cumulative product
            rollout = torch.bmm(attn_with_residual, rollout)

        return rollout

    def compute_hierarchical_importance(
        self,
        intra_attn_weights: Dict[str, List[torch.Tensor]],
        inter_attn_weights: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical feature importance combining intra and inter-group attention.

        Three levels of importance:
            1. Per-group feature importance (from intra-group attention rollout)
            2. Group importance (from inter-group attention)
            3. Global feature importance = group_importance × feature_importance

        Args:
            intra_attn_weights: Dict mapping group_name → list of attention tensors
                per layer. Each tensor shape: (batch, nhead, seq_len_g, seq_len_g).
            inter_attn_weights: List of attention tensors per layer.
                Each tensor shape: (batch, nhead, 5, 5).

        Returns:
            Dict with keys:
                - 'intra_importance': Dict[group_name → (batch, num_features_g)]
                - 'group_importance': (batch, num_groups)
                - 'global_importance': Dict[group_name → (batch, num_features_g)]
        """
        result: Dict[str, Any] = {}

        # 1. Intra-group feature importance via attention rollout
        intra_importance: Dict[str, torch.Tensor] = {}
        for group_name, attn_list in intra_attn_weights.items():
            rollout = self.compute_attention_rollout(attn_list)
            # CLS token (row 0), skip CLS column (col 0) → feature importance
            feature_importance = rollout[:, 0, 1:]  # (batch, num_features_g)
            # Normalize to sum to 1
            feature_importance = feature_importance / (
                feature_importance.sum(dim=-1, keepdim=True) + 1e-12
            )
            intra_importance[group_name] = feature_importance

        result["intra_importance"] = intra_importance

        # 2. Inter-group importance via attention rollout
        inter_rollout = self.compute_attention_rollout(inter_attn_weights)
        # Global CLS (row 0), skip global CLS column (col 0) → group importance
        group_importance = inter_rollout[:, 0, 1:]  # (batch, 4)
        group_importance = group_importance / (
            group_importance.sum(dim=-1, keepdim=True) + 1e-12
        )
        result["group_importance"] = group_importance

        # 3. Global feature importance = group_importance × intra_importance
        global_importance: Dict[str, torch.Tensor] = {}
        for i, (group_name, feat_imp) in enumerate(intra_importance.items()):
            # Scale by group importance
            gi = group_importance[:, i:i+1]  # (batch, 1)
            global_importance[group_name] = feat_imp * gi  # (batch, num_features_g)

        result["global_importance"] = global_importance

        return result

    def forward(
        self,
        intra_attn_weights: Dict[str, List[torch.Tensor]],
        inter_attn_weights: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass — compute all importance scores.

        Args:
            intra_attn_weights: Dict[group_name → list of attn tensor per layer].
            inter_attn_weights: List of attn tensor per layer for inter-group.

        Returns:
            Explanation dict with intra, group, and global importance scores.
        """
        return self.compute_hierarchical_importance(
            intra_attn_weights, inter_attn_weights
        )


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing ClassificationHead and ExplanationHead...")

    batch_size = 8
    d_emb = 64
    num_classes = 15

    # Test ClassificationHead
    cls_head = ClassificationHead(d_emb=d_emb, num_classes=num_classes)
    cls_token = torch.randn(batch_size, d_emb)
    logits = cls_head(cls_token)
    assert logits.shape == (batch_size, num_classes)
    print(f"  ClassificationHead: input={cls_token.shape} → "
          f"output={logits.shape} ✓")

    # Test ExplanationHead
    nhead = 4
    group_configs = {
        "temporal": 19, "volume": 25, "protocol": 15, "behavioral": 13
    }

    # Create dummy attention weights
    intra_attn = {}
    for name, size in group_configs.items():
        seq_len = size + 1  # +1 for CLS
        layers_attn = [
            torch.softmax(torch.randn(batch_size, nhead, seq_len, seq_len), dim=-1)
            for _ in range(2)  # 2 layers
        ]
        intra_attn[name] = layers_attn

    inter_attn = [
        torch.softmax(torch.randn(batch_size, nhead, 5, 5), dim=-1)
        for _ in range(2)
    ]

    exp_head = ExplanationHead()
    explanations = exp_head(intra_attn, inter_attn)

    print(f"  Group importance: {explanations['group_importance'].shape} ✓")
    for name in group_configs:
        intra_shape = explanations["intra_importance"][name].shape
        global_shape = explanations["global_importance"][name].shape
        print(f"  {name:12s}: intra={intra_shape}, global={global_shape} ✓")

    # Verify importance sums
    gi = explanations["group_importance"][0]
    print(f"\n  Group importance sum: {gi.sum().item():.4f} (should be ~1.0)")
    for name in group_configs:
        fi = explanations["intra_importance"][name][0]
        print(f"  {name} intra sum: {fi.sum().item():.4f} (should be ~1.0)")

    n_params_cls = sum(p.numel() for p in cls_head.parameters())
    print(f"\n  ClassificationHead params: {n_params_cls:,}")
    print("\n✓ All head tests passed!")

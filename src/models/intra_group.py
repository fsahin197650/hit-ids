"""
Intra-Group Transformer Encoder for HiT-IDS.

Processes features WITHIN a single semantic group.
Returns the [CLS] token representation and attention weights for XAI.

CRITICAL: Attention weights are stored for the Explanation Head.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class IntraGroupTransformerLayer(nn.Module):
    """
    Single transformer encoder layer with explicit attention weight output.

    Uses custom implementation instead of nn.TransformerEncoderLayer
    to ensure attention weights are always accessible for XAI.
    """

    def __init__(
        self,
        d_emb: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_emb,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # IMPORTANT: (batch, seq, d_emb)
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_emb, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_emb),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm architecture for stability)
        self.norm1 = nn.LayerNorm(d_emb)
        self.norm2 = nn.LayerNorm(d_emb)

        # Dropout for residual
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, d_emb).

        Returns:
            Tuple of:
                - Output tensor, shape (batch, seq_len, d_emb)
                - Attention weights, shape (batch, nhead, seq_len, seq_len)
        """
        # Pre-norm self-attention with residual
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False,  # Keep per-head weights for XAI
        )
        x = x + self.dropout(attn_output)

        # Pre-norm FFN with residual
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output

        return x, attn_weights


class IntraGroupTransformer(nn.Module):
    """
    Transformer encoder for within-group feature interaction.

    Processes tokens from a single feature group and returns:
    1. The [CLS] token embedding (group summary)
    2. All attention weights across layers (for XAI)

    Args:
        d_emb: Embedding dimension (default: 64).
        nhead: Number of attention heads (default: 4).
        num_layers: Number of transformer layers (default: 2).
        dim_feedforward: FFN hidden dimension (default: 128).
        dropout: Dropout rate (default: 0.1).

    Input:  (batch, num_features + 1, d_emb)  — from FeatureEmbedding
    Output:
        cls_token: (batch, d_emb)
        attention_weights: list of (batch, nhead, seq_len, seq_len) per layer
    """

    def __init__(
        self,
        d_emb: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_emb = d_emb
        self.nhead = nhead
        self.num_layers = num_layers

        # Stack of transformer layers
        self.layers = nn.ModuleList([
            IntraGroupTransformerLayer(
                d_emb=d_emb,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_emb)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through all transformer layers.

        Args:
            x: Token embeddings, shape (batch, seq_len, d_emb).
               Position 0 is the [CLS] token.

        Returns:
            Tuple of:
                - cls_token: [CLS] representation, shape (batch, d_emb)
                - attn_weights_list: List of attention weight tensors,
                  one per layer, each shape (batch, nhead, seq_len, seq_len)
        """
        attn_weights_list: List[torch.Tensor] = []

        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_list.append(attn_weights)

        # Final normalization
        x = self.final_norm(x)

        # Extract [CLS] token (position 0)
        cls_token = x[:, 0, :]  # (batch, d_emb)

        return cls_token, attn_weights_list

    def get_full_sequence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning full sequence (not just CLS).
        Useful for debugging and advanced XAI.

        Args:
            x: Token embeddings, shape (batch, seq_len, d_emb).

        Returns:
            Tuple of:
                - Full sequence output, shape (batch, seq_len, d_emb)
                - attn_weights_list
        """
        attn_weights_list: List[torch.Tensor] = []

        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_list.append(attn_weights)

        x = self.final_norm(x)
        return x, attn_weights_list


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing IntraGroupTransformer...")

    batch_size = 8
    d_emb = 64
    nhead = 4
    num_layers = 2

    group_sizes = {"temporal": 19, "volume": 25, "protocol": 15, "behavioral": 13}

    for name, num_feat in group_sizes.items():
        seq_len = num_feat + 1  # +1 for CLS token
        transformer = IntraGroupTransformer(
            d_emb=d_emb, nhead=nhead, num_layers=num_layers
        )
        x = torch.randn(batch_size, seq_len, d_emb)
        cls_token, attn_weights = transformer(x)

        assert cls_token.shape == (batch_size, d_emb), (
            f"CLS shape mismatch for {name}: {cls_token.shape}"
        )
        assert len(attn_weights) == num_layers, (
            f"Wrong number of attention weight tensors: {len(attn_weights)}"
        )
        for i, aw in enumerate(attn_weights):
            assert aw.shape == (batch_size, nhead, seq_len, seq_len), (
                f"Attn weight shape mismatch at layer {i}: {aw.shape}"
            )
        print(f"  {name:12s}: seq_len={seq_len}, cls={cls_token.shape}, "
              f"attn_layers={len(attn_weights)} ✓")

    # Count parameters
    t = IntraGroupTransformer(d_emb=64, nhead=4, num_layers=2)
    n_params = sum(p.numel() for p in t.parameters())
    print(f"\n  Parameters: {n_params:,}")
    print("\n✓ IntraGroupTransformer test passed!")

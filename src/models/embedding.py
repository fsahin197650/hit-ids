"""
Feature Embedding layer for HiT-IDS.

Projects each feature in a group into d_emb-dimensional space,
prepends a learnable [CLS] token, and adds positional embeddings.
Each feature is treated as an individual token.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """
    Projects a feature group into embedding space and prepends [CLS] token.

    Each scalar feature gets its own UNIQUE projection (FT-Transformer style):
        feature_value_i (scalar) → Linear_i(1, d_emb) → token embedding_i

    This is critical: shared projections make all features indistinguishable,
    while per-feature projections let the model learn feature-specific
    transformations (e.g., "Flow Duration" vs "Fwd IAT Mean" get different
    weight matrices).

    Then a learnable [CLS] token is prepended and positional embeddings are added.

    Args:
        num_features: Number of features in this group.
        d_emb: Embedding dimension (default: 64).
        dropout: Dropout rate (default: 0.1).

    Input:  (batch, num_features)
    Output: (batch, num_features + 1, d_emb)  — includes [CLS] at position 0
    """

    def __init__(
        self,
        num_features: int,
        d_emb: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.d_emb = d_emb

        # Per-feature UNIQUE projection: each feature gets its own weights
        # This is the key FT-Transformer insight
        self.feature_weight = nn.Parameter(
            torch.empty(num_features, d_emb)
        )
        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_emb)
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_emb) * 0.02)

        # Learnable positional embedding (CLS + num_features positions)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_features + 1, d_emb) * 0.02
        )

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_emb)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize per-feature projection weights with Xavier uniform."""
        nn.init.xavier_uniform_(
            self.feature_weight.unsqueeze(0)  # treat as (1, num_features, d_emb)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features, shape (batch, num_features).

        Returns:
            Embedded tokens with [CLS], shape (batch, num_features + 1, d_emb).
        """
        batch_size = x.shape[0]

        # Per-feature projection: x_i * w_i + b_i for each feature i
        # x: (batch, num_features) → (batch, num_features, 1)
        # feature_weight: (num_features, d_emb)
        # Result: (batch, num_features, d_emb)
        token_embeddings = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)

        # Expand [CLS] token for the batch: (1, 1, d_emb) → (batch, 1, d_emb)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend [CLS]: (batch, num_features+1, d_emb)
        embeddings = torch.cat([cls_tokens, token_embeddings], dim=1)

        # Add positional embedding
        embeddings = embeddings + self.pos_embedding

        # Layer norm + dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing FeatureEmbedding...")

    batch_size = 8
    group_sizes = {"temporal": 19, "volume": 25, "protocol": 15, "behavioral": 13}

    for name, num_feat in group_sizes.items():
        emb = FeatureEmbedding(num_features=num_feat, d_emb=64, dropout=0.1)
        x = torch.randn(batch_size, num_feat)
        out = emb(x)
        assert out.shape == (batch_size, num_feat + 1, 64), (
            f"Shape mismatch for {name}: {out.shape}"
        )
        print(f"  {name:12s}: input={x.shape} → output={out.shape} ✓")

    # Count parameters
    emb = FeatureEmbedding(num_features=19, d_emb=64)
    n_params = sum(p.numel() for p in emb.parameters())
    print(f"\n  Parameters (temporal): {n_params:,}")
    print("\n✓ FeatureEmbedding test passed!")

"""
Inter-Group Transformer Encoder for HiT-IDS.

Models cross-group interactions between the 4 feature group summaries.
Takes [CLS] tokens from each intra-group transformer and processes them
together with a global [CLS] token to produce the final representation.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .intra_group import IntraGroupTransformerLayer


class InterGroupTransformer(nn.Module):
    """
    Cross-group transformer that models interactions between feature groups.

    Input: List of 4 group CLS tokens, each (batch, d_emb)
    Output:
        - global_cls: (batch, d_emb) — final representation for classification
        - attention_weights: list of (batch, nhead, 5, 5) per layer
          (5 = global_CLS + 4 group CLS tokens)

    The inter-group attention weights reveal which feature groups are most
    important for each classification decision — key for XAI.

    Args:
        d_emb: Embedding dimension (default: 64).
        nhead: Number of attention heads (default: 4).
        num_layers: Number of transformer layers (default: 2).
        dim_feedforward: FFN hidden dimension (default: 128).
        dropout: Dropout rate (default: 0.1).
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

        # Global [CLS] token — aggregates information from all groups
        self.global_cls = nn.Parameter(torch.randn(1, 1, d_emb) * 0.02)

        # Positional embeddings for [CLS_global, CLS_G1, CLS_G2, CLS_G3, CLS_G4]
        # Fixed to 5 positions (1 global + 4 groups)
        self.pos_embedding = nn.Parameter(torch.randn(1, 5, d_emb) * 0.02)

        # Layer norm before transformer
        self.input_norm = nn.LayerNorm(d_emb)

        # Transformer layers (reuse IntraGroupTransformerLayer — same architecture)
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
        self,
        group_cls_tokens: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            group_cls_tokens: List of 4 tensors, each (batch, d_emb).
                Order: [temporal_cls, volume_cls, protocol_cls, behavioral_cls]

        Returns:
            Tuple of:
                - global_cls: Global CLS representation, shape (batch, d_emb)
                - attn_weights_list: List of attention weight tensors per layer,
                  each shape (batch, nhead, 5, 5)
        """
        batch_size = group_cls_tokens[0].shape[0]

        # Stack group CLS tokens: (batch, 4, d_emb)
        group_tokens = torch.stack(group_cls_tokens, dim=1)

        # Expand global CLS: (1, 1, d_emb) → (batch, 1, d_emb)
        global_cls = self.global_cls.expand(batch_size, -1, -1)

        # Prepend global CLS: (batch, 5, d_emb)
        x = torch.cat([global_cls, group_tokens], dim=1)

        # Add positional embedding
        x = x + self.pos_embedding

        # Input normalization
        x = self.input_norm(x)

        # Process through transformer layers
        attn_weights_list: List[torch.Tensor] = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_list.append(attn_weights)

        # Final normalization
        x = self.final_norm(x)

        # Extract global CLS token (position 0)
        global_cls_output = x[:, 0, :]  # (batch, d_emb)

        return global_cls_output, attn_weights_list

    def get_group_representations(
        self,
        group_cls_tokens: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning both global CLS and refined group representations.
        Useful for advanced analysis.

        Returns:
            Tuple of:
                - global_cls: (batch, d_emb)
                - group_outputs: (batch, 4, d_emb) — refined group representations
                - attn_weights_list
        """
        batch_size = group_cls_tokens[0].shape[0]

        group_tokens = torch.stack(group_cls_tokens, dim=1)
        global_cls = self.global_cls.expand(batch_size, -1, -1)
        x = torch.cat([global_cls, group_tokens], dim=1)
        x = x + self.pos_embedding
        x = self.input_norm(x)

        attn_weights_list: List[torch.Tensor] = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_list.append(attn_weights)

        x = self.final_norm(x)

        global_cls_output = x[:, 0, :]  # (batch, d_emb)
        group_outputs = x[:, 1:, :]     # (batch, 4, d_emb)

        return global_cls_output, group_outputs, attn_weights_list


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing InterGroupTransformer...")

    batch_size = 8
    d_emb = 64
    nhead = 4
    num_layers = 2

    # Create dummy group CLS tokens (as if from 4 intra-group transformers)
    group_cls_tokens = [torch.randn(batch_size, d_emb) for _ in range(4)]

    transformer = InterGroupTransformer(
        d_emb=d_emb, nhead=nhead, num_layers=num_layers
    )

    global_cls, attn_weights = transformer(group_cls_tokens)

    assert global_cls.shape == (batch_size, d_emb), (
        f"Global CLS shape mismatch: {global_cls.shape}"
    )
    assert len(attn_weights) == num_layers
    for i, aw in enumerate(attn_weights):
        # 5 = 1 global_cls + 4 group_cls
        assert aw.shape == (batch_size, nhead, 5, 5), (
            f"Attn weight shape at layer {i}: {aw.shape}"
        )
    print(f"  global_cls: {global_cls.shape} ✓")
    print(f"  attn_weights: {len(attn_weights)} layers × {attn_weights[0].shape} ✓")

    # Test get_group_representations
    g_cls, g_out, g_attn = transformer.get_group_representations(group_cls_tokens)
    assert g_out.shape == (batch_size, 4, d_emb)
    print(f"  group_outputs: {g_out.shape} ✓")

    # Count parameters
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"\n  Parameters: {n_params:,}")
    print("\n✓ InterGroupTransformer test passed!")

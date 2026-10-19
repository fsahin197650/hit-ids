"""
HiT-IDS: Full Hierarchical Transformer model for Network Intrusion Detection.

Assembles all components:
    4× FeatureEmbedding → 4× IntraGroupTransformer → InterGroupTransformer
    → ClassificationHead + ExplanationHead

This is the main model class used for training and inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml

from .embedding import FeatureEmbedding
from .intra_group import IntraGroupTransformer
from .inter_group import InterGroupTransformer
from .heads import ClassificationHead, ExplanationHead


class HiTIDS(nn.Module):
    """
    Hierarchical Transformer with Intrinsic Explainability for IDS.

    Architecture:
        Input (batch, ~72 features)
          │
          ├─ G1 features → FeatureEmbedding₁ → IntraGroupTransformer₁ → CLS₁
          ├─ G2 features → FeatureEmbedding₂ → IntraGroupTransformer₂ → CLS₂
          ├─ G3 features → FeatureEmbedding₃ → IntraGroupTransformer₃ → CLS₃
          └─ G4 features → FeatureEmbedding₄ → IntraGroupTransformer₄ → CLS₄
                                                                          │
                                        [CLS₁, CLS₂, CLS₃, CLS₄]       │
                                                                          ▼
                                                          InterGroupTransformer
                                                                          │
                                                                    Global CLS
                                                                     │       │
                                                        ClassificationHead  ExplanationHead
                                                              │                │
                                                         Predictions    Feature Importance

    Args:
        group_sizes: List of feature counts per group, e.g., [19, 25, 15, 13].
        group_names: List of group names. If None, uses defaults.
        d_emb: Embedding dimension (default: 64).
        nhead: Number of attention heads (default: 4).
        intra_layers: Number of intra-group transformer layers (default: 2).
        inter_layers: Number of inter-group transformer layers (default: 2).
        dim_feedforward: FFN hidden dimension (default: 128).
        num_classes: Number of output classes (default: 15).
        dropout: Dropout rate (default: 0.1).
    """

    DEFAULT_GROUP_NAMES = ["temporal", "volume", "protocol", "behavioral"]

    def __init__(
        self,
        group_sizes: List[int],
        group_names: Optional[List[str]] = None,
        d_emb: int = 64,
        nhead: int = 4,
        intra_layers: int = 2,
        inter_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.group_sizes = group_sizes
        self.group_names = group_names or self.DEFAULT_GROUP_NAMES
        self.d_emb = d_emb
        self.nhead = nhead
        self.num_classes = num_classes
        self.num_groups = len(group_sizes)

        assert len(self.group_sizes) == len(self.group_names), (
            f"group_sizes ({len(group_sizes)}) and group_names "
            f"({len(self.group_names)}) must have same length"
        )

        # ── Per-group components ─────────────────────────────────────────
        self.embeddings = nn.ModuleDict({
            name: FeatureEmbedding(
                num_features=size, d_emb=d_emb, dropout=dropout
            )
            for name, size in zip(self.group_names, group_sizes)
        })

        self.intra_transformers = nn.ModuleDict({
            name: IntraGroupTransformer(
                d_emb=d_emb,
                nhead=nhead,
                num_layers=intra_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for name in self.group_names
        })

        # ── Inter-group component ────────────────────────────────────────
        self.inter_transformer = InterGroupTransformer(
            d_emb=d_emb,
            nhead=nhead,
            num_layers=inter_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # ── Output heads ─────────────────────────────────────────────────
        self.classification_head = ClassificationHead(
            d_emb=d_emb,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.explanation_head = ExplanationHead(
            group_names=list(self.group_names),
        )

    @classmethod
    def from_config(cls, config_path: str) -> "HiTIDS":
        """
        Create HiTIDS model from a YAML config file.

        Args:
            config_path: Path to config.yaml.

        Returns:
            HiTIDS model instance.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, config: dict) -> "HiTIDS":
        """
        Create HiTIDS model from a config dictionary.

        Args:
            config: Parsed YAML config dict.

        Returns:
            HiTIDS model instance.
        """
        model_cfg = config["model"]
        fg_cfg = config["feature_groups"]

        group_names = list(fg_cfg.keys())
        group_sizes = [len(fg_cfg[name]) for name in group_names]

        return cls(
            group_sizes=group_sizes,
            group_names=group_names,
            d_emb=model_cfg.get("d_emb", 64),
            nhead=model_cfg.get("nhead", 4),
            intra_layers=model_cfg.get("intra_group_layers", 2),
            inter_layers=model_cfg.get("inter_group_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 128),
            num_classes=config["dataset"].get("num_classes", 15),
            dropout=model_cfg.get("dropout", 0.1),
        )

    def forward(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        return_explanations: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through the full hierarchical model.

        Args:
            x: Either:
                - Dict mapping group_name → tensor of shape (batch, num_features_g)
                - Single tensor of shape (batch, total_features) — will be split
            return_explanations: If True, also return XAI importance scores.

        Returns:
            If return_explanations is False:
                logits: (batch, num_classes)
            If return_explanations is True:
                Tuple of (logits, explanations_dict)
        """
        # Handle input formats
        if isinstance(x, torch.Tensor):
            x = self._split_features(x)

        # ── Stage 1: Intra-group processing ──────────────────────────────
        group_cls_tokens: List[torch.Tensor] = []
        intra_attn_weights: Dict[str, List[torch.Tensor]] = {}

        for name in self.group_names:
            group_features = x[name]  # (batch, num_features_g)

            # Embed: (batch, num_features_g) → (batch, num_features_g + 1, d_emb)
            embedded = self.embeddings[name](group_features)

            # Intra-group transform: → cls (batch, d_emb), attn_weights
            cls_token, attn_weights = self.intra_transformers[name](embedded)

            group_cls_tokens.append(cls_token)
            intra_attn_weights[name] = attn_weights

        # ── Stage 2: Inter-group processing ──────────────────────────────
        global_cls, inter_attn_weights = self.inter_transformer(group_cls_tokens)

        # ── Stage 3: Classification ──────────────────────────────────────
        logits = self.classification_head(global_cls)

        # ── Stage 4: Optional explanation ────────────────────────────────
        if return_explanations:
            explanations = self.explanation_head(
                intra_attn_weights, inter_attn_weights
            )
            # Also include raw attention weights for advanced analysis
            explanations["raw_intra_attn"] = intra_attn_weights
            explanations["raw_inter_attn"] = inter_attn_weights
            return logits, explanations

        return logits

    def _split_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split a flat feature tensor into group-specific tensors.

        Args:
            x: Flat feature tensor, shape (batch, total_features).

        Returns:
            Dict mapping group_name → tensor (batch, num_features_g).
        """
        groups: Dict[str, torch.Tensor] = {}
        offset = 0
        for name, size in zip(self.group_names, self.group_sizes):
            groups[name] = x[:, offset:offset + size]
            offset += size
        return groups

    def get_attention_entropy(self) -> torch.Tensor:
        """
        Compute average attention entropy for regularization (Phase 2 training).

        Higher entropy = more uniform attention (less focused).
        Lower entropy = more peaked attention (more decisive).

        Returns:
            Scalar tensor — mean entropy across all attention heads & layers.
        """
        # This should be called after a forward pass with stored activations
        # For now, we compute it during the forward pass separately
        raise NotImplementedError(
            "Call forward() with return_explanations=True and compute "
            "entropy from the returned attention weights."
        )

    @property
    def total_features(self) -> int:
        """Total number of input features."""
        return sum(self.group_sizes)

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        counts = {
            "total": sum(p.numel() for p in self.parameters()),
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

        for name in self.group_names:
            counts[f"embedding_{name}"] = sum(
                p.numel() for p in self.embeddings[name].parameters()
            )
            counts[f"intra_transformer_{name}"] = sum(
                p.numel() for p in self.intra_transformers[name].parameters()
            )

        counts["inter_transformer"] = sum(
            p.numel() for p in self.inter_transformer.parameters()
        )
        counts["classification_head"] = sum(
            p.numel() for p in self.classification_head.parameters()
        )

        return counts

    def summary(self) -> str:
        """Human-readable model summary."""
        counts = self.count_parameters()
        lines = [
            "=" * 60,
            "HiT-IDS Model Summary",
            "=" * 60,
            f"Groups: {self.group_names}",
            f"Group sizes: {self.group_sizes}",
            f"Total features: {self.total_features}",
            f"d_emb: {self.d_emb}, nhead: {self.nhead}",
            f"Num classes: {self.num_classes}",
            "",
            "Parameter counts:",
        ]
        for key, count in counts.items():
            lines.append(f"  {key:30s}: {count:>10,}")
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_attention_entropy(
    attention_weights: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute attention entropy for regularization.

    L_entropy = -mean(sum(A * log(A + ε)))

    Low entropy → peaked attention (good for XAI).
    Used in Phase 2 fine-tuning to encourage focused attention.

    Args:
        attention_weights: List of attention tensors per layer,
            each shape (batch, nhead, seq_len, seq_len).

    Returns:
        Scalar mean entropy.
    """
    eps = 1e-12
    total_entropy = 0.0
    count = 0

    for attn in attention_weights:
        # attn: (batch, nhead, seq, seq)
        # Entropy per row: -sum(a * log(a))
        entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)  # (batch, nhead, seq)
        total_entropy += entropy.mean()
        count += 1

    return total_entropy / max(count, 1)


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing HiTIDS full model...")

    batch_size = 8
    group_sizes = [19, 25, 15, 13]  # temporal, volume, protocol, behavioral
    total_features = sum(group_sizes)

    model = HiTIDS(
        group_sizes=group_sizes,
        d_emb=64,
        nhead=4,
        intra_layers=2,
        inter_layers=2,
        dim_feedforward=128,
        num_classes=15,
        dropout=0.1,
    )
    print(model.summary())

    # Test with dict input
    x_dict = {
        "temporal": torch.randn(batch_size, 19),
        "volume": torch.randn(batch_size, 25),
        "protocol": torch.randn(batch_size, 15),
        "behavioral": torch.randn(batch_size, 13),
    }

    # Forward without explanations
    logits = model(x_dict, return_explanations=False)
    assert logits.shape == (batch_size, 15)
    print(f"\nLogits shape: {logits.shape} ✓")

    # Forward with explanations
    logits, explanations = model(x_dict, return_explanations=True)
    assert logits.shape == (batch_size, 15)
    print(f"Logits shape (with XAI): {logits.shape} ✓")
    print(f"Group importance: {explanations['group_importance'].shape} ✓")

    for name, size in zip(model.group_names, group_sizes):
        gi = explanations["global_importance"][name]
        print(f"  {name}: global_importance shape = {gi.shape} ✓")

    # Test with flat tensor input
    x_flat = torch.randn(batch_size, total_features)
    logits_flat = model(x_flat)
    assert logits_flat.shape == (batch_size, 15)
    print(f"\nFlat tensor input: {x_flat.shape} → {logits_flat.shape} ✓")

    # Test attention entropy
    _, expl = model(x_dict, return_explanations=True)
    all_attn = []
    for name, attn_list in expl["raw_intra_attn"].items():
        all_attn.extend(attn_list)
    all_attn.extend(expl["raw_inter_attn"])
    entropy = compute_attention_entropy(all_attn)
    print(f"\nAttention entropy: {entropy.item():.4f}")

    # Test from config
    import os
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "config.yaml"
    )
    if os.path.exists(config_path):
        model2 = HiTIDS.from_config(config_path)
        print(f"\nFrom config: {model2.group_sizes}")
        logits2 = model2(x_dict)
        print(f"Config model logits: {logits2.shape} ✓")

    print("\n✓ All HiTIDS tests passed!")

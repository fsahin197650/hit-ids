"""
Baseline models for comparison with HiT-IDS.

Wraps sklearn and PyTorch models in a unified interface:
    - Random Forest
    - XGBoost
    - 1D-CNN
    - BiLSTM
    - Vanilla Transformer (no hierarchy)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ═══════════════════════════════════════════════════════════════════════════
# Sklearn-based Baselines
# ═══════════════════════════════════════════════════════════════════════════


class SklearnBaseline:
    """Wrapper for sklearn classifiers (RF, XGBoost)."""

    def __init__(self, model_type: str = "random_forest", **kwargs: Any) -> None:
        """
        Args:
            model_type: 'random_forest' or 'xgboost'.
            **kwargs: Model-specific hyperparameters.
        """
        self.model_type = model_type

        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1,
                class_weight="balanced",
            )
        elif model_type == "xgboost":
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 8),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                tree_method="hist",
            )
        else:
            raise ValueError(f"Unknown sklearn model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        print(f"  Training {self.model_type}...")
        start = time.time()
        self.model.fit(X, y)
        print(f"  Training time: {time.time()-start:.1f}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Get feature importances (for tree-based models)."""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch-based Baselines
# ═══════════════════════════════════════════════════════════════════════════


class CNN1D(nn.Module):
    """
    1D-CNN baseline for IDS.

    Architecture:
        Conv1D → BN → ReLU → MaxPool → Conv1D → BN → ReLU → GlobalAvgPool → FC
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 15,
        channels: List[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256]

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[2], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features)
        Returns:
            logits: (batch, num_classes)
        """
        x = x.unsqueeze(1)  # (batch, 1, num_features) — single channel
        x = self.features(x)
        return self.classifier(x)


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM baseline for IDS.

    Architecture:
        Embedding → BiLSTM → Attention Pooling → FC
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.proj = nn.Linear(1, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features)
        Returns:
            logits: (batch, num_classes)
        """
        x = x.unsqueeze(-1)  # (batch, num_features, 1)
        x = self.proj(x)  # (batch, num_features, hidden_size)

        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        return self.classifier(context)


class VanillaTransformer(nn.Module):
    """
    Vanilla (flat) Transformer baseline — no hierarchical grouping.

    This is the ablation baseline: same total parameters, but all features
    processed in a single flat transformer.

    Architecture:
        Linear(1, d_emb) per feature → [CLS] + tokens → TransformerEncoder → FC
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 15,
        d_emb: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_emb = d_emb
        self.num_features = num_features

        # Feature projection (same as HiT-IDS embedding)
        self.feature_proj = nn.Linear(1, d_emb)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_emb) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_features + 1, d_emb) * 0.02
        )
        self.layer_norm = nn.LayerNorm(d_emb)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.shape[0]

        # (batch, num_features, 1) → (batch, num_features, d_emb)
        x = x.unsqueeze(-1)
        x = self.feature_proj(x)

        # Prepend CLS
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embedding
        x = self.layer_norm(x)

        # Transformer
        x = self.transformer(x)

        # CLS token → classify
        cls_out = x[:, 0, :]
        return self.classifier(cls_out)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Factory
# ═══════════════════════════════════════════════════════════════════════════


def create_baseline(
    model_type: str,
    num_features: int,
    num_classes: int = 15,
    **kwargs: Any,
) -> Any:
    """
    Factory function to create a baseline model.

    Args:
        model_type: One of 'random_forest', 'xgboost', 'cnn_1d', 'bilstm',
                    'vanilla_transformer'.
        num_features: Total number of input features.
        num_classes: Number of output classes.

    Returns:
        Baseline model instance.
    """
    if model_type in ("random_forest", "xgboost"):
        return SklearnBaseline(model_type=model_type, **kwargs)
    elif model_type == "cnn_1d":
        return CNN1D(num_features=num_features, num_classes=num_classes, **kwargs)
    elif model_type == "bilstm":
        return BiLSTMClassifier(num_features=num_features, num_classes=num_classes, **kwargs)
    elif model_type == "vanilla_transformer":
        return VanillaTransformer(num_features=num_features, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown baseline model: {model_type}")


def train_pytorch_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Train a PyTorch baseline model with standard training loop.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader (flat tensors, not grouped).
        val_loader: Validation DataLoader.
        num_epochs: Max training epochs.
        lr: Learning rate.
        device: Device to train on.

    Returns:
        Dict with training history and best metrics.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None

    from sklearn.metrics import f1_score

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_targets.extend(y_batch.tolist())

        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"loss={total_loss/len(train_loader):.4f}, val_f1={val_f1:.4f}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    return {"best_f1": best_f1}


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing baseline models...")

    batch_size = 16
    num_features = 72
    num_classes = 15

    x = torch.randn(batch_size, num_features)

    # Test CNN1D
    cnn = CNN1D(num_features=num_features, num_classes=num_classes)
    out = cnn(x)
    assert out.shape == (batch_size, num_classes)
    n_params = sum(p.numel() for p in cnn.parameters())
    print(f"  CNN1D: output={out.shape}, params={n_params:,} ✓")

    # Test BiLSTM
    lstm = BiLSTMClassifier(num_features=num_features, num_classes=num_classes)
    out = lstm(x)
    assert out.shape == (batch_size, num_classes)
    n_params = sum(p.numel() for p in lstm.parameters())
    print(f"  BiLSTM: output={out.shape}, params={n_params:,} ✓")

    # Test VanillaTransformer
    vt = VanillaTransformer(num_features=num_features, num_classes=num_classes)
    out = vt(x)
    assert out.shape == (batch_size, num_classes)
    n_params = sum(p.numel() for p in vt.parameters())
    print(f"  VanillaTransformer: output={out.shape}, params={n_params:,} ✓")

    # Test factory
    for model_type in ["cnn_1d", "bilstm", "vanilla_transformer"]:
        m = create_baseline(model_type, num_features, num_classes)
        print(f"  Factory({model_type}): {type(m).__name__} ✓")

    print("\n✓ All baseline tests passed!")

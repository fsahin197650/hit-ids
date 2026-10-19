"""
Two-phase trainer for HiT-IDS.

Phase 1 — Classification: Focal loss, standard training loop.
Phase 2 — Explanation Fine-tuning: Lower LR + attention entropy regularization.

Includes W&B logging, gradient clipping, and learning rate scheduling.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.hit_ids import HiTIDS, compute_attention_entropy
from .callbacks import EarlyStopping, ModelCheckpoint
from .focal_loss import FocalLoss, compute_class_weights


class HiTIDSTrainer:
    """
    Two-phase trainer for HiT-IDS model.

    Phase 1 — Classification Training:
        - Focal loss with class weights
        - AdamW optimizer, CosineAnnealingWarmRestarts scheduler
        - Linear warmup for first N epochs
        - Save best model by validation macro-F1

    Phase 2 — Explanation Fine-tuning:
        - Lower LR (10x reduction)
        - Additional loss: L_total = L_focal + λ × L_entropy
        - Attention entropy penalty encourages focused attention patterns
        - Shorter training duration

    Args:
        model: HiTIDS model instance.
        config: Full config dict from YAML.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model: HiTIDS,
        config: dict,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training configs
        self.phase1_cfg = config["training"]["phase1"]
        self.phase2_cfg = config["training"]["phase2"]
        self.focal_cfg = config["training"]["focal_loss"]

        # Paths
        self.checkpoint_dir = config["paths"]["checkpoints"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # W&B
        self.wandb_enabled = config.get("wandb", {}).get("enabled", False)
        self.wandb_run = None

        # Training state
        self.current_epoch = 0
        self.current_phase = 0
        self.best_metrics: Dict[str, float] = {}
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "train_acc": [], "val_acc": [],
            "learning_rate": [],
        }

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        if not self.wandb_enabled:
            return
        try:
            import wandb
            wandb_cfg = self.config.get("wandb", {})
            self.wandb_run = wandb.init(
                project=wandb_cfg.get("project", "HiT-IDS"),
                entity=wandb_cfg.get("entity", None),
                tags=wandb_cfg.get("tags", []),
                config={
                    "model": self.config["model"],
                    "training": self.config["training"],
                    "dataset": self.config.get("dataset", {}),
                },
                name=f"hit-ids-{time.strftime('%Y%m%d_%H%M%S')}",
            )
            wandb.watch(self.model, log="gradients", log_freq=100)
            print("✓ W&B initialized")
        except Exception as e:
            print(f"[WARNING] W&B init failed: {e}. Continuing without W&B.")
            self.wandb_enabled = False

    def _setup_criterion(self, train_loader: DataLoader) -> FocalLoss:
        """Set up focal loss with class weights from training data."""
        # Collect all labels to compute class weights
        all_labels = []
        for _, labels in train_loader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels)

        num_classes = self.config["dataset"]["num_classes"]

        if self.focal_cfg["alpha"] == "auto":
            alpha = compute_class_weights(
                all_labels, num_classes=num_classes, method="inverse_freq"
            )
            print(f"  Auto class weights: {alpha.tolist()[:5]}...")
        elif isinstance(self.focal_cfg["alpha"], list):
            alpha = torch.tensor(self.focal_cfg["alpha"], dtype=torch.float32)
        else:
            alpha = None

        criterion = FocalLoss(
            alpha=alpha,
            gamma=self.focal_cfg["gamma"],
            label_smoothing=self.focal_cfg.get("label_smoothing", 0.0),
        )

        return criterion

    def _setup_optimizer_phase1(self) -> Tuple[torch.optim.Optimizer, Any]:
        """Set up optimizer and scheduler for Phase 1."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.phase1_cfg["learning_rate"],
            weight_decay=self.phase1_cfg["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.phase1_cfg["scheduler_T0"],
            T_mult=self.phase1_cfg["scheduler_Tmult"],
        )

        return optimizer, scheduler

    def _setup_optimizer_phase2(self) -> Tuple[torch.optim.Optimizer, Any]:
        """Set up optimizer and scheduler for Phase 2 (lower LR)."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.phase2_cfg["learning_rate"],
            weight_decay=self.phase1_cfg["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.phase2_cfg["max_epochs"],
        )

        return optimizer, scheduler

    def _get_warmup_lr(self, epoch: int, base_lr: float) -> float:
        """Linear warmup LR scaling."""
        warmup = self.phase1_cfg["warmup_epochs"]
        if epoch < warmup:
            return base_lr * (epoch + 1) / warmup
        return base_lr

    # ── Training Loops ───────────────────────────────────────────────────

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: FocalLoss,
        optimizer: torch.optim.Optimizer,
        phase: int = 1,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader.
            criterion: Loss function.
            optimizer: Optimizer.
            phase: 1 (classification) or 2 (explanation fine-tuning).

        Returns:
            Dict with training metrics (loss, accuracy, f1).
        """
        self.model.train()
        total_loss = 0.0
        total_entropy_loss = 0.0
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        pbar = tqdm(train_loader, desc=f"  Train (Phase {phase})", leave=False)

        for batch_groups, batch_labels in pbar:
            # Move to device
            batch_groups = {
                k: v.to(self.device) for k, v in batch_groups.items()
            }
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            if phase == 2:
                logits, explanations = self.model(
                    batch_groups, return_explanations=True
                )
            else:
                logits = self.model(batch_groups, return_explanations=False)

            # Classification loss
            loss = criterion(logits, batch_labels)

            # Phase 2: Add attention entropy regularization
            if phase == 2:
                all_attn = []
                for attn_list in explanations["raw_intra_attn"].values():
                    all_attn.extend(attn_list)
                all_attn.extend(explanations["raw_inter_attn"])
                entropy_loss = compute_attention_entropy(all_attn)
                lam = self.phase2_cfg["attention_entropy_lambda"]
                loss = loss + lam * entropy_loss
                total_entropy_loss += entropy_loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.phase1_cfg["gradient_clip_max_norm"],
            )

            optimizer.step()

            # Metrics
            total_loss += loss.item() * batch_labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch_labels.cpu().tolist())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct/total:.4f}",
            )

        # Compute metrics
        from sklearn.metrics import f1_score
        avg_loss = total_loss / total
        accuracy = correct / total
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
        if phase == 2:
            metrics["entropy_loss"] = total_entropy_loss / len(train_loader)

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        criterion: FocalLoss,
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation DataLoader.
            criterion: Loss function.

        Returns:
            Dict with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for batch_groups, batch_labels in val_loader:
            batch_groups = {
                k: v.to(self.device) for k, v in batch_groups.items()
            }
            batch_labels = batch_labels.to(self.device)

            logits = self.model(batch_groups, return_explanations=False)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item() * batch_labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch_labels.cpu().tolist())

        from sklearn.metrics import f1_score
        avg_loss = total_loss / total
        accuracy = correct / total
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

    # ── Full Training ────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Run full 2-phase training.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            Dict with training history and best metrics.
        """
        self._init_wandb()

        criterion = self._setup_criterion(train_loader)

        print("\n" + "=" * 60)
        print("PHASE 1: Classification Training")
        print("=" * 60)
        self._train_phase(
            train_loader, val_loader, criterion,
            phase=1,
        )

        print("\n" + "=" * 60)
        print("PHASE 2: Explanation Fine-tuning")
        print("=" * 60)

        # Load best Phase 1 model
        checkpoint_path = os.path.join(
            self.checkpoint_dir, "hit_ids_phase1_best.pt"
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded best Phase 1 model (epoch {checkpoint['epoch']}, "
                  f"F1={checkpoint['score']:.4f})")

        self._train_phase(
            train_loader, val_loader, criterion,
            phase=2,
        )

        if self.wandb_run:
            import wandb
            wandb.finish()

        return {
            "history": self.history,
            "best_metrics": self.best_metrics,
        }

    def _train_phase(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: FocalLoss,
        phase: int,
    ) -> None:
        """Run one training phase."""
        if phase == 1:
            optimizer, scheduler = self._setup_optimizer_phase1()
            max_epochs = self.phase1_cfg["max_epochs"]
            patience = self.phase1_cfg["early_stopping_patience"]
        else:
            optimizer, scheduler = self._setup_optimizer_phase2()
            max_epochs = self.phase2_cfg["max_epochs"]
            patience = self.phase2_cfg["early_stopping_patience"]

        early_stopping = EarlyStopping(patience=patience, mode="max")
        checkpoint = ModelCheckpoint(
            save_dir=self.checkpoint_dir,
            filename_prefix=f"hit_ids_phase{phase}",
            mode="max",
        )

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            self.current_phase = phase

            # Warmup (Phase 1 only)
            if phase == 1 and epoch < self.phase1_cfg["warmup_epochs"]:
                warmup_lr = self._get_warmup_lr(
                    epoch, self.phase1_cfg["learning_rate"]
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            print(f"\nEpoch {epoch+1}/{max_epochs} "
                  f"(Phase {phase}, LR={optimizer.param_groups[0]['lr']:.2e})")

            # Train
            train_metrics = self.train_epoch(
                train_loader, criterion, optimizer, phase=phase
            )

            # Validate
            val_metrics = self.validate(val_loader, criterion)

            # Log
            print(f"  Train — Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['macro_f1']:.4f}")
            print(f"  Val   — Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['macro_f1']:.4f}")

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_f1"].append(train_metrics["macro_f1"])
            self.history["val_f1"].append(val_metrics["macro_f1"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["learning_rate"].append(
                optimizer.param_groups[0]["lr"]
            )

            # W&B logging
            if self.wandb_enabled:
                try:
                    import wandb
                    log_dict = {
                        f"phase{phase}/train_loss": train_metrics["loss"],
                        f"phase{phase}/val_loss": val_metrics["loss"],
                        f"phase{phase}/train_f1": train_metrics["macro_f1"],
                        f"phase{phase}/val_f1": val_metrics["macro_f1"],
                        f"phase{phase}/train_acc": train_metrics["accuracy"],
                        f"phase{phase}/val_acc": val_metrics["accuracy"],
                        f"phase{phase}/lr": optimizer.param_groups[0]["lr"],
                    }
                    if "entropy_loss" in train_metrics:
                        log_dict[f"phase{phase}/entropy_loss"] = train_metrics["entropy_loss"]
                    wandb.log(log_dict)
                except Exception:
                    pass

            # Checkpoint
            checkpoint(
                self.model, optimizer, epoch, val_metrics["macro_f1"],
                extra={"phase": phase, "config": self.config},
            )

            # Step scheduler (after warmup)
            if phase == 1 and epoch >= self.phase1_cfg["warmup_epochs"]:
                scheduler.step()
            elif phase == 2:
                scheduler.step()

            # Early stopping
            if early_stopping(val_metrics["macro_f1"], epoch):
                break

        # Update best metrics
        self.best_metrics[f"phase{phase}_best_f1"] = early_stopping.best_score
        self.best_metrics[f"phase{phase}_best_epoch"] = early_stopping.best_epoch
        print(f"\n  Phase {phase} complete — Best F1: "
              f"{early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing HiTIDSTrainer (minimal)...")

    # Create a small model
    model = HiTIDS(
        group_sizes=[4, 5, 3, 3],
        d_emb=32,
        nhead=2,
        intra_layers=1,
        inter_layers=1,
        dim_feedforward=64,
        num_classes=5,
        dropout=0.1,
    )

    # Minimal config
    config = {
        "model": {"d_emb": 32, "nhead": 2},
        "dataset": {"num_classes": 5},
        "paths": {"checkpoints": "/tmp/hit_ids_test_ckpt"},
        "training": {
            "phase1": {
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "max_epochs": 2,
                "warmup_epochs": 1,
                "scheduler_T0": 5,
                "scheduler_Tmult": 2,
                "early_stopping_patience": 3,
                "gradient_clip_max_norm": 1.0,
                "batch_size": 16,
            },
            "phase2": {
                "learning_rate": 1e-4,
                "max_epochs": 1,
                "early_stopping_patience": 2,
                "attention_entropy_lambda": 0.01,
            },
            "focal_loss": {"gamma": 2.0, "alpha": "auto"},
        },
        "wandb": {"enabled": False},
    }

    # Create dummy dataloader
    from src.data.dataset import HiTIDSDataset, collate_fn

    N = 100
    group_data = {
        "temporal": torch.randn(N, 4),
        "volume": torch.randn(N, 5),
        "protocol": torch.randn(N, 3),
        "behavioral": torch.randn(N, 3),
    }
    X = torch.cat(list(group_data.values()), dim=1)
    y = torch.randint(0, 5, (N,))
    dataset = HiTIDSDataset(X=X, y=y, group_data=group_data)

    train_loader = DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    trainer = HiTIDSTrainer(model=model, config=config, device="cpu")
    results = trainer.train(train_loader, val_loader)

    print(f"\nTraining history keys: {list(results['history'].keys())}")
    print(f"Best metrics: {results['best_metrics']}")
    print("\n✓ Trainer test passed!")

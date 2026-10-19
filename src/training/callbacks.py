"""
Callbacks for HiT-IDS training.

Provides EarlyStopping and ModelCheckpoint callbacks.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' (loss) or 'max' (accuracy/f1).
        verbose: Print messages on events.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

        # Set comparison function
        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        elif mode == "max":
            self.is_better = lambda score, best: score > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} "
                      f"(best={self.best_score:.4f} at epoch {self.best_epoch})")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: TRIGGERED at epoch {epoch}")
                return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state (for phase transitions)."""
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Save model checkpoints when validation metric improves.

    Args:
        save_dir: Directory to save checkpoints.
        filename_prefix: Prefix for checkpoint files.
        mode: 'min' (loss) or 'max' (accuracy/f1).
        save_best_only: Only save when metric improves.
        verbose: Print messages on events.
    """

    def __init__(
        self,
        save_dir: str,
        filename_prefix: str = "hit_ids",
        mode: str = "max",
        save_best_only: bool = True,
        verbose: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_score: Optional[float] = None

        if mode == "min":
            self.is_better = lambda score, best: score < best
        elif mode == "max":
            self.is_better = lambda score, best: score > best
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        extra: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Save checkpoint if metric improved.

        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            epoch: Current epoch.
            score: Current validation metric.
            extra: Additional items to save.

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        should_save = True

        if self.save_best_only:
            if self.best_score is None:
                self.best_score = score
            elif self.is_better(score, self.best_score):
                self.best_score = score
            else:
                should_save = False

        if should_save:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": score,
            }
            if extra:
                checkpoint.update(extra)

            # Save best model
            best_path = self.save_dir / f"{self.filename_prefix}_best.pt"
            torch.save(checkpoint, best_path)

            if not self.save_best_only:
                epoch_path = self.save_dir / f"{self.filename_prefix}_epoch{epoch}.pt"
                torch.save(checkpoint, epoch_path)

            if self.verbose:
                print(f"  Checkpoint saved: {best_path} (score={score:.4f})")

            return str(best_path)

        return None

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> dict:
        """
        Load the best saved checkpoint.

        Args:
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state.
            device: Device to load onto.

        Returns:
            Checkpoint dict with epoch and score info.
        """
        best_path = self.save_dir / f"{self.filename_prefix}_best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {best_path}")

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.verbose:
            print(f"  Loaded checkpoint: epoch={checkpoint['epoch']}, "
                  f"score={checkpoint['score']:.4f}")

        return checkpoint


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("Testing EarlyStopping...")

    es = EarlyStopping(patience=3, mode="max", verbose=True)

    # Simulate improving then stagnating metrics
    scores = [0.5, 0.6, 0.7, 0.65, 0.68, 0.69, 0.68]
    for epoch, score in enumerate(scores):
        stopped = es(score, epoch)
        if stopped:
            print(f"  Stopped at epoch {epoch}")
            break

    print(f"  Best score: {es.best_score}, Best epoch: {es.best_epoch}")

    print("\nTesting ModelCheckpoint...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt = ModelCheckpoint(save_dir=tmpdir, mode="max")
        ckpt(model, optimizer, epoch=0, score=0.5)
        ckpt(model, optimizer, epoch=1, score=0.7)
        ckpt(model, optimizer, epoch=2, score=0.6)  # Not saved

        loaded = ckpt.load_best(model, device="cpu")
        print(f"  Loaded best: epoch={loaded['epoch']}, score={loaded['score']:.4f}")

    print("\n✓ All callback tests passed!")

#!/usr/bin/env python3
"""
HiT-IDS Training Entry Point.

Runs the full 2-phase training pipeline:
    Phase 1: Classification with Focal Loss
    Phase 2: Explanation Fine-tuning with attention entropy regularization

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --phase 1  # Only Phase 1
    python scripts/train.py --dry-run  # Quick test with dummy data
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn, create_dataloaders
from src.models.hit_ids import HiTIDS
from src.training.trainer import HiTIDSTrainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dummy_training_data(config: dict) -> dict:
    """Create dummy data for dry-run testing."""
    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    num_classes = config["dataset"]["num_classes"]

    N_train, N_val = 500, 100
    loaders = {}

    for split_name, N in [("train", N_train), ("val", N_val), ("test", N_val)]:
        group_data = {
            name: torch.randn(N, size)
            for name, size in zip(group_names, group_sizes)
        }
        X = torch.cat(list(group_data.values()), dim=1)
        y = torch.randint(0, num_classes, (N,))
        dataset = HiTIDSDataset(X=X, y=y, group_data=group_data)

        batch_size = config["training"]["phase1"]["batch_size"]
        loaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(batch_size, N),
            shuffle=(split_name == "train"),
            collate_fn=collate_fn,
            num_workers=0,
        )

    return loaders


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="HiT-IDS Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Run only Phase 1 or Phase 2 (default: both)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick test with dummy data",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (cuda/cpu)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override settings
    if args.no_wandb:
        config["wandb"]["enabled"] = False
    if args.device:
        config["project"]["device"] = args.device

    # Set device — also check CUDA compute capability
    device = config["project"].get("device", "cuda")
    if device == "cuda":
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            # Check if GPU architecture is supported
            try:
                cap = torch.cuda.get_device_capability(0)
                # Test with a small tensor to verify CUDA actually works
                _test = torch.zeros(1, device="cuda")
                del _test
            except Exception as e:
                print(f"[WARNING] CUDA available but not functional: {e}")
                print("  Falling back to CPU (RTX 5090 sm_120 may need newer PyTorch)")
                device = "cpu"

    # Set seed
    seed = config["project"].get("seed", 42)
    set_seed(seed)

    print("=" * 60)
    print("  HiT-IDS Training Pipeline")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  Seed: {seed}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Phase: {args.phase or 'Both (1→2)'}")
    if device == "cuda":
        try:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception:
            pass
    print("=" * 60)

    # Create model
    model = HiTIDS.from_dict(config)
    print(f"\n{model.summary()}")

    # Load data
    if args.dry_run:
        print("\n[DRY RUN] Creating dummy data...")
        loaders = create_dummy_training_data(config)
    else:
        splits_dir = config["paths"]["splits"]
        print(f"\nLoading data from {splits_dir}...")
        loaders = create_dataloaders(
            splits_dir=splits_dir,
            batch_size=config["training"]["phase1"]["batch_size"],
            num_workers=4,
            use_weighted_sampler=True,
        )

    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches: {len(loaders['val'])}")

    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded epoch {checkpoint['epoch']}, score={checkpoint['score']:.4f}")

    # Train
    trainer = HiTIDSTrainer(model=model, config=config, device=device)

    start_time = time.time()
    results = trainer.train(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
    )
    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} min")
    for key, value in results["best_metrics"].items():
        print(f"  {key}: {value}")
    print(f"\n  Checkpoints saved to: {config['paths']['checkpoints']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

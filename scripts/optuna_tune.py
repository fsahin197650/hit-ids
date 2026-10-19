#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for HiT-IDS.

Performs Bayesian optimization over model architecture and training
hyperparameters. Uses MedianPruner to kill bad trials early.

Usage:
    python scripts/optuna_tune.py --config configs/config.yaml --n-trials 30
    python scripts/optuna_tune.py --config configs/config.yaml --n-trials 10 --max-epochs 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.models.hit_ids import HiTIDS
from src.training.focal_loss import FocalLoss
from sklearn.metrics import f1_score


def load_data(config: dict, batch_size: int) -> dict:
    """Load preprocessed split data."""
    splits_dir = config["paths"]["splits"]
    loaders = {}

    for split_name in ["train", "val"]:
        split_data = torch.load(
            os.path.join(splits_dir, f"{split_name}.pt"),
            weights_only=False,
        )
        dataset = HiTIDSDataset(
            X=split_data["X"],
            y=split_data["y"],
            group_data=split_data["groups"],
        )

        # Weighted sampler for training
        if split_name == "train":
            class_counts = torch.bincount(split_data["y"])
            weights = 1.0 / class_counts.float()
            sample_weights = weights[split_data["y"]]
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )
            loaders[split_name] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                collate_fn=collate_fn, num_workers=4, pin_memory=True,
            )
        else:
            loaders[split_name] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=4, pin_memory=True,
            )

    return loaders


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        groups_dict, y = batch
        groups = {k: v.to(device) for k, v in groups_dict.items()}
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass — build flat X from groups
        X = torch.cat(list(groups.values()), dim=1)
        logits = model(X, return_explanations=False)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    """Evaluate model, return macro-F1."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        groups_dict, y = batch
        X = torch.cat(list(groups_dict.values()), dim=1).to(device)

        logits = model(X, return_explanations=False)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return f1_score(all_labels, all_preds, average="macro", zero_division=0)


def objective(trial: optuna.Trial, config: dict, args) -> float:
    """Optuna objective: returns validation macro-F1."""
    device = config["project"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ── Hyperparameters to search ──
    d_emb = trial.suggest_categorical("d_emb", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    intra_layers = trial.suggest_int("intra_layers", 2, 4)
    inter_layers = trial.suggest_int("inter_layers", 2, 4)
    dim_ff = trial.suggest_categorical("dim_feedforward", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.05, 0.3, step=0.05)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    gamma = trial.suggest_float("focal_gamma", 1.0, 5.0, step=0.5)

    # Ensure d_emb is divisible by nhead
    if d_emb % nhead != 0:
        d_emb = (d_emb // nhead) * nhead

    # ── Build model ──
    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    num_classes = config["dataset"]["num_classes"]

    model = HiTIDS(
        group_sizes=group_sizes,
        d_emb=d_emb,
        nhead=nhead,
        intra_layers=intra_layers,
        inter_layers=inter_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("n_params", n_params)

    # ── Load data ──
    loaders = load_data(config, batch_size)

    # ── Loss + Optimizer ──
    # Compute class weights
    train_data = torch.load(
        os.path.join(config["paths"]["splits"], "train.pt"),
        weights_only=False,
    )
    class_counts = torch.bincount(train_data["y"], minlength=num_classes).float()
    alpha_weights = (class_counts.sum() / (num_classes * class_counts)).clamp(max=10.0)
    del train_data

    criterion = FocalLoss(
        gamma=gamma,
        alpha=alpha_weights.to(device),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # ── Training loop ──
    max_epochs = args.max_epochs
    best_f1 = 0.0
    patience_counter = 0
    patience = 8  # Tighter for Optuna

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_f1 = evaluate(model, loaders["val"], device)
        scheduler.step()

        # Report to Optuna for pruning
        trial.report(val_f1, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{max_epochs}: loss={train_loss:.4f}, "
                  f"val_f1={val_f1:.4f} (best={best_f1:.4f})")

    # Cleanup GPU memory
    del model, optimizer, criterion, loaders
    gc.collect()
    torch.cuda.empty_cache()

    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for HiT-IDS")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials")
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Max epochs per trial (shorter for speed)")
    parser.add_argument("--study-name", default="hit_ids_hpo")
    parser.add_argument("--db", default=None,
                        help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  HiT-IDS Hyperparameter Optimization (Optuna)")
    print("=" * 60)
    print(f"  Trials: {args.n_trials}")
    print(f"  Max epochs per trial: {args.max_epochs}")
    print(f"  Study: {args.study_name}")
    print("=" * 60)

    # Create study
    storage = args.db or f"sqlite:///results/{args.study_name}.db"
    os.makedirs("results", exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",  # maximize macro-F1
        sampler=TPESampler(seed=config["project"].get("seed", 42)),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=storage,
        load_if_exists=True,
    )

    def obj_wrapper(trial):
        return objective(trial, config, args)

    start_time = time.time()
    study.optimize(
        obj_wrapper,
        n_trials=args.n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    elapsed = time.time() - start_time

    # ── Results ──
    print("\n" + "=" * 60)
    print("  Optuna Optimization Complete!")
    print("=" * 60)
    print(f"  Total time: {elapsed/3600:.1f} hours")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best macro-F1: {study.best_value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save best params
    best_config = {
        "best_f1": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "n_params": study.best_trial.user_attrs.get("n_params", "N/A"),
    }

    out_path = "results/optuna_best_params.json"
    with open(out_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  Best params saved to: {out_path}")

    # Print command to retrain with best params
    bp = study.best_params
    print(f"\n  To apply best params, update config.yaml:")
    print(f"    model.d_emb: {bp.get('d_emb', 128)}")
    print(f"    model.nhead: {bp.get('nhead', 8)}")
    print(f"    model.intra_group_layers: {bp.get('intra_layers', 3)}")
    print(f"    model.inter_group_layers: {bp.get('inter_layers', 3)}")
    print(f"    model.dim_feedforward: {bp.get('dim_feedforward', 256)}")
    print(f"    model.dropout: {bp.get('dropout', 0.15)}")
    print(f"    training.phase1.learning_rate: {bp.get('learning_rate', 3e-4)}")
    print(f"    training.phase1.weight_decay: {bp.get('weight_decay', 1e-4)}")
    print(f"    training.phase1.batch_size: {bp.get('batch_size', 512)}")
    print(f"    training.focal_loss.gamma: {bp.get('focal_gamma', 3.0)}")

    print("\n  Then retrain with full epochs:")
    print("    python scripts/train.py --config configs/config.yaml --no-wandb")
    print("=" * 60)


if __name__ == "__main__":
    main()

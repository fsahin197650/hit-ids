#!/usr/bin/env python3
"""
Cascade Training for HiT-IDS: Binary → Multi-class.

Strategy:
    1. Train Model A: BENIGN vs ATTACK (binary, easy task → F1 ~0.99)
    2. Train Model B: 14 attack types (multi-class on attacks only → F1 ~0.95)
    3. At inference: Model A filters → Model B classifies

This cascade approach eliminates the 83% BENIGN dominance problem.

Usage:
    python scripts/train_cascade.py --config configs/config.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
import yaml

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.models.hit_ids import HiTIDS
from src.training.focal_loss import FocalLoss, compute_class_weights


def load_split(splits_dir: str, split_name: str):
    """Load a data split."""
    return torch.load(
        os.path.join(splits_dir, f"{split_name}.pt"),
        weights_only=False,
    )


def make_binary_labels(y: torch.Tensor) -> torch.Tensor:
    """Convert multi-class labels to binary: 0=BENIGN, 1=ATTACK."""
    return (y > 0).long()


def create_dataloader(X, y, groups, batch_size, shuffle=True, weighted=False,
                      rare_boost_factor=1.0, rare_threshold=500):
    """Create a DataLoader with optional weighted sampling and rare-class boost."""
    dataset = HiTIDSDataset(X=X, y=y, group_data=groups)

    if weighted and shuffle:
        class_counts = torch.bincount(y)
        weights = 1.0 / class_counts.float().clamp(min=1)
        # Apply rare-class boost
        if rare_boost_factor > 1.0:
            for cls_idx in range(len(class_counts)):
                if class_counts[cls_idx] < rare_threshold:
                    weights[cls_idx] *= rare_boost_factor
        sample_weights = weights[y]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )


def apply_smote_attack(X, y, groups, group_names):
    """Apply adaptive SMOTE to attack-only data."""
    if not HAS_SMOTE:
        print("  [WARN] imbalanced-learn not installed, skipping SMOTE")
        return X, y, groups

    X_np = X.numpy()
    y_np = y.numpy()

    class_counts = np.bincount(y_np)
    targets = {}
    for cls, cnt in enumerate(class_counts):
        if cnt == 0:
            continue
        if cnt < 50:
            targets[cls] = max(3 * cnt, 200)
        elif cnt < 500:
            targets[cls] = 5000
        else:
            targets[cls] = max(cnt, 10000)

    print(f"  SMOTE targets: {targets}")

    try:
        smote = SMOTE(
            sampling_strategy=targets,
            k_neighbors=min(3, min(class_counts[class_counts > 0]) - 1),
            random_state=42,
        )
        X_res, y_res = smote.fit_resample(X_np, y_np)
    except Exception as e:
        print(f"  [WARN] SMOTE failed: {e}, using original data")
        return X, y, groups

    X_res = torch.tensor(X_res, dtype=torch.float32)
    y_res = torch.tensor(y_res, dtype=torch.long)

    # Rebuild group data
    fg_sizes = [groups[name].shape[1] for name in group_names]
    new_groups = {}
    offset = 0
    for name, sz in zip(group_names, fg_sizes):
        new_groups[name] = X_res[:, offset:offset + sz]
        offset += sz

    added = len(X_res) - len(X)
    print(f"  SMOTE: {len(X)} → {len(X_res)} (+{added} samples)")
    return X_res, y_res, new_groups


def phase2_gradient_align(
    model, train_loader, device, epochs=15, lr=1e-5, lam=0.05,
    checkpoint_path=None, group_names=None,
):
    """Phase 2: Train attention to align with input gradients.

    This makes attention-based explanations more faithful to the
    model's actual decision process, improving XCS (SHAP agreement).
    """
    from src.xai.attention_rollout import AttentionRollout

    print(f"\n  Phase 2: Gradient-aligned attention ({epochs} epochs, λ={lam})")

    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_align = -1.0
    best_state = None
    n_features = None  # will be determined at runtime

    for epoch in range(epochs):
        model.train()
        total_cls_loss = 0
        total_align_loss = 0
        n_batches = 0

        for groups_dict, labels in train_loader:
            groups = {k: v.to(device) for k, v in groups_dict.items()}
            labels = labels.to(device)
            X = torch.cat(list(groups.values()), dim=1)

            if n_features is None:
                n_features = X.shape[1]

            # Forward with explanations
            X_grad = X.clone().requires_grad_(True)
            logits, explanations = model(X_grad, return_explanations=True)

            # Classification loss
            cls_loss = nn.CrossEntropyLoss()(logits, labels)

            # Compute input gradients (feature sensitivity)
            pred_scores = logits.gather(1, labels.unsqueeze(1)).sum()
            grads = torch.autograd.grad(
                pred_scores, X_grad, create_graph=True
            )[0]  # (B, 72)
            grad_imp = grads.abs()
            grad_imp = grad_imp / (grad_imp.sum(dim=1, keepdim=True) + 1e-12)

            # Get attention importance via rollout (handles CLS tokens correctly)
            intra_attn = explanations["raw_intra_attn"]
            inter_attn = explanations["raw_inter_attn"]

            try:
                hi_result = rollout.hierarchical_importance(
                    intra_attn, inter_attn, group_names or sorted(intra_attn.keys())
                )
                attn_imp = hi_result["global_flat"]  # (B, total_features)
                attn_imp = attn_imp / (attn_imp.sum(dim=1, keepdim=True) + 1e-12)

                # Ensure dimensions match
                if attn_imp.shape[1] != n_features:
                    # Truncate or pad to match
                    if attn_imp.shape[1] > n_features:
                        attn_imp = attn_imp[:, :n_features]
                    else:
                        pad = torch.zeros(attn_imp.shape[0], n_features - attn_imp.shape[1], device=device)
                        attn_imp = torch.cat([attn_imp, pad], dim=1)
                    attn_imp = attn_imp / (attn_imp.sum(dim=1, keepdim=True) + 1e-12)

                # Alignment loss: MSE between normalized importances
                align_loss = nn.MSELoss()(attn_imp, grad_imp.detach())
            except Exception as e:
                if n_batches == 0:
                    print(f"    [WARN] Rollout failed: {e}, using cls_loss only")
                align_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + lam * align_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_align_loss += align_loss.item()
            n_batches += 1

        avg_cls = total_cls_loss / max(n_batches, 1)
        avg_align = total_align_loss / max(n_batches, 1)

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: cls={avg_cls:.4f}, "
                  f"align={avg_align:.6f}")

        # Save best by alignment
        if avg_align > best_align or best_state is None:
            best_align = avg_align
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "phase": "phase2_gradient_align",
                }, checkpoint_path)

    print(f"  Phase 2 complete — align_loss: {avg_align:.6f}")
    return model


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, max_epochs, patience, phase_name, checkpoint_path,
    grad_clip=1.0,
):
    """Generic training loop with early stopping."""
    best_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        for groups_dict, labels in train_loader:
            groups = {k: v.to(device) for k, v in groups_dict.items()}
            labels = labels.to(device)

            X = torch.cat(list(groups.values()), dim=1)
            logits = model(X, return_explanations=False)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for groups_dict, labels in val_loader:
                X = torch.cat(list(groups_dict.values()), dim=1).to(device)
                logits = model(X, return_explanations=False)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        if (epoch + 1) % 5 == 0 or val_f1 > best_f1:
            print(f"  Epoch {epoch+1}/{max_epochs}: loss={avg_loss:.4f}, "
                  f"val_f1={val_f1:.4f} {'★' if val_f1 > best_f1 else ''}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "score": val_f1,
                "phase": phase_name,
            }
            torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best state
    if best_state:
        model.load_state_dict(best_state["model_state_dict"])
    print(f"  {phase_name} complete — Best F1: {best_f1:.4f}")
    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Cascade Training for HiT-IDS")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config["project"].get("seed", 42))
    np.random.seed(config["project"].get("seed", 42))

    splits_dir = config["paths"]["splits"]
    checkpoint_dir = config["paths"]["checkpoints"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]

    model_cfg = config["model"]
    train_cfg = config["training"]["phase1"]
    batch_size = train_cfg["batch_size"]

    print("=" * 60)
    print("  HiT-IDS CASCADE TRAINING")
    print("  Stage 1: Binary (BENIGN vs ATTACK)")
    print("  Stage 2: Multi-class (14 attack types)")
    print("=" * 60)

    # ─── Load Data ───────────────────────────────────────────────
    train_data = load_split(splits_dir, "train")
    val_data = load_split(splits_dir, "val")
    test_data = load_split(splits_dir, "test")

    # ════════════════════════════════════════════════════════════
    # STAGE 1: Binary Classification (BENIGN vs ATTACK)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: Binary Classification")
    print("=" * 60)

    # Create binary labels
    y_train_bin = make_binary_labels(train_data["y"])
    y_val_bin = make_binary_labels(val_data["y"])

    train_loader_bin = create_dataloader(
        train_data["X"], y_train_bin, train_data["groups"],
        batch_size=batch_size, weighted=True,
    )
    val_loader_bin = create_dataloader(
        train_data["X"][:len(val_data["y"])],  # placeholder, fix below
        y_val_bin, val_data["groups"],
        batch_size=batch_size, shuffle=False,
    )
    # Fix val loader
    val_loader_bin = create_dataloader(
        val_data["X"], y_val_bin, val_data["groups"],
        batch_size=batch_size, shuffle=False,
    )

    # Binary model (2 classes)
    model_binary = HiTIDS(
        group_sizes=group_sizes,
        group_names=group_names,
        d_emb=model_cfg["d_emb"],
        nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        num_classes=2,
    ).to(device)

    n_params = sum(p.numel() for p in model_binary.parameters())
    print(f"  Binary model: {n_params:,} parameters")

    # Loss + optimizer
    alpha_bin = compute_class_weights(y_train_bin, num_classes=2)
    criterion_bin = FocalLoss(
        alpha=alpha_bin.to(device),
        gamma=config["training"]["focal_loss"]["gamma"],
        label_smoothing=config["training"]["focal_loss"].get("label_smoothing", 0.0),
    )
    optimizer_bin = torch.optim.AdamW(
        model_binary.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler_bin = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_bin, T_0=10, T_mult=2,
    )

    binary_f1 = train_model(
        model_binary, train_loader_bin, val_loader_bin, criterion_bin,
        optimizer_bin, scheduler_bin, device,
        max_epochs=50, patience=10,
        phase_name="BINARY",
        checkpoint_path=os.path.join(checkpoint_dir, "hit_ids_binary.pt"),
    )

    # ════════════════════════════════════════════════════════════
    # STAGE 2: Multi-class (Attack Types Only)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2: Multi-class (14 Attack Types)")
    print("=" * 60)

    # Filter to attack-only samples
    attack_mask_train = train_data["y"] > 0
    attack_mask_val = val_data["y"] > 0

    X_train_attack = train_data["X"][attack_mask_train]
    y_train_attack = train_data["y"][attack_mask_train] - 1  # Shift: class 1→0, etc.
    groups_train_attack = {
        k: v[attack_mask_train] for k, v in train_data["groups"].items()
    }

    X_val_attack = val_data["X"][attack_mask_val]
    y_val_attack = val_data["y"][attack_mask_val] - 1
    groups_val_attack = {
        k: v[attack_mask_val] for k, v in val_data["groups"].items()
    }

    num_attack_classes = int(train_data["y"].max().item())  # 14
    print(f"  Attack train samples: {len(y_train_attack):,}")
    print(f"  Attack val samples: {len(y_val_attack):,}")
    print(f"  Attack classes: {num_attack_classes}")

    train_loader_attack = create_dataloader(
        X_train_attack, y_train_attack, groups_train_attack,
        batch_size=batch_size, weighted=True,
        rare_boost_factor=5.0, rare_threshold=500,
    )
    val_loader_attack = create_dataloader(
        X_val_attack, y_val_attack, groups_val_attack,
        batch_size=batch_size, shuffle=False,
    )

    # Multi-class model (14 classes)
    model_attack = HiTIDS(
        group_sizes=group_sizes,
        group_names=group_names,
        d_emb=model_cfg["d_emb"],
        nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        num_classes=num_attack_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model_attack.parameters())
    print(f"  Attack model: {n_params:,} parameters")

    # Loss + optimizer
    alpha_attack = compute_class_weights(y_train_attack, num_classes=num_attack_classes)
    criterion_attack = FocalLoss(
        alpha=alpha_attack.to(device),
        gamma=config["training"]["focal_loss"]["gamma"],
        label_smoothing=config["training"]["focal_loss"].get("label_smoothing", 0.0),
    )
    optimizer_attack = torch.optim.AdamW(
        model_attack.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler_attack = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_attack, T_0=10, T_mult=2,
    )

    attack_f1 = train_model(
        model_attack, train_loader_attack, val_loader_attack, criterion_attack,
        optimizer_attack, scheduler_attack, device,
        max_epochs=100, patience=15,
        phase_name="ATTACK-MULTI",
        checkpoint_path=os.path.join(checkpoint_dir, "hit_ids_attack_v2.pt"),
    )

    # ════════════════════════════════════════════════════════════
    # STAGE 2.5: Phase 2 — Gradient-aligned Attention
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 2.5: Phase 2 — Gradient-Aligned Attention")
    print("=" * 60)

    phase2_refine_loader = create_dataloader(
        X_train_attack, y_train_attack, groups_train_attack,
        batch_size=batch_size, shuffle=True,
    )
    model_attack = phase2_gradient_align(
        model_attack, phase2_refine_loader, device,
        epochs=15, lr=1e-5, lam=0.05,
        checkpoint_path=os.path.join(checkpoint_dir, "hit_ids_attack_v2.pt"),
        group_names=group_names,
    )

    # ════════════════════════════════════════════════════════════
    # STAGE 3: Cascade Evaluation on Test Set
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  CASCADE EVALUATION (Test Set)")
    print("=" * 60)

    # Reload best models
    ckpt_bin = torch.load(
        os.path.join(checkpoint_dir, "hit_ids_binary.pt"),
        map_location=device, weights_only=False,
    )
    model_binary.load_state_dict(ckpt_bin["model_state_dict"])

    # Use v2 attack model
    v2_path = os.path.join(checkpoint_dir, "hit_ids_attack_v2.pt")
    if os.path.exists(v2_path):
        ckpt_attack = torch.load(v2_path, map_location=device, weights_only=False)
    else:
        ckpt_attack = torch.load(
            os.path.join(checkpoint_dir, "hit_ids_attack.pt"),
            map_location=device, weights_only=False,
        )
    model_attack.load_state_dict(ckpt_attack["model_state_dict"])

    model_binary.eval()
    model_attack.eval()

    # Run cascade on test set
    test_loader = create_dataloader(
        test_data["X"], test_data["y"], test_data["groups"],
        batch_size=batch_size, shuffle=False,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for groups_dict, labels in test_loader:
            X = torch.cat(list(groups_dict.values()), dim=1).to(device)

            # Stage 1: Binary classification
            bin_logits = model_binary(X, return_explanations=False)
            bin_preds = bin_logits.argmax(dim=1)  # 0=BENIGN, 1=ATTACK

            # Stage 2: For predicted attacks, classify attack type
            final_preds = torch.zeros_like(labels)
            attack_mask = bin_preds == 1

            if attack_mask.any():
                X_attacks = X[attack_mask]
                attack_logits = model_attack(X_attacks, return_explanations=False)
                attack_preds = attack_logits.argmax(dim=1).cpu() + 1  # Shift back
                final_preds[attack_mask] = attack_preds

            all_preds.append(final_preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Metrics
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = (all_preds == all_labels).mean()

    print(f"\n  Cascade Results:")
    print(f"    Accuracy:    {accuracy:.4f}")
    print(f"    Macro-F1:    {macro_f1:.4f}")
    print(f"    Weighted-F1: {weighted_f1:.4f}")
    print(f"\n    Binary F1:   {binary_f1:.4f}")
    print(f"    Attack F1:   {attack_f1:.4f}")

    # Save cascade checkpoint
    cascade_ckpt = {
        "binary_model": model_binary.state_dict(),
        "attack_model": model_attack.state_dict(),
        "binary_f1": binary_f1,
        "attack_f1": attack_f1,
        "cascade_macro_f1": macro_f1,
        "cascade_accuracy": accuracy,
    }
    cascade_path = os.path.join(checkpoint_dir, "hit_ids_cascade_v2.pt")
    torch.save(cascade_ckpt, cascade_path)

    print(f"\n  Saved cascade checkpoint: {cascade_path}")

    # Also save as merged checkpoint for XAI compatibility
    merged_path = os.path.join(checkpoint_dir, "hit_ids_attack_merged_v2.pt")
    torch.save({
        "model_state_dict": model_attack.state_dict(),
        "cascade_macro_f1": macro_f1,
    }, merged_path)
    print(f"  Saved merged attack model: {merged_path}")

    # Classification report
    label_names = list(config["dataset"]["label_map"].keys())
    print(f"\n  Per-class Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_names,
        zero_division=0,
    ))

    print("=" * 60)
    print(f"  CASCADE TRAINING COMPLETE!")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Retrain ONLY the attack model with merged Web Attack classes.
Binary model stays the same.

Original: 14 attack classes (index 1-14 in original labels)
Merged:   12 attack classes (Web Attack - BF/SQL/XSS → "Web Attack")

Usage:
    python scripts/retrain_attack_merged.py --config configs/config.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.models.hit_ids import HiTIDS
from src.training.focal_loss import FocalLoss, compute_class_weights


# Original label map (from config):
# 0:BENIGN, 1:Bot, 2:DDoS, 3:DoS GoldenEye, 4:DoS Hulk,
# 5:DoS Slowhttptest, 6:DoS slowloris, 7:FTP-Patator,
# 8:Heartbleed, 9:Infiltration, 10:PortScan, 11:SSH-Patator,
# 12:Web Attack - Brute Force, 13:Web Attack - Sql Injection, 14:Web Attack - XSS

# Merge mapping: 12,13,14 → 12 (all become "Web Attack")
MERGE_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
    12: 12, 13: 12, 14: 12,  # All Web Attacks → 12
}

MERGED_NAMES = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS slowloris", "FTP-Patator",
    "Heartbleed", "Infiltration", "PortScan", "SSH-Patator",
    "Web Attack",  # Merged
]


def merge_labels(y):
    """Apply label merging."""
    return torch.tensor([MERGE_MAP[int(v)] for v in y], dtype=torch.long)


def create_dataloader(X, y, groups, batch_size, shuffle=True, weighted=False):
    dataset = HiTIDSDataset(X=X, y=y, group_data=groups)
    if weighted and shuffle:
        class_counts = torch.bincount(y)
        weights = 1.0 / class_counts.float().clamp(min=1)
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


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, max_epochs, patience, phase_name, ckpt_path):
    best_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        total_loss, n = 0, 0
        for groups_dict, labels in train_loader:
            groups = {k: v.to(device) for k, v in groups_dict.items()}
            labels = labels.to(device)
            X = torch.cat(list(groups.values()), dim=1)
            logits = model(X, return_explanations=False)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for gd, lb in val_loader:
                X = torch.cat(list(gd.values()), dim=1).to(device)
                p = model(X, return_explanations=False).argmax(1).cpu()
                all_preds.append(p)
                all_labels.append(lb)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        if (epoch + 1) % 5 == 0 or val_f1 > best_f1:
            print(f"  E{epoch+1}/{max_epochs}: loss={total_loss/n:.4f}, "
                  f"val_f1={val_f1:.4f} {'★' if val_f1 > best_f1 else ''}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_state = {"model_state_dict": model.state_dict(),
                          "epoch": epoch, "score": val_f1}
            torch.save(best_state, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop @ epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state["model_state_dict"])
    print(f"  {phase_name} — Best F1: {best_f1:.4f}")
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    splits_dir = config["paths"]["splits"]
    ckpt_dir = config["paths"]["checkpoints"]
    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    model_cfg = config["model"]
    batch_size = config["training"]["phase1"]["batch_size"]

    # Load data
    train_data = torch.load(f"{splits_dir}/train.pt", weights_only=False)
    val_data = torch.load(f"{splits_dir}/val.pt", weights_only=False)
    test_data = torch.load(f"{splits_dir}/test.pt", weights_only=False)

    # Merge labels
    y_train_m = merge_labels(train_data["y"])
    y_val_m = merge_labels(val_data["y"])
    y_test_m = merge_labels(test_data["y"])

    print("=" * 60)
    print("  RETRAIN ATTACK MODEL (Merged Web Attacks)")
    print("  13 classes: 0=BENIGN + 12 attack types")
    print("=" * 60)

    # Filter attack-only
    attack_train = y_train_m > 0
    attack_val = y_val_m > 0

    X_train_a = train_data["X"][attack_train]
    y_train_a = y_train_m[attack_train] - 1  # shift to 0-indexed
    g_train_a = {k: v[attack_train] for k, v in train_data["groups"].items()}

    X_val_a = val_data["X"][attack_val]
    y_val_a = y_val_m[attack_val] - 1
    g_val_a = {k: v[attack_val] for k, v in val_data["groups"].items()}

    num_attack_cls = int(y_train_a.max().item()) + 1  # 12
    print(f"  Attack classes: {num_attack_cls}")
    print(f"  Train samples: {len(y_train_a):,}")
    print(f"  Val samples:   {len(y_val_a):,}")

    # Class distribution
    for i in range(num_attack_cls):
        cnt = (y_train_a == i).sum().item()
        print(f"    Class {i} ({MERGED_NAMES[i+1]}): {cnt:,}")

    train_loader = create_dataloader(X_train_a, y_train_a, g_train_a,
                                     batch_size, weighted=True)
    val_loader = create_dataloader(X_val_a, y_val_a, g_val_a,
                                   batch_size, shuffle=False)

    # Build model
    model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        num_classes=num_attack_cls,
    ).to(device)

    alpha = compute_class_weights(y_train_a, num_classes=num_attack_cls)
    criterion = FocalLoss(
        alpha=alpha.to(device),
        gamma=config["training"]["focal_loss"]["gamma"],
        label_smoothing=config["training"]["focal_loss"].get("label_smoothing", 0.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["phase1"]["learning_rate"],
        weight_decay=config["training"]["phase1"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2,
    )

    ckpt_path = os.path.join(ckpt_dir, "hit_ids_attack_merged.pt")
    attack_f1 = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, max_epochs=100, patience=15,
        phase_name="ATTACK-MERGED", ckpt_path=ckpt_path,
    )

    # ── CASCADE EVALUATION ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CASCADE EVALUATION (Merged, Test Set)")
    print("=" * 60)

    # Load binary model
    bin_model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        num_classes=2,
    ).to(device)

    bin_ckpt = torch.load(f"{ckpt_dir}/hit_ids_binary.pt",
                          map_location=device, weights_only=False)
    bin_model.load_state_dict(bin_ckpt["model_state_dict"])
    bin_model.eval()

    # Reload best attack model
    atk_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(atk_ckpt["model_state_dict"])
    model.eval()

    test_loader = create_dataloader(
        test_data["X"], y_test_m, test_data["groups"],
        batch_size, shuffle=False,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for gd, lb in test_loader:
            X = torch.cat(list(gd.values()), dim=1).to(device)
            bin_preds = bin_model(X, return_explanations=False).argmax(1)
            final = torch.zeros_like(lb)
            atk_mask = bin_preds == 1
            if atk_mask.any():
                atk_logits = model(X[atk_mask], return_explanations=False)
                final[atk_mask] = atk_logits.argmax(1).cpu() + 1
            all_preds.append(final)
            all_labels.append(lb)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = (all_preds == all_labels).mean()

    print(f"\n  Cascade Results (Merged):")
    print(f"    Accuracy:    {accuracy:.4f}")
    print(f"    Macro-F1:    {macro_f1:.4f}")
    print(f"    Weighted-F1: {weighted_f1:.4f}")
    print(f"    Attack F1:   {attack_f1:.4f}")

    print(f"\n  Per-class Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=MERGED_NAMES,
        zero_division=0,
    ))

    # Save
    torch.save({
        "binary_model": bin_model.state_dict(),
        "attack_model": model.state_dict(),
        "cascade_macro_f1": macro_f1,
        "cascade_accuracy": accuracy,
        "attack_f1": attack_f1,
        "num_classes_merged": 13,
    }, os.path.join(ckpt_dir, "hit_ids_cascade_merged.pt"))

    print("=" * 60)
    print(f"  DONE! Macro-F1: {macro_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
HiT-IDS Evaluation Entry Point.

Evaluates HiT-IDS and all baseline models on the test set.
Produces comparison tables and per-class metrics.

Usage:
    python scripts/evaluate.py --config configs/config.yaml
    python scripts/evaluate.py --config configs/config.yaml --model-only
    python scripts/evaluate.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.evaluation.metrics import MetricsCalculator, create_comparison_table
from src.models.baselines import (
    SklearnBaseline,
    create_baseline,
    train_pytorch_baseline,
)
from src.models.hit_ids import HiTIDS

# Merge mapping: 15 classes → 13 classes
# 12:Web Attack - Brute Force, 13:Sql Injection, 14:XSS → all become 12:Web Attack
MERGE_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
    12: 12, 13: 12, 14: 12,
}
MERGED_NAMES = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS slowloris", "FTP-Patator",
    "Heartbleed", "Infiltration", "PortScan", "SSH-Patator",
    "Web Attack",
]
NUM_MERGED = 13


def merge_labels(y):
    """Merge Web Attack sub-classes into single class."""
    if isinstance(y, np.ndarray):
        return np.array([MERGE_MAP[int(v)] for v in y])
    return np.array([MERGE_MAP[int(v)] for v in y.numpy()])


def evaluate_hit_ids(
    model: HiTIDS,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> tuple:
    """Evaluate HiT-IDS on test set, return (y_true, y_pred)."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_groups, batch_labels in test_loader:
            batch_groups = {k: v.to(device) for k, v in batch_groups.items()}
            logits = model(batch_groups, return_explanations=False)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_labels.tolist())

    return np.array(all_targets), np.array(all_preds)


def evaluate_sklearn_baseline(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple:
    """Train and evaluate a sklearn baseline."""
    baseline = SklearnBaseline(model_type=model_type)
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    return y_test, y_pred


def evaluate_pytorch_baseline(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_features: int,
    num_classes: int,
    device: str = "cuda",
) -> tuple:
    """Train and evaluate a PyTorch baseline."""
    model = create_baseline(model_type, num_features, num_classes)

    # Create DataLoaders (flat tensors, not grouped)
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # Train
    print(f"\n  Training {model_type}...")
    train_pytorch_baseline(
        model, train_loader, test_loader,
        num_epochs=30, lr=1e-3, device=device,
    )

    # Evaluate
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())

    return y_test, np.array(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="HiT-IDS Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--model-only", action="store_true",
                        help="Evaluate only HiT-IDS (skip baselines)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test with dummy data")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = args.device or config["project"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Apply merged labels
    num_classes = NUM_MERGED
    label_names = MERGED_NAMES

    print("=" * 60)
    print("  HiT-IDS Evaluation (13-class merged)")
    print("=" * 60)

    # Load data
    if args.dry_run:
        print("\n[DRY RUN] Creating dummy data...")
        fg_cfg = config["feature_groups"]
        group_names = list(fg_cfg.keys())
        group_sizes = [len(fg_cfg[name]) for name in group_names]
        total_features = sum(group_sizes)

        N_train, N_test = 300, 100
        X_train = np.random.randn(N_train, total_features).astype(np.float32)
        y_train = np.random.randint(0, num_classes, N_train)
        X_test = np.random.randn(N_test, total_features).astype(np.float32)
        y_test = np.random.randint(0, num_classes, N_test)

        group_data = {}
        offset = 0
        for name, size in zip(group_names, group_sizes):
            group_data[name] = torch.tensor(X_test[:, offset:offset+size])
            offset += size

        test_dataset = HiTIDSDataset(
            X=torch.tensor(X_test, dtype=torch.float32),
            y=torch.tensor(y_test, dtype=torch.long),
            group_data=group_data,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, collate_fn=collate_fn
        )
    else:
        splits_dir = config["paths"]["splits"]
        test_dataset = HiTIDSDataset.from_split(splits_dir, "test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, collate_fn=collate_fn
        )
        train_dataset = HiTIDSDataset.from_split(splits_dir, "train")

        X_train = train_dataset.X.numpy()
        y_train = merge_labels(train_dataset.y)
        X_test = test_dataset.X.numpy()
        y_test = merge_labels(test_dataset.y)
        total_features = X_test.shape[1]

    # Metrics calculator
    calc = MetricsCalculator(num_classes=num_classes, label_names=label_names)
    all_results: dict = {}

    # ── Evaluate HiT-IDS (CASCADE) ───────────────────────────────────────
    print("\n" + "─" * 40)
    print("Evaluating HiT-IDS (Cascade)...")

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    model_cfg = config["model"]

    # Load binary model
    bin_ckpt_path = os.path.join(config["paths"]["checkpoints"], "hit_ids_binary.pt")
    atk_ckpt_path = os.path.join(config["paths"]["checkpoints"], "hit_ids_attack_merged.pt")

    if os.path.exists(bin_ckpt_path) and os.path.exists(atk_ckpt_path):
        bin_model = HiTIDS(
            group_sizes=group_sizes, group_names=group_names,
            d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
            intra_layers=model_cfg["intra_group_layers"],
            inter_layers=model_cfg["inter_group_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"], num_classes=2,
        ).to(device)
        bin_ckpt = torch.load(bin_ckpt_path, map_location=device, weights_only=False)
        bin_model.load_state_dict(bin_ckpt["model_state_dict"])

        atk_model = HiTIDS(
            group_sizes=group_sizes, group_names=group_names,
            d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
            intra_layers=model_cfg["intra_group_layers"],
            inter_layers=model_cfg["inter_group_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"], num_classes=NUM_MERGED - 1,  # 12 attack classes
        ).to(device)
        atk_ckpt = torch.load(atk_ckpt_path, map_location=device, weights_only=False)
        atk_model.load_state_dict(atk_ckpt["model_state_dict"])

        bin_model.eval()
        atk_model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_groups, batch_labels in test_loader:
                batch_groups_dev = {k: v.to(device) for k, v in batch_groups.items()}
                X = torch.cat(list(batch_groups_dev.values()), dim=1)
                bin_preds = bin_model(X, return_explanations=False).argmax(1)
                final = torch.zeros(len(batch_labels), dtype=torch.long)
                atk_mask = bin_preds == 1
                if atk_mask.any():
                    atk_logits = atk_model(X[atk_mask], return_explanations=False)
                    final[atk_mask] = atk_logits.argmax(1).cpu() + 1
                all_preds.extend(final.tolist())
                all_targets.extend(batch_labels.tolist())

        y_true = merge_labels(np.array(all_targets))
        y_pred = np.array(all_preds)
        print(f"  Loaded cascade models: {bin_ckpt_path}, {atk_ckpt_path}")
    else:
        print(f"  [WARNING] Cascade checkpoints not found. Using single model.")
        model = HiTIDS.from_dict(config)
        checkpoint_path = args.checkpoint or os.path.join(
            config["paths"]["checkpoints"], "hit_ids_phase2_best.pt"
        )
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        y_true, y_pred = evaluate_hit_ids(model, test_loader, device)
        y_true = merge_labels(y_true)
        y_pred = merge_labels(y_pred)

    hit_ids_metrics = calc.compute(y_true, y_pred)
    calc.print_report(hit_ids_metrics)
    all_results["HiT-IDS (Cascade)"] = hit_ids_metrics

    # ── Evaluate Baselines ───────────────────────────────────────────────
    if not args.model_only:
        baselines = config.get("evaluation", {}).get("baselines", [])

        for baseline_type in baselines:
            print(f"\n{'─' * 40}")
            print(f"Evaluating {baseline_type}...")
            start = time.time()

            try:
                if baseline_type in ("random_forest", "xgboost"):
                    yt, yp = evaluate_sklearn_baseline(
                        baseline_type, X_train, y_train, X_test, y_test
                    )
                else:
                    yt, yp = evaluate_pytorch_baseline(
                        baseline_type, X_train, y_train, X_test, y_test,
                        total_features, num_classes, device
                    )

                metrics = calc.compute(yt, yp)
                all_results[baseline_type] = metrics
                print(f"  F1 (macro): {metrics['f1_macro']:.4f} "
                      f"({time.time()-start:.1f}s)")
            except Exception as e:
                print(f"  [ERROR] {baseline_type} failed: {e}")

    # ── Comparison Table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    table = create_comparison_table(all_results)

    # Save results
    tables_dir = config["paths"]["tables"]
    os.makedirs(tables_dir, exist_ok=True)

    results_path = os.path.join(tables_dir, "comparison_results.json")
    serializable = {}
    for model_name, metrics in all_results.items():
        serializable[model_name] = {
            k: v for k, v in metrics.items()
            if k not in ("classification_report", "confusion_matrix")
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    with open(os.path.join(tables_dir, "comparison_table.txt"), "w") as f:
        f.write(table)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

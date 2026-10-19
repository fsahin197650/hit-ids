#!/usr/bin/env python3
"""
Robust Multi-Seed Evaluation with Statistical Significance Tests.

Trains ALL models (HiT-IDS Cascade + 5 baselines) across multiple random seeds,
then performs pairwise statistical tests (Wilcoxon, Cohen's d) comparing HiT-IDS
against each baseline.

Designed for Q1/Q2 journal-grade reproducibility.

Usage:
    python scripts/evaluate_robust.py --config configs/config.yaml --seeds 3
    python scripts/evaluate_robust.py --config configs/config.yaml --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from scipy import stats
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.models.hit_ids import HiTIDS
from src.models.baselines import SklearnBaseline, create_baseline, train_pytorch_baseline
from src.training.focal_loss import FocalLoss, compute_class_weights

# ── Label Merging ────────────────────────────────────────────────────────
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

SEED_LIST = [42, 123, 456, 789, 2024]


def merge_labels(y):
    if isinstance(y, np.ndarray):
        return np.array([MERGE_MAP[int(v)] for v in y])
    return np.array([MERGE_MAP[int(v)] for v in y.numpy()])


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 +
                          (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


# ── Model Training Functions ────────────────────────────────────────────

def create_dataloader(X, y, groups, batch_size, shuffle=True, weighted=False):
    dataset = HiTIDSDataset(X=X, y=y, group_data=groups)
    if weighted and shuffle:
        class_counts = torch.bincount(y).float().clamp(min=1)
        sample_weights = 1.0 / class_counts[y]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=4, pin_memory=True)


def train_hit_ids_model(model, train_loader, val_loader, criterion,
                        lr, wd, device, max_epochs, patience):
    """Train a HiT-IDS model and return best val F1."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    best_f1, patience_ctr, best_state = 0.0, 0, None

    for epoch in range(max_epochs):
        model.train()
        for gd, lb in train_loader:
            groups = {k: v.to(device) for k, v in gd.items()}
            lb = lb.to(device)
            X = torch.cat(list(groups.values()), dim=1)
            loss = criterion(model(X, return_explanations=False), lb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for gd, lb in val_loader:
                X = torch.cat(list(gd.values()), dim=1).to(device)
                p = model(X, return_explanations=False).argmax(1).cpu()
                preds.append(p)
                labels.append(lb)
        val_f1 = f1_score(torch.cat(labels).numpy(), torch.cat(preds).numpy(),
                          average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_f1


def train_and_eval_cascade(config, train_data, val_data, test_data, device, seed):
    """Train cascade HiT-IDS and evaluate on test set."""
    set_seed(seed)

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    model_cfg = config["model"]
    batch_size = config["training"]["phase1"]["batch_size"]
    lr = config["training"]["phase1"]["learning_rate"]
    wd = config["training"]["phase1"]["weight_decay"]
    gamma = config["training"]["focal_loss"]["gamma"]
    ls = config["training"]["focal_loss"].get("label_smoothing", 0.0)

    y_train_m = merge_labels(train_data["y"])
    y_val_m = merge_labels(val_data["y"])
    y_test_m = merge_labels(test_data["y"])

    # ── BINARY MODEL ──
    y_train_bin = (torch.tensor(y_train_m) > 0).long()
    y_val_bin = (torch.tensor(y_val_m) > 0).long()

    bin_train_loader = create_dataloader(
        train_data["X"], y_train_bin, train_data["groups"],
        batch_size, weighted=True)
    bin_val_loader = create_dataloader(
        val_data["X"], y_val_bin, val_data["groups"],
        batch_size, shuffle=False)

    bin_model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"], num_classes=2,
    ).to(device)

    alpha_bin = compute_class_weights(y_train_bin, num_classes=2)
    criterion_bin = FocalLoss(alpha=alpha_bin.to(device), gamma=gamma, label_smoothing=ls)

    print(f"    Binary model training...")
    bin_f1 = train_hit_ids_model(
        bin_model, bin_train_loader, bin_val_loader, criterion_bin,
        lr, wd, device, max_epochs=50, patience=10)
    print(f"    Binary val F1: {bin_f1:.4f}")

    # ── ATTACK MODEL ──
    y_train_t = torch.tensor(y_train_m)
    y_val_t = torch.tensor(y_val_m)

    atk_mask_train = y_train_t > 0
    atk_mask_val = y_val_t > 0

    y_train_atk = y_train_t[atk_mask_train] - 1
    y_val_atk = y_val_t[atk_mask_val] - 1
    num_atk = NUM_MERGED - 1

    atk_train_loader = create_dataloader(
        train_data["X"][atk_mask_train], y_train_atk,
        {k: v[atk_mask_train] for k, v in train_data["groups"].items()},
        batch_size, weighted=True)
    atk_val_loader = create_dataloader(
        val_data["X"][atk_mask_val], y_val_atk,
        {k: v[atk_mask_val] for k, v in val_data["groups"].items()},
        batch_size, shuffle=False)

    atk_model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"], num_classes=num_atk,
    ).to(device)

    alpha_atk = compute_class_weights(y_train_atk, num_classes=num_atk)
    criterion_atk = FocalLoss(alpha=alpha_atk.to(device), gamma=gamma, label_smoothing=ls)

    print(f"    Attack model training...")
    atk_f1 = train_hit_ids_model(
        atk_model, atk_train_loader, atk_val_loader, criterion_atk,
        lr, wd, device, max_epochs=100, patience=15)
    print(f"    Attack val F1: {atk_f1:.4f}")

    # ── CASCADE EVAL ──
    bin_model.eval()
    atk_model.eval()

    test_loader = create_dataloader(
        test_data["X"], torch.tensor(y_test_m), test_data["groups"],
        batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for gd, lb in test_loader:
            X = torch.cat(list(gd.values()), dim=1).to(device)
            bp = bin_model(X, return_explanations=False).argmax(1)
            final = torch.zeros(len(lb), dtype=torch.long)
            mask = bp == 1
            if mask.any():
                ap = atk_model(X[mask], return_explanations=False).argmax(1).cpu() + 1
                final[mask] = ap
            all_preds.append(final)
            all_labels.append(lb)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = (y_true == y_pred).mean()

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "binary_f1": bin_f1,
        "attack_f1": atk_f1,
    }


def train_and_eval_baseline(model_type, X_train, y_train, X_test, y_test,
                            num_features, num_classes, device, seed):
    """Train and evaluate a single baseline model."""
    set_seed(seed)

    if model_type in ("random_forest", "xgboost"):
        baseline = SklearnBaseline(model_type=model_type)
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
    else:
        model = create_baseline(model_type, num_features, num_classes)
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long))
        test_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        train_pytorch_baseline(model, train_loader, test_loader,
                               num_epochs=30, lr=1e-3, device=device)
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
        y_pred = np.array(preds)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = (y_test == y_pred).mean()

    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "accuracy": accuracy}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEED_LIST[:args.seeds]
    splits_dir = config["paths"]["splits"]
    tables_dir = config["paths"]["tables"]
    os.makedirs(tables_dir, exist_ok=True)

    baselines = config.get("evaluation", {}).get("baselines", [])
    all_models = ["HiT-IDS (Cascade)"] + baselines

    print("=" * 70)
    print("  ROBUST MULTI-SEED EVALUATION")
    print(f"  Seeds: {seeds}")
    print(f"  Models: {all_models}")
    print(f"  Classes: {NUM_MERGED} (merged)")
    print("=" * 70)

    # Load data
    train_data = torch.load(f"{splits_dir}/train.pt", weights_only=False)
    val_data = torch.load(f"{splits_dir}/val.pt", weights_only=False)
    test_data = torch.load(f"{splits_dir}/test.pt", weights_only=False)

    X_train = train_data["X"].numpy()
    y_train = merge_labels(train_data["y"])
    X_test = test_data["X"].numpy()
    y_test = merge_labels(test_data["y"])
    total_features = X_train.shape[1]

    # ── Results storage ──
    results = {model: [] for model in all_models}

    for si, seed in enumerate(seeds):
        print(f"\n{'═' * 70}")
        print(f"  SEED {si+1}/{len(seeds)}: {seed}")
        print(f"{'═' * 70}")

        # ── HiT-IDS Cascade ──
        print(f"\n  [HiT-IDS (Cascade)] seed={seed}")
        t0 = time.time()
        hit_result = train_and_eval_cascade(
            config, train_data, val_data, test_data, device, seed)
        elapsed = time.time() - t0
        results["HiT-IDS (Cascade)"].append(hit_result)
        print(f"  → Macro-F1: {hit_result['macro_f1']:.4f} "
              f"(bin={hit_result['binary_f1']:.4f}, "
              f"atk={hit_result['attack_f1']:.4f}) [{elapsed:.0f}s]")

        # ── Baselines ──
        for bl in baselines:
            print(f"\n  [{bl}] seed={seed}")
            t0 = time.time()
            try:
                bl_result = train_and_eval_baseline(
                    bl, X_train, y_train, X_test, y_test,
                    total_features, NUM_MERGED, device, seed)
                results[bl].append(bl_result)
                elapsed = time.time() - t0
                print(f"  → Macro-F1: {bl_result['macro_f1']:.4f} [{elapsed:.0f}s]")
            except Exception as e:
                print(f"  → FAILED: {e}")
                results[bl].append({"macro_f1": 0.0, "weighted_f1": 0.0, "accuracy": 0.0})

    # ════════════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  STATISTICAL ANALYSIS")
    print("=" * 70)

    hit_f1s = np.array([r["macro_f1"] for r in results["HiT-IDS (Cascade)"]])

    # Summary Table
    print(f"\n{'Model':<30s} {'Mean F1':>8s} {'Std':>6s} {'Min':>6s} {'Max':>6s}")
    print("-" * 56)
    for model in all_models:
        f1s = np.array([r["macro_f1"] for r in results[model]])
        if len(f1s) > 0 and f1s.max() > 0:
            print(f"{model:<30s} {f1s.mean():>8.4f} {f1s.std():>6.4f} "
                  f"{f1s.min():>6.4f} {f1s.max():>6.4f}")

    # Pairwise Tests: HiT-IDS vs each baseline
    print(f"\n{'─' * 70}")
    print(f"  Pairwise Tests: HiT-IDS (Cascade) vs Baselines")
    print(f"{'─' * 70}")

    stat_results = {}
    for bl in baselines:
        bl_f1s = np.array([r["macro_f1"] for r in results[bl]])
        if len(bl_f1s) == 0 or bl_f1s.max() == 0:
            continue

        delta = hit_f1s - bl_f1s
        mean_delta = delta.mean()

        # Cohen's d
        d = cohens_d(hit_f1s, bl_f1s)

        # Wilcoxon signed-rank test (need at least 6 samples for reliability)
        if len(seeds) >= 5:
            try:
                w_stat, w_p = stats.wilcoxon(hit_f1s, bl_f1s, alternative="greater")
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
        else:
            # With <5 paired samples, use paired t-test instead
            t_stat, t_p = stats.ttest_rel(hit_f1s, bl_f1s)
            w_stat = t_stat
            w_p = t_p / 2  # one-sided

        # 95% CI on delta
        if len(seeds) > 1:
            ci_low = mean_delta - 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
            ci_high = mean_delta + 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
        else:
            ci_low = ci_high = mean_delta

        sig = "✓" if w_p < 0.05 else "✗"
        effect = ("large" if abs(d) >= 0.8 else
                  "medium" if abs(d) >= 0.5 else "small")

        stat_results[bl] = {
            "delta_mean": float(mean_delta),
            "delta_ci_low": float(ci_low),
            "delta_ci_high": float(ci_high),
            "cohens_d": float(d),
            "effect_size": effect,
            "p_value": float(w_p),
            "significant": w_p < 0.05,
        }

        direction = "better" if mean_delta > 0 else "worse"
        print(f"\n  vs {bl}:")
        print(f"    ΔF1 = {mean_delta:+.4f} (HiT-IDS {direction})")
        print(f"    95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"    Cohen's d = {d:.3f} ({effect})")
        print(f"    p-value = {w_p:.4f} {sig}")

    # ── Save Results ──
    output = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "num_classes": NUM_MERGED,
        "class_names": MERGED_NAMES,
        "per_seed_results": {
            model: results[model] for model in all_models
        },
        "summary": {
            model: {
                "mean_macro_f1": float(np.mean([r["macro_f1"] for r in results[model]])),
                "std_macro_f1": float(np.std([r["macro_f1"] for r in results[model]])),
                "mean_accuracy": float(np.mean([r["accuracy"] for r in results[model]])),
                "mean_weighted_f1": float(np.mean([r["weighted_f1"] for r in results[model]])),
            }
            for model in all_models if results[model]
        },
        "statistical_tests": stat_results,
    }

    outpath = os.path.join(tables_dir, "robust_evaluation.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {outpath}")

    # ── Final Summary Table ──
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Model':<30s} {'Macro-F1':>10s} {'Accuracy':>10s} {'W-F1':>10s}")
    print("-" * 62)
    for model in all_models:
        s = output["summary"].get(model, {})
        if s:
            f1_str = f"{s['mean_macro_f1']:.4f}±{s['std_macro_f1']:.4f}"
            print(f"{model:<30s} {f1_str:>10s} "
                  f"{s['mean_accuracy']:>10.4f} {s['mean_weighted_f1']:>10.4f}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

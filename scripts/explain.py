#!/usr/bin/env python3
"""
HiT-IDS XAI Analysis Entry Point.

Runs the full explainability analysis:
    1. Attention rollout → per-class group importance heatmap
    2. SHAP analysis → feature importance
    3. XCS computation → consistency histogram
    4. Faithfulness test → masking curves
    5. All visualizations → results/figures/

Usage:
    python scripts/explain.py --config configs/config.yaml --checkpoint results/checkpoints/hit_ids_phase2_best.pt
    python scripts/explain.py --dry-run
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
from src.data.feature_groups import FeatureGroups
from src.models.hit_ids import HiTIDS
from src.xai.attention_rollout import AttentionRollout
from src.xai.faithfulness import FaithfulnessTest
from src.xai.visualization import XAIVisualizer
from src.xai.xcs_metric import (
    compute_xcs,
    compute_xcs_batch,
    compute_top_k_overlap_batch,
)


def get_attention_importance(
    model: HiTIDS,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    max_samples: int = 500,
    rollout_cfg: dict = None,
) -> dict:
    """
    Compute attention-based importance for test samples.

    Returns:
        Dict with:
            - 'global_flat': (N, total_features) global importance
            - 'group': (N, 4) group importance
            - 'intra': dict of group → (N, features_g)
            - 'y_true': (N,) true labels
    """
    model.eval()
    cfg = rollout_cfg or {}
    rollout = AttentionRollout(
        residual_weight=cfg.get("residual_weight", 0.5),
        head_fusion=cfg.get("head_fusion", "mean"),
    )

    all_global: list = []
    all_group: list = []
    all_intra: dict = {name: [] for name in model.group_names}
    all_labels: list = []
    total = 0

    with torch.no_grad():
        for batch_groups, batch_labels in data_loader:
            if total >= max_samples:
                break

            batch_groups = {k: v.to(device) for k, v in batch_groups.items()}
            logits, explanations = model(batch_groups, return_explanations=True)

            # Compute hierarchical importance
            result = rollout.hierarchical_importance(
                explanations["raw_intra_attn"],
                explanations["raw_inter_attn"],
                group_names=model.group_names,
            )

            all_global.append(result["global_flat"].cpu())
            all_group.append(result["group"].cpu())
            for name in model.group_names:
                all_intra[name].append(result["intra"][name].cpu())
            all_labels.append(batch_labels)

            total += batch_labels.shape[0]

    return {
        "global_flat": torch.cat(all_global)[:max_samples].numpy(),
        "group": torch.cat(all_group)[:max_samples].numpy(),
        "intra": {
            name: torch.cat(tensors)[:max_samples].numpy()
            for name, tensors in all_intra.items()
        },
        "y_true": torch.cat(all_labels)[:max_samples].numpy(),
    }


def compute_per_class_group_importance(
    group_importance: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Average group importance per class.

    Returns:
        (num_classes, num_groups) array.
    """
    num_groups = group_importance.shape[1]
    result = np.zeros((num_classes, num_groups))
    counts = np.zeros(num_classes)

    for i in range(len(y_true)):
        cls = int(y_true[i])
        if cls < num_classes:
            result[cls] += group_importance[i]
            counts[cls] += 1

    for cls in range(num_classes):
        if counts[cls] > 0:
            result[cls] /= counts[cls]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="HiT-IDS XAI Analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP (slow)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = args.device or config["project"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    num_classes = config["dataset"]["num_classes"]
    label_map = config["dataset"].get("label_map", {})
    label_names = list(label_map.keys()) if label_map else None
    fg = FeatureGroups.from_dict(config)

    print("=" * 60)
    print("  HiT-IDS XAI Analysis")
    print("=" * 60)

    # Load model
    model = HiTIDS.from_dict(config)
    checkpoint_path = args.checkpoint or os.path.join(
        config["paths"]["checkpoints"], "hit_ids_phase2_best.pt"
    )
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  [WARNING] No checkpoint found. Using random weights.")

    model = model.to(device)

    # Load data
    if args.dry_run:
        print("\n[DRY RUN] Creating dummy data...")
        N = 200
        group_data = {
            name: torch.randn(N, size)
            for name, size in zip(fg.group_names, fg.group_sizes)
        }
        X = torch.cat(list(group_data.values()), dim=1)
        y = torch.randint(0, num_classes, (N,))
        dataset = HiTIDSDataset(X=X, y=y, group_data=group_data)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, collate_fn=collate_fn
        )
    else:
        splits_dir = config["paths"]["splits"]
        dataset = HiTIDSDataset.from_split(splits_dir, "test")
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, collate_fn=collate_fn
        )

    # Initialize visualizer
    figures_dir = config["paths"]["figures"]
    viz = XAIVisualizer(
        save_dir=figures_dir,
        label_names=label_names,
        group_names=[n.capitalize() for n in fg.group_names],
    )

    xai_results: dict = {}

    # ── 1. Attention Rollout ─────────────────────────────────────────────
    print("\n[1/4] Computing Attention Rollout...")
    rollout_cfg = config.get("xai", {}).get("rollout", {})
    attn_data = get_attention_importance(
        model, test_loader, device,
        max_samples=args.max_samples,
        rollout_cfg=rollout_cfg,
    )

    # Per-class group importance
    per_class_gi = compute_per_class_group_importance(
        attn_data["group"], attn_data["y_true"], num_classes
    )
    viz.plot_group_importance_heatmap(per_class_gi)
    viz.plot_attack_signature_radar(per_class_gi)

    mean_group_importance = attn_data["group"].mean(axis=0)
    print(f"  Mean group importance: {mean_group_importance}")
    xai_results["group_importance"] = mean_group_importance.tolist()

    # ── 2. SHAP Analysis (optional) ──────────────────────────────────────
    shap_importance = None
    if not args.skip_shap:
        print("\n[2/4] Computing SHAP values...")
        try:
            from src.xai.shap_analysis import SHAPAnalyzer

            # Background data (subset of test data)
            bg_size = min(config.get("xai", {}).get("shap", {}).get("background_samples", 100),
                          len(dataset))
            bg_indices = np.random.choice(len(dataset), bg_size, replace=False)
            bg_data = dataset.X[bg_indices].to(device)

            analyzer = SHAPAnalyzer(model, bg_data, device=device)
            test_X = dataset.X[:args.max_samples]
            shap_importance = analyzer.feature_importance(test_X, max_samples=args.max_samples)

            print(f"  SHAP importance shape: {shap_importance.shape}")
            xai_results["shap_importance"] = shap_importance.tolist()

            # Top features comparison
            viz.plot_top_features_comparison(
                attn_data["global_flat"].mean(axis=0),
                shap_importance,
                fg.all_features,
                top_k=10,
            )
        except Exception as e:
            print(f"  [WARNING] SHAP analysis failed: {e}")
            print("  Continuing without SHAP...")
            args.skip_shap = True
    else:
        print("\n[2/4] SHAP skipped (--skip-shap)")

    # ── 3. XCS Computation ───────────────────────────────────────────────
    if not args.skip_shap and shap_importance is not None:
        print("\n[3/4] Computing XCS...")

        # Per-sample XCS
        attn_flat = attn_data["global_flat"]
        N = min(len(attn_flat), args.max_samples)

        # Expand SHAP to match per-sample
        # (SHAP is aggregated, so we use the same for all vs per-sample attention)
        shap_expanded = np.tile(shap_importance, (N, 1))

        xcs_batch = compute_xcs_batch(attn_flat[:N], shap_expanded[:N])
        print(f"  XCS stats: {xcs_batch}")
        xai_results["xcs"] = xcs_batch

        # Top-K overlap
        topk_overlap = compute_top_k_overlap_batch(
            attn_flat[:N], shap_expanded[:N],
            k_values=config.get("xai", {}).get("faithfulness", {}).get("top_k_values", [5, 10, 15, 20])
        )
        print(f"  Top-K overlap: {topk_overlap}")
        xai_results["top_k_overlap"] = topk_overlap

        # XCS distribution plot
        xcs_scores = np.array([
            compute_xcs(attn_flat[i], shap_importance)[0]
            for i in range(N)
        ])
        xcs_scores = xcs_scores[~np.isnan(xcs_scores)]
        viz.plot_xcs_distribution(xcs_scores)
    else:
        print("\n[3/4] XCS skipped (no SHAP data)")

    # ── 4. Faithfulness Test ─────────────────────────────────────────────
    print("\n[4/4] Running Faithfulness Test...")
    faith_cfg = config.get("xai", {}).get("faithfulness", {})
    tester = FaithfulnessTest(
        model, device=device,
        masking_strategy=faith_cfg.get("masking_strategy", "zero"),
    )

    test_X = dataset.X[:args.max_samples]
    test_y = dataset.y[:args.max_samples]
    mean_importance = attn_data["global_flat"].mean(axis=0)

    faith_results = tester.run_faithfulness_test(
        test_X, test_y, mean_importance,
        top_k_values=faith_cfg.get("top_k_values", [5, 10, 15, 20, 25]),
    )
    xai_results["faithfulness"] = {
        "score": faith_results["faithfulness_score"],
        "baseline_accuracy": faith_results["baseline_accuracy"],
        "baseline_f1": faith_results["baseline_f1"],
    }

    viz.plot_faithfulness_curves(faith_results)

    # ── Save results ─────────────────────────────────────────────────────
    tables_dir = config["paths"]["tables"]
    os.makedirs(tables_dir, exist_ok=True)
    results_path = os.path.join(tables_dir, "xai_results.json")
    with open(results_path, "w") as f:
        json.dump(xai_results, f, indent=2, default=str)
    print(f"\n  XAI results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("  XAI Analysis Complete!")
    print(f"  Figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

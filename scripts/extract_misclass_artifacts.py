#!/usr/bin/env python3
"""
Misclassification Artifact Extraction for Section 5.9 (Pattern-Aware Error Analysis).

Runs the HiT-IDS cascade on the test set, collects:
  - y_true, y_pred (per-sample)
  - attention rollout (feature-level & group-level importance per sample)
  - confidence scores
  - confusion matrix
  
Saves everything to misclass_artifacts.npz.

Usage:
    conda activate hit_ids
    python scripts/extract_misclass_artifacts.py
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.models.hit_ids import HiTIDS
from src.xai.attention_rollout import AttentionRollout
from sklearn.metrics import confusion_matrix, classification_report

# ── Constants ────────────────────────────────────────────────────────────────
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
    if isinstance(y, np.ndarray):
        return np.array([MERGE_MAP[int(v)] for v in y])
    return np.array([MERGE_MAP[int(v)] for v in y.numpy()])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", default="auto",
                    help="Stage-1 threshold on p(attack); 'auto' = results/tables/deployment_eval.json")
    ap.add_argument("--out", default="results/tables/misclass_artifacts_clean.npz")
    args = ap.parse_args()
    from src.evaluation.deployment import read_threshold
    THR = read_threshold() if args.threshold == "auto" else float(args.threshold)
    print(f"Stage-1 threshold: {THR:.3f}")
    # ── Load config ──────────────────────────────────────────────────────
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    model_cfg = config["model"]

    # ── Load test data ───────────────────────────────────────────────────
    test_dataset = HiTIDSDataset.from_split(config["paths"]["splits"], "test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    print(f"Test samples: {len(test_dataset)}")

    # ── Load cascade models ──────────────────────────────────────────────
    bin_ckpt_path = os.path.join(config["paths"]["checkpoints"], "hit_ids_binary.pt")
    atk_ckpt_path = os.path.join(config["paths"]["checkpoints"], "hit_ids_attack_merged.pt")

    print(f"Loading binary model: {bin_ckpt_path}")
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
    bin_model.eval()

    print(f"Loading attack model: {atk_ckpt_path}")
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
    atk_model.eval()

    # ── Attention rollout tool ───────────────────────────────────────────
    rollout_tool = AttentionRollout(residual_weight=0.5, head_fusion="mean")

    # ── Run inference with attention extraction ──────────────────────────
    all_y_true = []
    all_y_pred = []
    all_stage1_pred = []
    all_confidence = []
    all_attn_group = []   # (N, 4) group-level importance
    all_attn_feat = []    # (N, 72) feature-level importance (global flat)

    total_features = sum(group_sizes)
    print(f"\nRunning cascade inference with attention extraction...")
    print(f"Feature groups: {dict(zip(group_names, group_sizes))}")
    print(f"Total features: {total_features}")

    batch_count = 0
    with torch.no_grad():
        for batch_groups, batch_labels in test_loader:
            batch_groups_dev = {k: v.to(device) for k, v in batch_groups.items()}
            batch_size = batch_labels.shape[0]

            # Concatenate for flat input
            X_flat = torch.cat([batch_groups_dev[name] for name in group_names], dim=1)

            # ── Stage 1: Binary classification ───────────────────────────
            bin_logits, bin_explanations = bin_model(X_flat, return_explanations=True)
            bin_preds = (torch.softmax(bin_logits, dim=1)[:, 1] >= THR).long()  # 0=BENIGN, 1=ATTACK
            bin_conf = torch.softmax(bin_logits, dim=1).max(dim=1).values

            # Compute attention rollout for binary model
            bin_hier = rollout_tool.hierarchical_importance(
                bin_explanations["raw_intra_attn"],
                bin_explanations["raw_inter_attn"],
                group_names,
            )

            # Initialize per-sample outputs
            final_preds = torch.zeros(batch_size, dtype=torch.long)
            confidences = bin_conf.cpu().numpy()
            group_imp = bin_hier["group"].cpu().numpy()       # (batch, 4)
            feat_imp = bin_hier["global_flat"].cpu().numpy()   # (batch, 72)

            # ── Stage 2: Attack classification (only for predicted attacks) ──
            atk_mask = bin_preds == 1
            if atk_mask.any():
                X_attacks = X_flat[atk_mask]
                atk_logits, atk_explanations = atk_model(
                    X_attacks, return_explanations=True
                )
                atk_preds = atk_logits.argmax(dim=1) + 1  # offset for BENIGN=0
                atk_conf = torch.softmax(atk_logits, dim=1).max(dim=1).values

                # Compute attention rollout for attack model
                atk_hier = rollout_tool.hierarchical_importance(
                    atk_explanations["raw_intra_attn"],
                    atk_explanations["raw_inter_attn"],
                    group_names,
                )

                # Fill in attack predictions
                final_preds[atk_mask] = atk_preds.cpu()
                confidences[atk_mask.cpu().numpy()] = atk_conf.cpu().numpy()
                group_imp[atk_mask.cpu().numpy()] = atk_hier["group"].cpu().numpy()
                feat_imp[atk_mask.cpu().numpy()] = atk_hier["global_flat"].cpu().numpy()

            # Merge labels
            y_true_merged = merge_labels(batch_labels)

            all_y_true.extend(y_true_merged.tolist())
            all_y_pred.extend(final_preds.tolist())
            all_stage1_pred.extend(bin_preds.cpu().tolist())
            all_confidence.extend(confidences.tolist())
            all_attn_group.append(group_imp)
            all_attn_feat.append(feat_imp)

            batch_count += 1
            if batch_count % 50 == 0:
                print(f"  Processed {batch_count * 64} samples...")

    # ── Aggregate ────────────────────────────────────────────────────────
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    stage1_pred = np.array(all_stage1_pred)
    confidence = np.array(all_confidence)
    attn_group = np.vstack(all_attn_group)  # (N, 4)
    attn_feat = np.vstack(all_attn_feat)    # (N, 72)

    # ── Compute confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_MERGED)))
    misclassified_mask = y_true != y_pred
    n_correct = (y_true == y_pred).sum()
    n_misclass = misclassified_mask.sum()

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples:     {len(y_true)}")
    print(f"Correct:           {n_correct} ({100*n_correct/len(y_true):.2f}%)")
    print(f"Misclassified:     {n_misclass} ({100*n_misclass/len(y_true):.2f}%)")
    print(f"Stage 1 (binary):  {(stage1_pred==1).sum()} predicted attack, "
          f"{(stage1_pred==0).sum()} predicted benign")
    print()

    # Top-K off-diagonal confusion pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    top_k = 10
    flat_indices = np.argsort(cm_copy.ravel())[::-1][:top_k]
    print(f"Top-{top_k} confusion pairs:")
    for idx in flat_indices:
        i, j = divmod(idx, NUM_MERGED)
        count = cm_copy[i, j]
        if count == 0:
            break
        print(f"  {MERGED_NAMES[i]:20s} → {MERGED_NAMES[j]:20s}: {count:>6d}")

    # Per-class summary
    print(f"\nPer-class classification report:")
    print(classification_report(y_true, y_pred, target_names=MERGED_NAMES, 
                                 digits=4, zero_division=0))

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = args.out
    np.savez_compressed(
        output_path,
        y_true=y_true,
        y_pred=y_pred,
        stage1_pred=stage1_pred,
        confidence=confidence,
        attn_group=attn_group,
        attn_feat=attn_feat,
        confusion_matrix=cm,
        group_names=np.array(group_names),
        class_names=np.array(MERGED_NAMES),
    )

    print(f"\n✓ Saved to: {output_path}")
    print(f"  y_true shape:     {y_true.shape}")
    print(f"  attn_group shape: {attn_group.shape}")
    print(f"  attn_feat shape:  {attn_feat.shape}")
    print(f"  confusion_matrix: {cm.shape}")

    # Also save a quick JSON summary for easy reading
    summary = {
        "total_samples": int(len(y_true)),
        "correct": int(n_correct),
        "misclassified": int(n_misclass),
        "accuracy": float(n_correct / len(y_true)),
        "top_confusion_pairs": [],
    }
    for idx in flat_indices:
        i, j = divmod(idx, NUM_MERGED)
        count = int(cm_copy[i, j])
        if count == 0:
            break
        summary["top_confusion_pairs"].append({
            "true": MERGED_NAMES[i],
            "predicted": MERGED_NAMES[j],
            "count": count,
        })

    summary["stage1_threshold"] = float(THR)
    with open(args.out.replace(".npz", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved next to {args.out}")


if __name__ == "__main__":
    main()

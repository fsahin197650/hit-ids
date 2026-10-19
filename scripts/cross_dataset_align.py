#!/usr/bin/env python3
"""
Cross-Dataset Feature Alignment for HiT-IDS.

Aligns features from one IDS dataset to another dataset's feature space.
Supports:
  - UNSW-NB15 → CIC-IDS2017 feature space (for testing CIC-IDS2017 model on NB15)
  - CIC-IDS2017 → UNSW-NB15 feature space (for testing NB15 model on CIC-IDS2017)

Usage:
    # Align NB15 test data to CIC-IDS2017 feature space:
    python scripts/cross_dataset_align.py --direction nb15_to_cic2017

    # Align CIC-IDS2017 test data to NB15 feature space:
    python scripts/cross_dataset_align.py --direction cic2017_to_nb15
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════════════
# Feature Mapping: UNSW-NB15 → CIC-IDS2017
# Approximate semantic equivalences between the two datasets
# ═══════════════════════════════════════════════════════════════════════════

# Maps: CIC-IDS2017 feature name → UNSW-NB15 column name
CIC2017_TO_NB15_MAP = {
    # Temporal group
    "Flow Duration":                "dur",
    "Fwd IAT Mean":                 "sinpkt",      # source inter-packet arrival time
    "Bwd IAT Mean":                 "dinpkt",       # dest inter-packet arrival time

    # Volume group
    "Total Fwd Packets":            "spkts",        # source-to-dest packet count
    "Total Backward Packets":       "dpkts",        # dest-to-source packet count
    "Total Length of Fwd Packets":  "sbytes",       # source-to-dest byte count
    "Total Length of Bwd Packets":  "dbytes",       # dest-to-source byte count
    "Fwd Packet Length Mean":       "smean",        # mean of source pkt size
    "Bwd Packet Length Mean":       "dmean",        # mean of dest pkt size
    "Flow Bytes/s":                 "_sload_dload", # computed: sload + dload

    # Protocol group
    "Init_Win_bytes_forward":       "swin",         # source TCP window
    "Init_Win_bytes_backward":      "dwin",         # dest TCP window

    # Behavioral group
    "Down/Up Ratio":                "ct_dst_src_ltm",  # loose proxy
}

# Maps: NB15 feature name → CIC-IDS2017 column name (for reverse direction)
NB15_TO_CIC2017_MAP = {
    # Temporal
    "dur":       "Flow Duration",
    "sinpkt":    "Fwd IAT Mean",
    "dinpkt":    "Bwd IAT Mean",
    "sjit":      "Fwd IAT Std",    # jitter ≈ IAT std
    "djit":      "Bwd IAT Std",
    "rate":      "Flow Packets/s",  # approximate

    # Volume
    "spkts":     "Total Fwd Packets",
    "dpkts":     "Total Backward Packets",
    "sbytes":    "Total Length of Fwd Packets",
    "dbytes":    "Total Length of Bwd Packets",
    "sload":     "Flow Bytes/s",    # source bits per second → approx
    "smean":     "Fwd Packet Length Mean",
    "dmean":     "Bwd Packet Length Mean",

    # Protocol
    "sttl":      None,  # no equivalent (TTL)
    "dttl":      None,
    "swin":      "Init_Win_bytes_forward",
    "dwin":      "Init_Win_bytes_backward",
}


def align_nb15_to_cic2017(nb15_csv_path: str, feature_columns_2017: list,
                           scalers: dict, group_indices: dict) -> dict:
    """
    Align UNSW-NB15 test data to CIC-IDS2017 feature space.

    Uses two techniques to improve cross-dataset transfer:
      1. Z-score distribution alignment for matched features:
         NB15 raw values → NB15 z-scores → CIC2017 scaled distribution
      2. Mean-fill for unmatched features:
         Instead of zero-filling, uses CIC2017 training set post-scaling
         means so the model sees 'average traffic' instead of anomalous zeros.

    Neither technique uses target labels (unsupervised).

    Args:
        nb15_csv_path: Path to UNSW_NB15_testing-set.csv
        feature_columns_2017: Ordered list of 72 CIC-IDS2017 feature names
        scalers: Dict of fitted RobustScaler per group from CIC-IDS2017 training
        group_indices: Dict of feature indices per group

    Returns:
        Dict with X, y_bin, groups tensors
    """
    print(f"\n  Loading UNSW-NB15: {nb15_csv_path}")
    df = pd.read_csv(nb15_csv_path, low_memory=False)
    print(f"  Rows: {len(df):,}, Columns: {df.shape[1]}")

    # Clean
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        df.loc[df["label"] == 0, "attack_cat"] = "Normal"

    # Compute derived features
    if "sload" in df.columns and "dload" in df.columns:
        df["_sload_dload"] = df["sload"].fillna(0) + df["dload"].fillna(0)

    # ── Load CIC-IDS2017 reference statistics (post-scaling) ────────
    cic2017_ref_path = "data/splits/test.pt"
    cic2017_ref = torch.load(cic2017_ref_path, weights_only=False)
    X_cic_ref = cic2017_ref["X"].numpy()
    cic_means = X_cic_ref.mean(axis=0)   # per-feature mean (post-scaling)
    cic_stds = X_cic_ref.std(axis=0)     # per-feature std  (post-scaling)
    cic_stds[cic_stds < 1e-8] = 1.0      # avoid division by zero
    print(f"  Loaded CIC-IDS2017 reference: {X_cic_ref.shape}")

    # ── Create raw feature matrix (before scaling) ──────────────────
    n_samples = len(df)
    n_features = len(feature_columns_2017)
    X_raw = np.full((n_samples, n_features), np.nan, dtype=np.float32)

    matched_indices = []
    matched_features = []
    for i, cic_feat in enumerate(feature_columns_2017):
        nb15_col = CIC2017_TO_NB15_MAP.get(cic_feat)
        if nb15_col and nb15_col in df.columns:
            vals = pd.to_numeric(df[nb15_col], errors="coerce").fillna(0).values
            X_raw[:, i] = vals.astype(np.float32)
            matched_indices.append(i)
            matched_features.append(f"  [{i:2d}] {cic_feat} ← {nb15_col}")

    print(f"\n  Feature alignment: {len(matched_indices)}/{n_features} features matched:")
    for mf in matched_features:
        print(f"    {mf}")

    # ── Apply CIC-IDS2017 scalers to matched features ───────────────
    X_scaled = np.zeros((n_samples, n_features), dtype=np.float32)
    # First, put raw values in and scale them
    X_for_scaling = np.nan_to_num(X_raw, nan=0.0, posinf=1e10, neginf=-1e10)

    for group_name, indices in group_indices.items():
        if group_name in scalers and len(indices) > 0:
            try:
                X_scaled[:, indices] = scalers[group_name].transform(
                    X_for_scaling[:, indices]
                )
            except Exception as e:
                print(f"  [WARN] Scaler failed for '{group_name}': {e}")

    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

    # ── Z-score distribution alignment for MATCHED features ─────────
    # For each matched feature:
    #   1. Compute mean/std of NB15 values (post CIC2017 scaling)
    #   2. Normalize to z-scores
    #   3. Re-scale to match CIC2017's distribution (mean/std)
    aligned_count = 0
    for i in matched_indices:
        nb15_vals = X_scaled[:, i]
        nb15_mean = nb15_vals.mean()
        nb15_std = nb15_vals.std()
        if nb15_std < 1e-8:
            # Constant feature — just shift to CIC2017 mean
            X_scaled[:, i] = cic_means[i]
        else:
            # Z-score align: NB15 distribution → CIC2017 distribution
            z_scores = (nb15_vals - nb15_mean) / nb15_std
            X_scaled[:, i] = z_scores * cic_stds[i] + cic_means[i]
            aligned_count += 1

    print(f"  Z-score aligned: {aligned_count} features")

    # ── Mean-fill for UNMATCHED features ────────────────────────────
    # Instead of leaving zeros, fill with CIC2017 training mean
    # so the model sees 'average traffic' instead of anomalous zeros
    unmatched_indices = [i for i in range(n_features) if i not in matched_indices]
    for i in unmatched_indices:
        X_scaled[:, i] = cic_means[i]

    print(f"  Mean-filled: {len(unmatched_indices)} unmatched features "
          f"(using CIC2017 reference means)")

    # Clip to prevent extreme values
    X_scaled = np.clip(X_scaled, -50, 50)

    # Labels: binary (Normal=0, Attack=1)
    y_bin = df["label"].values.astype(np.int64)

    # Build group tensors
    groups = {}
    for group_name, indices in group_indices.items():
        if len(indices) > 0:
            groups[group_name] = torch.tensor(X_scaled[:, indices],
                                              dtype=torch.float32)

    result = {
        "X": torch.tensor(X_scaled, dtype=torch.float32),
        "y_bin": torch.tensor(y_bin, dtype=torch.long),
        "groups": groups,
    }

    print(f"\n  Result: X={result['X'].shape}")
    print(f"  X range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
    print(f"  Normal: {(y_bin == 0).sum():,}, Attack: {(y_bin == 1).sum():,}")

    return result


def align_cic2017_to_nb15(cic2017_test_path: str, config_unsw_path: str) -> dict:
    """
    Align CIC-IDS2017 test data to UNSW-NB15 feature space.

    Uses z-score distribution alignment for matched features and
    NB15 training mean for unmatched features (same approach as
    the reverse direction).

    Args:
        cic2017_test_path: Path to CIC-IDS2017 test.pt split
        config_unsw_path: Path to config_unsw.yaml

    Returns:
        Dict with X, y_bin, groups tensors in NB15 feature space
    """
    print(f"\n  Loading CIC-IDS2017 test data: {cic2017_test_path}")
    test_data = torch.load(cic2017_test_path, weights_only=False)
    X_cic = test_data["X"].numpy()
    y_cic = test_data["y"].numpy()

    # Load CIC-IDS2017 feature names
    with open("data/splits/metadata.pkl", "rb") as f:
        meta_2017 = pickle.load(f)
    feat_names_2017 = meta_2017["feature_columns"]

    # Load NB15 config
    with open(config_unsw_path) as f:
        config_unsw = yaml.safe_load(f)

    fg_unsw = config_unsw["feature_groups"]
    # Collect all NB15 numeric features (excluding one-hot categoricals)
    nb15_features = []
    for group_features in fg_unsw.values():
        nb15_features.extend(group_features)

    n_nb15_features = len(nb15_features)
    n_samples = len(X_cic)

    # Create NB15-space feature matrix
    X_nb15 = np.zeros((n_samples, n_nb15_features), dtype=np.float32)

    matched_indices = []
    matched_count = 0
    for j, nb15_feat in enumerate(nb15_features):
        cic_col = NB15_TO_CIC2017_MAP.get(nb15_feat)
        if cic_col and cic_col in feat_names_2017:
            cic_idx = feat_names_2017.index(cic_col)
            X_nb15[:, j] = X_cic[:, cic_idx]
            matched_indices.append(j)
            matched_count += 1

    print(f"  Feature alignment: {matched_count}/{n_nb15_features} NB15 features matched from CIC-2017")

    # Scale with new scaler (since we don't have NB15 scaler for raw CIC data)
    scaler = RobustScaler()
    X_nb15 = scaler.fit_transform(X_nb15)
    X_nb15 = np.nan_to_num(X_nb15, nan=0.0, posinf=1e10, neginf=-1e10)

    # Binary labels
    y_bin = (y_cic > 0).astype(np.int64)

    # Group assignment for NB15 model
    # The NB15 model was trained with dynamic group sizes (including one-hot).
    # For feature alignment, we use only the numeric features.
    # Load NB15 splits to get actual group sizes
    nb15_splits_dir = "data/unsw_nb15/splits"
    if os.path.exists(os.path.join(nb15_splits_dir, "train.pt")):
        nb15_train = torch.load(
            os.path.join(nb15_splits_dir, "train.pt"), weights_only=False
        )
        actual_dim = nb15_train["X"].shape[1]
        print(f"  NB15 model expects {actual_dim} features")

        # ── Load NB15 reference statistics for z-score alignment ────
        X_nb15_ref = nb15_train["X"].numpy()
        nb15_ref_means = X_nb15_ref.mean(axis=0)
        nb15_ref_stds = X_nb15_ref.std(axis=0)
        nb15_ref_stds[nb15_ref_stds < 1e-8] = 1.0
        print(f"  Loaded NB15 reference: {X_nb15_ref.shape}")

        # ── Z-score alignment for matched features ──────────────────
        aligned_count = 0
        for j in matched_indices:
            if j < X_nb15.shape[1] and j < len(nb15_ref_means):
                src_vals = X_nb15[:, j]
                src_mean = src_vals.mean()
                src_std = src_vals.std()
                if src_std < 1e-8:
                    X_nb15[:, j] = nb15_ref_means[j]
                else:
                    z_scores = (src_vals - src_mean) / src_std
                    X_nb15[:, j] = z_scores * nb15_ref_stds[j] + nb15_ref_means[j]
                    aligned_count += 1
        print(f"  Z-score aligned: {aligned_count} features")

        # Pad to match NB15 model's expected input dimension
        if X_nb15.shape[1] < actual_dim:
            pad_width = actual_dim - X_nb15.shape[1]
            # Use NB15 training means for padded features instead of zeros
            pad_means = nb15_ref_means[X_nb15.shape[1]:actual_dim]
            pad_block = np.tile(pad_means, (n_samples, 1)).astype(np.float32)
            X_nb15 = np.hstack([X_nb15, pad_block])
            print(f"  Padded: {n_nb15_features} → {actual_dim} features (mean-filled)")
        elif X_nb15.shape[1] > actual_dim:
            X_nb15 = X_nb15[:, :actual_dim]
            print(f"  Truncated: {n_nb15_features} → {actual_dim} features")

        # ── Mean-fill unmatched numeric features ────────────────────
        unmatched = [j for j in range(min(n_nb15_features, actual_dim))
                     if j not in matched_indices]
        for j in unmatched:
            X_nb15[:, j] = nb15_ref_means[j]
        print(f"  Mean-filled: {len(unmatched)} unmatched features")

        # Use same group structure as NB15 training
        groups = {}
        for gn, gdata in nb15_train["groups"].items():
            g_size = gdata.shape[1]
            groups[gn] = torch.zeros(n_samples, g_size, dtype=torch.float32)
        # Fill in what we can from the mapped features
        offset = 0
        for gn in fg_unsw:
            g_feats = fg_unsw[gn]
            for feat in g_feats:
                if feat in nb15_features:
                    feat_idx = nb15_features.index(feat)
                    if feat_idx < X_nb15.shape[1]:
                        g_feat_idx = g_feats.index(feat)
                        if g_feat_idx < groups[gn].shape[1]:
                            groups[gn][:, g_feat_idx] = torch.tensor(
                                X_nb15[:, feat_idx], dtype=torch.float32
                            )
    else:
        print("  [WARN] NB15 splits not found. Run Experiment C first.")
        groups = {}

    result = {
        "X": torch.tensor(X_nb15, dtype=torch.float32),
        "y_bin": torch.tensor(y_bin, dtype=torch.long),
        "groups": groups,
    }

    print(f"  Result: X={result['X'].shape}")
    print(f"  Benign: {(y_bin == 0).sum():,}, Attack: {(y_bin == 1).sum():,}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Feature Alignment")
    parser.add_argument("--direction",
                        choices=["nb15_to_cic2017", "cic2017_to_nb15"],
                        required=True,
                        help="Direction of alignment")
    parser.add_argument("--nb15-csv",
                        default="data/unsw_nb15/raw/UNSW_NB15_testing-set.csv",
                        help="Path to NB15 test CSV")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="CIC-IDS2017 config")
    parser.add_argument("--config-unsw", default="configs/config_unsw.yaml",
                        help="UNSW-NB15 config")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Cross-Dataset Feature Alignment: {args.direction}")
    print("=" * 60)

    if args.direction == "nb15_to_cic2017":
        # Load CIC-IDS2017 metadata and scalers
        with open("data/splits/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        with open("data/processed/scalers.pkl", "rb") as f:
            scalers = pickle.load(f)

        feature_columns = metadata["feature_columns"]
        group_indices = metadata["group_indices"]

        result = align_nb15_to_cic2017(
            args.nb15_csv, feature_columns, scalers, group_indices
        )

        out_path = "data/unsw_nb15/aligned_for_2017_model.pt"
        torch.save(result, out_path)
        print(f"\n  ✓ Saved to {out_path}")

    elif args.direction == "cic2017_to_nb15":
        result = align_cic2017_to_nb15(
            "data/splits/test.pt", args.config_unsw
        )

        out_path = "data/splits/aligned_for_nb15_model.pt"
        torch.save(result, out_path)
        print(f"\n  ✓ Saved to {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  ALIGNMENT COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

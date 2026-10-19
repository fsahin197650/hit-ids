#!/usr/bin/env python3
"""
CIC-IDS2018 Preprocessing for Cross-Dataset Validation.

Loads raw CIC-IDS2018 CSV files, renames columns to match CIC-IDS2017
feature names, applies the same RobustScaler, and produces an aligned
.pt file for direct evaluation with the CIC-IDS2017-trained HiT-IDS model.

Also produces a standalone .pt split for in-domain training (Experiment D).

Usage:
    # For cross-dataset transfer (align to CIC-IDS2017 model):
    python scripts/preprocess_cicids2018.py --mode align

    # For in-domain training on CIC-IDS2018:
    python scripts/preprocess_cicids2018.py --mode train
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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════════════
# CIC-IDS2018 → CIC-IDS2017 Column Rename Mapping
# Both datasets use CICFlowMeter but with slightly different column names
# ═══════════════════════════════════════════════════════════════════════════

COLUMN_RENAME_MAP = {
    "Dst Port":              "Destination Port",
    "Tot Fwd Pkts":          "Total Fwd Packets",
    "Tot Bwd Pkts":          "Total Backward Packets",
    "TotLen Fwd Pkts":       "Total Length of Fwd Packets",
    "TotLen Bwd Pkts":       "Total Length of Bwd Packets",
    "Fwd Pkt Len Max":       "Fwd Packet Length Max",
    "Fwd Pkt Len Min":       "Fwd Packet Length Min",
    "Fwd Pkt Len Mean":      "Fwd Packet Length Mean",
    "Fwd Pkt Len Std":       "Fwd Packet Length Std",
    "Bwd Pkt Len Max":       "Bwd Packet Length Max",
    "Bwd Pkt Len Min":       "Bwd Packet Length Min",
    "Bwd Pkt Len Mean":      "Bwd Packet Length Mean",
    "Bwd Pkt Len Std":       "Bwd Packet Length Std",
    "Flow Byts/s":           "Flow Bytes/s",
    "Flow Pkts/s":           "Flow Packets/s",
    "Fwd IAT Tot":           "Fwd IAT Total",
    "Bwd IAT Tot":           "Bwd IAT Total",
    "Fwd Header Len":        "Fwd Header Length",
    "Bwd Header Len":        "Bwd Header Length",
    "Fwd Pkts/s":            "Fwd Packets/s",
    "Bwd Pkts/s":            "Bwd Packets/s",
    "Pkt Len Min":           "Min Packet Length",
    "Pkt Len Max":           "Max Packet Length",
    "Pkt Len Mean":          "Packet Length Mean",
    "Pkt Len Std":           "Packet Length Std",
    "Pkt Len Var":           "Packet Length Variance",
    "FIN Flag Cnt":          "FIN Flag Count",
    "SYN Flag Cnt":          "SYN Flag Count",
    "RST Flag Cnt":          "RST Flag Count",
    "PSH Flag Cnt":          "PSH Flag Count",
    "ACK Flag Cnt":          "ACK Flag Count",
    "URG Flag Cnt":          "URG Flag Count",
    "ECE Flag Cnt":          "ECE Flag Count",
    "Fwd Seg Size Avg":      "Avg Fwd Segment Size",
    "Bwd Seg Size Avg":      "Avg Bwd Segment Size",
    "Fwd Byts/b Avg":        "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg":        "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg":      "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg":        "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg":        "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg":      "Bwd Avg Bulk Rate",
    "Init Fwd Win Byts":     "Init_Win_bytes_forward",
    "Init Bwd Win Byts":     "Init_Win_bytes_backward",
    "Fwd Act Data Pkts":     "act_data_pkt_fwd",
    "Fwd Seg Size Min":      "min_seg_size_forward",
    "Subflow Fwd Pkts":      "Subflow Fwd Packets",
    "Subflow Fwd Byts":      "Subflow Fwd Bytes",
    "Subflow Bwd Pkts":      "Subflow Bwd Packets",
    "Subflow Bwd Byts":      "Subflow Bwd Bytes",
}

# CIC-IDS2018 → CIC-IDS2017 label mapping (for multi-class transfer)
LABEL_MAP_2018_TO_2017 = {
    "Benign": 0,        # BENIGN
    "Bot": 1,           # Bot
    "Infilteration": 9, # Infiltration (note: 2018 spells it differently)
}

# Columns present in 2018 but NOT in 2017 feature groups (to drop)
DROP_EXTRA_2018 = [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Pkt Size Avg", "Protocol", "Timestamp",
]


def load_cicids2018(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate all CIC-IDS2018 CSV files."""
    csv_files = sorted([
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.endswith(".csv")
    ])

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"  Loading {len(csv_files)} CSV files from {raw_dir}:")
    dfs = []
    for f in csv_files:
        print(f"    {os.path.basename(f)}...", end=" ")
        df = pd.read_csv(f, low_memory=False)
        print(f"{len(df):,} rows")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(df):,} rows, {df.shape[1]} columns")
    return df


def clean_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and rename columns to match CIC-IDS2017."""
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop extra 2018 columns
    drop_cols = [c for c in DROP_EXTRA_2018 if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Rename to CIC-IDS2017 names
    rename_applicable = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_applicable)
    print(f"  Renamed {len(rename_applicable)} columns to CIC-IDS2017 format")

    # Clean labels
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()

    # Convert all non-Label columns to numeric (catches string "Infinity")
    for col in df.columns:
        if col != "Label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace inf → NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in numeric columns
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    print(f"  Dropped {before - len(df):,} rows with NaN/inf")

    # Clip extreme values to prevent float32 overflow during scaling
    clip_bound = np.finfo(np.float32).max / 100  # ~3.4e36
    df[numeric_cols] = df[numeric_cols].clip(-clip_bound, clip_bound)
    print(f"  Clipped extreme values to ±{clip_bound:.1e}")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df):,} duplicate rows")

    return df


def align_to_2017(df: pd.DataFrame, feature_columns_2017: list,
                  scalers: dict, group_indices: dict) -> dict:
    """
    Align CIC-IDS2018 data to CIC-IDS2017 feature space.

    Creates a 72-dimensional feature vector matching the CIC-IDS2017
    feature order. Missing features are zero-filled.
    """
    n_samples = len(df)
    n_features = len(feature_columns_2017)
    X = np.zeros((n_samples, n_features), dtype=np.float32)

    matched = 0
    for i, feat in enumerate(feature_columns_2017):
        if feat in df.columns:
            vals = pd.to_numeric(df[feat], errors="coerce").fillna(0).values
            X[:, i] = vals.astype(np.float32)
            matched += 1

    print(f"  Feature alignment: {matched}/{n_features} features matched")

    # Clip extreme values before scaling to prevent float32 overflow
    clip_bound = np.finfo(np.float32).max / 100
    X = np.clip(X, -clip_bound, clip_bound)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply CIC-IDS2017 scalers
    for group_name, indices in group_indices.items():
        if group_name in scalers and len(indices) > 0:
            try:
                X[:, indices] = scalers[group_name].transform(X[:, indices])
                print(f"  ✓ Scaled group '{group_name}' ({len(indices)} features)")
            except Exception as e:
                print(f"  [WARN] Scaler failed for group '{group_name}': {e}")

    # Handle any remaining NaN/inf from scaling
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    # Labels
    labels = df["Label"].values
    y_bin = np.array([0 if l == "Benign" else 1 for l in labels], dtype=np.int64)
    y_multi = np.array([
        LABEL_MAP_2018_TO_2017.get(l, -1) for l in labels
    ], dtype=np.int64)

    # Build group tensors
    groups = {}
    for group_name, indices in group_indices.items():
        if len(indices) > 0:
            groups[group_name] = torch.tensor(X[:, indices], dtype=torch.float32)

    result = {
        "X": torch.tensor(X, dtype=torch.float32),
        "y_bin": torch.tensor(y_bin, dtype=torch.long),
        "y_multi": torch.tensor(y_multi, dtype=torch.long),
        "groups": groups,
        "label_names": labels,
    }

    print(f"  Benign: {(y_bin == 0).sum():,}, Attack: {(y_bin == 1).sum():,}")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        print(f"    {lbl}: {cnt:,}")

    return result


def prepare_for_training(df: pd.DataFrame, feature_columns_2017: list,
                         group_indices: dict, seed: int = 42) -> dict:
    """
    Prepare CIC-IDS2018 for in-domain HiT-IDS training.
    Uses CIC-IDS2017 feature ordering for architecture compatibility.
    """
    n_features = len(feature_columns_2017)
    X = np.zeros((len(df), n_features), dtype=np.float32)

    for i, feat in enumerate(feature_columns_2017):
        if feat in df.columns:
            X[:, i] = pd.to_numeric(df[feat], errors="coerce").fillna(0).values.astype(np.float32)

    # Clip extreme values before scaling
    clip_bound = np.finfo(np.float32).max / 100
    X = np.clip(X, -clip_bound, clip_bound)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Labels for in-domain training
    label_map = {"Benign": 0, "Bot": 1, "Infilteration": 2}
    labels = df["Label"].values
    y = np.array([label_map.get(l, -1) for l in labels], dtype=np.int64)

    # Remove unknown labels
    valid = y >= 0
    X, y = X[valid], y[valid]

    # Stratified split: 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Scale per group (fit on train)
    scalers = {}
    for group_name, indices in group_indices.items():
        if len(indices) > 0:
            scaler = RobustScaler()
            X_train[:, indices] = scaler.fit_transform(X_train[:, indices])
            X_test[:, indices] = scaler.transform(X_test[:, indices])
            scalers[group_name] = scaler

    # Handle NaN/inf after scaling
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)

    # Build group tensors
    def make_groups(X_data):
        groups = {}
        for gn, idx in group_indices.items():
            if len(idx) > 0:
                groups[gn] = torch.tensor(X_data[:, idx], dtype=torch.float32)
        return groups

    result = {
        "train": {
            "X": torch.tensor(X_train, dtype=torch.float32),
            "y": torch.tensor(y_train, dtype=torch.long),
            "y_bin": torch.tensor((y_train > 0).astype(np.int64), dtype=torch.long),
            "groups": make_groups(X_train),
        },
        "test": {
            "X": torch.tensor(X_test, dtype=torch.float32),
            "y": torch.tensor(y_test, dtype=torch.long),
            "y_bin": torch.tensor((y_test > 0).astype(np.int64), dtype=torch.long),
            "groups": make_groups(X_test),
        },
        "scalers": scalers,
        "label_map": label_map,
        "num_classes": len(label_map),
    }

    print(f"  Train: {len(y_train):,}, Test: {len(y_test):,}")
    for lbl_name, lbl_id in label_map.items():
        tr_cnt = (y_train == lbl_id).sum()
        te_cnt = (y_test == lbl_id).sum()
        print(f"    {lbl_name}: train={tr_cnt:,}, test={te_cnt:,}")

    return result


def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2018 Preprocessing")
    parser.add_argument("--mode", choices=["align", "train", "both"],
                        default="both",
                        help="'align' for cross-dataset, 'train' for in-domain, 'both' for both")
    parser.add_argument("--raw-dir", default="data/cicids2018/raw",
                        help="Directory with raw CIC-IDS2018 CSVs")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="CIC-IDS2017 config for feature ordering")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  CIC-IDS2018 Preprocessing")
    print("=" * 60)

    # Load CIC-IDS2017 metadata
    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open("data/splits/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    feature_columns_2017 = metadata["feature_columns"]
    group_indices = metadata["group_indices"]

    print(f"  CIC-IDS2017 feature space: {len(feature_columns_2017)} features")
    print(f"  Groups: {list(group_indices.keys())}")

    # Load and clean CIC-IDS2018
    print(f"\n  Loading CIC-IDS2018...")
    df = load_cicids2018(args.raw_dir)
    df = clean_and_rename(df)

    print(f"\n  Final dataset: {len(df):,} rows")
    print(f"  Label distribution:")
    for lbl, cnt in df["Label"].value_counts().items():
        print(f"    {lbl}: {cnt:,}")

    # Mode: align (cross-dataset transfer)
    if args.mode in ("align", "both"):
        print(f"\n{'=' * 60}")
        print(f"  Mode: ALIGN (Cross-Dataset Transfer)")
        print(f"{'=' * 60}")

        with open("data/processed/scalers.pkl", "rb") as f:
            scalers_2017 = pickle.load(f)

        aligned = align_to_2017(df, feature_columns_2017, scalers_2017, group_indices)

        out_path = "data/cicids2018/aligned_for_2017_model.pt"
        torch.save(aligned, out_path)
        print(f"\n  ✓ Saved aligned data to {out_path}")
        print(f"    X shape: {aligned['X'].shape}")

    # Mode: train (in-domain training)
    if args.mode in ("train", "both"):
        print(f"\n{'=' * 60}")
        print(f"  Mode: TRAIN (In-Domain Splits)")
        print(f"{'=' * 60}")

        splits = prepare_for_training(
            df, feature_columns_2017, group_indices, seed=args.seed
        )

        out_dir = "data/cicids2018/splits"
        os.makedirs(out_dir, exist_ok=True)

        torch.save(splits["train"], os.path.join(out_dir, "train.pt"))
        torch.save(splits["test"], os.path.join(out_dir, "test.pt"))
        with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
            pickle.dump(splits["scalers"], f)

        print(f"\n  ✓ Saved training splits to {out_dir}/")

    print(f"\n{'=' * 60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

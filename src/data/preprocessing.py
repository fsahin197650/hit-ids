"""
Full preprocessing pipeline for CIC-IDS2017 dataset.

Steps:
    1. Load all CSV files and concatenate
    2. Strip whitespace from column names
    3. Drop identifier columns (Flow ID, IPs, Ports, Timestamp)
    4. Replace inf → NaN, drop NaN rows
    5. Remove exact duplicate rows
    6. Label encoding (string labels → integers)
    7. Assign features to semantic groups (G1-G4)
    8. Apply RobustScaler per group (fit on train only)
    9. Stratified split: 70% train, 15% val, 15% test
    10. Save processed data
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

from .feature_groups import FeatureGroups


class CICIDSPreprocessor:
    """
    End-to-end preprocessing for CIC-IDS2017.

    Usage:
        >>> preprocessor = CICIDSPreprocessor.from_config("configs/config.yaml")
        >>> preprocessor.run()   # Loads CSV → cleans → groups → scales → splits → saves
    """

    def __init__(
        self,
        raw_data_path: str,
        processed_data_path: str,
        splits_path: str,
        feature_groups: FeatureGroups,
        label_column: str = "Label",
        label_map: Optional[Dict[str, int]] = None,
        drop_columns: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            raw_data_path: Directory containing raw CSV files.
            processed_data_path: Directory to save processed data.
            splits_path: Directory to save train/val/test splits.
            feature_groups: FeatureGroups instance defining G1-G4.
            label_column: Name of the label column.
            label_map: Mapping from string labels to integer indices.
            drop_columns: Columns to drop (identifiers).
            random_state: Random seed for reproducibility.
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.splits_path = Path(splits_path)
        self.feature_groups = feature_groups
        self.label_column = label_column
        self.label_map = label_map or {}
        self.drop_columns = drop_columns or FeatureGroups.DROP_COLUMNS
        self.random_state = random_state

        # Will be populated during processing
        self.label_encoder: Optional[LabelEncoder] = None
        self.scalers: Dict[str, RobustScaler] = {}
        self.stats: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, config_path: str) -> "CICIDSPreprocessor":
        """Create preprocessor from YAML config file."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        feature_groups = FeatureGroups.from_dict(cfg)

        return cls(
            raw_data_path=cfg["paths"]["raw_data"],
            processed_data_path=cfg["paths"]["processed_data"],
            splits_path=cfg["paths"]["splits"],
            feature_groups=feature_groups,
            label_column=cfg["dataset"].get("label_column", "Label"),
            label_map=cfg["dataset"].get("label_map", None),
            drop_columns=cfg["dataset"].get("drop_columns", None),
            random_state=cfg["project"].get("seed", 42),
        )

    # ── Main Pipeline ────────────────────────────────────────────────────

    def run(self, save: bool = True) -> Dict[str, Any]:
        """
        Execute the full preprocessing pipeline.

        Args:
            save: Whether to save processed data to disk.

        Returns:
            Dictionary with train/val/test splits and metadata.
        """
        print("=" * 60)
        print("HiT-IDS Preprocessing Pipeline")
        print("=" * 60)

        # Step 1: Load CSVs
        df = self._load_csvs()

        # Step 2: Clean data
        df = self._clean_data(df)

        # Step 3: Encode labels
        df, labels = self._encode_labels(df)

        # Step 4: Get feature columns and validate groups
        feature_cols = [c for c in df.columns if c != self.label_column]
        group_indices = self.feature_groups.get_group_indices(feature_cols)
        self._log_group_info(feature_cols, group_indices)

        # Step 5: Extract feature matrix
        X = df[feature_cols].values.astype(np.float32)
        y = labels

        # Step 6: Stratified split (70/15/15)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp,
            random_state=self.random_state
        )
        print(f"\n[6/8] Stratified split:")
        print(f"  Train: {X_train.shape[0]:,} samples")
        print(f"  Val:   {X_val.shape[0]:,} samples")
        print(f"  Test:  {X_test.shape[0]:,} samples")

        # Step 7: Scale per group (fit on train only)
        X_train, X_val, X_test = self._scale_per_group(
            X_train, X_val, X_test, feature_cols, group_indices
        )

        # Step 8: Build grouped tensors
        splits = self._build_grouped_data(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, group_indices
        )

        # Collect metadata
        metadata = {
            "feature_columns": feature_cols,
            "group_indices": group_indices,
            "group_sizes": self.feature_groups.group_sizes,
            "group_names": self.feature_groups.group_names,
            "num_classes": len(self.label_map) if self.label_map else len(np.unique(y)),
            "label_map": self.label_map,
            "class_counts_train": dict(zip(*np.unique(y_train, return_counts=True))),
        }
        splits["metadata"] = metadata

        # Save
        if save:
            self._save_splits(splits)

        print("\n" + "=" * 60)
        print("✓ Preprocessing complete!")
        print("=" * 60)

        return splits

    # ── Step Functions ───────────────────────────────────────────────────

    def _load_csvs(self) -> pd.DataFrame:
        """Step 1: Load all CSV files from raw_data_path and concatenate."""
        csv_files = sorted(self.raw_data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.raw_data_path}. "
                f"Please download CIC-IDS2017 dataset first."
            )

        print(f"\n[1/8] Loading {len(csv_files)} CSV files...")
        dfs = []
        for csv_file in csv_files:
            print(f"  Loading: {csv_file.name}")
            df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        print(f"  Total rows: {len(df):,}")
        print(f"  Total columns: {df.shape[1]}")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steps 2-5: Clean the DataFrame."""
        initial_rows = len(df)

        # Strip whitespace from column names (CIC-IDS2017 has leading/trailing spaces)
        df.columns = df.columns.str.strip()
        print(f"\n[2/8] Stripped column name whitespace ({df.shape[1]} columns)")

        # Drop identifier columns
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        print(f"[3/8] Dropped {len(cols_to_drop)} identifier columns: {cols_to_drop}")

        # Remove duplicate column names (keep first occurrence)
        dup_mask = df.columns.duplicated(keep='first')
        if dup_mask.any():
            dup_names = df.columns[dup_mask].tolist()
            print(f"  [INFO] Removing {len(dup_names)} duplicate columns: {dup_names[:5]}...")
            df = df.loc[:, ~dup_mask]

        # --- SELECT ONLY NEEDED COLUMNS BEFORE NaN REMOVAL ---
        # This is critical: CIC-IDS2017 has ~78 columns, we only need our 72 + Label.
        # NaN in unused columns would otherwise drop all valid rows.
        needed_features = self.feature_groups.all_features
        needed_cols = [c for c in needed_features if c in df.columns]
        missing_features = [c for c in needed_features if c not in df.columns]

        if missing_features:
            print(f"  [WARNING] {len(missing_features)} features not found in data:")
            for f in missing_features[:10]:
                print(f"    - {f}")
            if len(missing_features) > 10:
                print(f"    ... and {len(missing_features) - 10} more")

        # Keep only needed features + label
        keep_cols = needed_cols + ([self.label_column] if self.label_column in df.columns else [])
        df = df[keep_cols]
        print(f"  Selected {len(needed_cols)} features + label column "
              f"(dropped {df.shape[1] - len(keep_cols) + len(cols_to_drop)} unused cols)")

        # Replace inf → NaN (only on feature columns)
        feature_df = df[needed_cols]
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        df[needed_cols] = feature_df

        nan_count = df[needed_cols].isna().sum().sum()
        nan_pct = nan_count / (len(df) * len(needed_cols)) * 100 if len(df) > 0 else 0
        print(f"[4/8] Replaced inf → NaN. NaN cells in features: {nan_count:,} ({nan_pct:.3f}%)")

        # Drop rows with NaN (only in feature columns)
        df = df.dropna(subset=needed_cols)
        print(f"  Rows after NaN removal: {len(df):,} "
              f"(dropped {initial_rows - len(df):,})")

        if len(df) == 0:
            raise ValueError(
                "All rows dropped during cleaning! Check that feature names in "
                "config.yaml match the CSV column names (after stripping whitespace)."
            )

        # Remove duplicate rows
        before_dedup = len(df)
        df = df.drop_duplicates()
        print(f"[5/8] Removed {before_dedup - len(df):,} duplicate rows. "
              f"Remaining: {len(df):,}")

        return df

    def _encode_labels(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Encode string labels to integers."""
        if self.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found. "
                f"Available: {list(df.columns)}"
            )

        # Ensure Label column is a Series (not DataFrame from duplicate cols)
        label_series = df[self.label_column]
        if isinstance(label_series, pd.DataFrame):
            label_series = label_series.iloc[:, 0]

        # Strip whitespace from labels
        label_series = label_series.astype(str).str.strip()

        # Normalize special characters (CIC-IDS2017 has encoding issues
        # with dashes in 'Web Attack' labels: en-dash, em-dash, corrupted bytes)
        label_series = self._normalize_labels(label_series)
        df[self.label_column] = label_series

        if self.label_map:
            # Normalize label_map keys too
            normalized_map = {
                self._normalize_label_str(k): v
                for k, v in self.label_map.items()
            }
            self.label_map = normalized_map

            # Use provided mapping
            labels = df[self.label_column].map(self.label_map)
            # Check for unmapped labels
            unmapped = df[self.label_column][labels.isna()].unique()
            if len(unmapped) > 0:
                print(f"  [WARNING] Unmapped labels found: {unmapped}")
                # Create mapping for unmapped labels
                next_id = max(self.label_map.values()) + 1
                for label in unmapped:
                    self.label_map[label] = next_id
                    next_id += 1
                labels = df[self.label_column].map(self.label_map)
            labels = labels.values.astype(np.int64)
        else:
            # Auto-encode
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(df[self.label_column])
            self.label_map = {
                label: idx for idx, label in
                enumerate(self.label_encoder.classes_)
            }

        # Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        inv_map = {v: k for k, v in self.label_map.items()}
        print(f"\n[6/8] Label encoding — {len(unique)} classes:")
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            name = inv_map.get(u, f"class_{u}")
            print(f"  {u:2d}: {name:30s} → {c:>8,} ({c/len(labels)*100:.2f}%)")

        return df, labels

    @staticmethod
    def _normalize_label_str(label: str) -> str:
        """Normalize a single label string by replacing dash variants."""
        import re
        # Replace en-dash, em-dash, minus sign, and other dash-like chars
        # Also handle corrupted UTF-8 sequences (common in CIC-IDS2017)
        label = re.sub(r'[\u2013\u2014\u2212\ufffd]', '-', label)
        # Handle common corrupted byte patterns: Â–, â€", Ã¢â‚¬â€œ, etc.
        label = re.sub(r'[Ã¢â‚¬â€"\xc3\xa2\xe2\x80\x93]+', '-', label)
        # Collapse multiple dashes/spaces
        label = re.sub(r'\s*-+\s*', ' - ', label).strip()
        return label

    @staticmethod
    def _normalize_labels(series: pd.Series) -> pd.Series:
        """Normalize all labels in a Series."""
        return series.apply(CICIDSPreprocessor._normalize_label_str)

    def _log_group_info(
        self,
        feature_cols: List[str],
        group_indices: Dict[str, List[int]],
    ) -> None:
        """Log feature group assignment info."""
        total_grouped = sum(len(v) for v in group_indices.values())
        ungrouped = len(feature_cols) - total_grouped

        print(f"\n[7/8] Feature grouping ({total_grouped}/{len(feature_cols)} "
              f"features assigned, {ungrouped} ungrouped):")
        for name, indices in group_indices.items():
            print(f"  {name:12s}: {len(indices):3d} features")

        if ungrouped > 0:
            grouped_set = set()
            for indices in group_indices.values():
                grouped_set.update(indices)
            ungrouped_feats = [
                feature_cols[i] for i in range(len(feature_cols))
                if i not in grouped_set
            ]
            print(f"  [INFO] Ungrouped features: {ungrouped_feats[:10]}...")

    def _scale_per_group(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        feature_cols: List[str],
        group_indices: Dict[str, List[int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply RobustScaler per group (fit on train only)."""
        print(f"\n[8/8] Applying RobustScaler per group...")

        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()

        for group_name, indices in group_indices.items():
            if len(indices) == 0:
                continue
            scaler = RobustScaler()
            X_train_scaled[:, indices] = scaler.fit_transform(X_train[:, indices])
            X_val_scaled[:, indices] = scaler.transform(X_val[:, indices])
            X_test_scaled[:, indices] = scaler.transform(X_test[:, indices])
            self.scalers[group_name] = scaler
            print(f"  {group_name:12s}: scaled {len(indices)} features")

        # Scale ungrouped features together
        grouped_set = set()
        for indices in group_indices.values():
            grouped_set.update(indices)
        ungrouped = [i for i in range(X_train.shape[1]) if i not in grouped_set]

        if ungrouped:
            scaler = RobustScaler()
            X_train_scaled[:, ungrouped] = scaler.fit_transform(X_train[:, ungrouped])
            X_val_scaled[:, ungrouped] = scaler.transform(X_val[:, ungrouped])
            X_test_scaled[:, ungrouped] = scaler.transform(X_test[:, ungrouped])
            self.scalers["ungrouped"] = scaler
            print(f"  {'ungrouped':12s}: scaled {len(ungrouped)} features")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def _build_grouped_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_cols: List[str],
        group_indices: Dict[str, List[int]],
    ) -> Dict[str, Any]:
        """Build grouped tensor data for each split."""
        splits: Dict[str, Any] = {}

        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            split_data = {
                "X": torch.tensor(X, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.long),
                "groups": {},
            }
            for group_name, indices in group_indices.items():
                if len(indices) > 0:
                    split_data["groups"][group_name] = torch.tensor(
                        X[:, indices], dtype=torch.float32
                    )
            splits[split_name] = split_data

        return splits

    def _save_splits(self, splits: Dict[str, Any]) -> None:
        """Save processed data to disk."""
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.splits_path.mkdir(parents=True, exist_ok=True)

        # Save each split
        for split_name in ["train", "val", "test"]:
            split_data = splits[split_name]
            save_path = self.splits_path / f"{split_name}.pt"
            torch.save(split_data, save_path)
            print(f"  Saved {split_name} → {save_path}")

        # Save metadata
        metadata_path = self.splits_path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(splits["metadata"], f)
        print(f"  Saved metadata → {metadata_path}")

        # Save scalers
        scaler_path = self.processed_data_path / "scalers.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scalers, f)
        print(f"  Saved scalers → {scaler_path}")

        # Save label map
        label_map_path = self.processed_data_path / "label_map.pkl"
        with open(label_map_path, "wb") as f:
            pickle.dump(self.label_map, f)
        print(f"  Saved label_map → {label_map_path}")


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CIC-IDS2017 preprocessing")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Create dummy data and run pipeline (for testing)"
    )
    args = parser.parse_args()

    if args.dry_run:
        # Create dummy data for testing
        print("Running dry-run with dummy data...")
        np.random.seed(42)
        fg = FeatureGroups()
        n_samples = 1000
        n_features = fg.total_features

        # Create dummy DataFrame
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        df = pd.DataFrame(data, columns=fg.all_features)
        labels = np.random.choice(
            ["BENIGN", "DDoS", "PortScan", "Bot", "DoS Hulk"],
            size=n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05]
        )
        df["Label"] = labels

        # Save as CSV
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/dummy_test.csv", index=False)
        print("Saved dummy data to data/raw/dummy_test.csv")

    preprocessor = CICIDSPreprocessor.from_config(args.config)
    results = preprocessor.run()
    print(f"\nResults keys: {list(results.keys())}")
    for k in ["train", "val", "test"]:
        print(f"  {k}: X={results[k]['X'].shape}, y={results[k]['y'].shape}")

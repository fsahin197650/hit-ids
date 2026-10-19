#!/usr/bin/env python3
"""
HiT-IDS Preprocessing Entry Point.

Runs the full preprocessing pipeline:
    1. Load CIC-IDS2017 CSVs
    2. Clean and encode
    3. Group features
    4. Scale per group
    5. Stratified split
    6. Optional SMOTE+Tomek resampling
    7. Save splits

Usage:
    python scripts/preprocess.py --config configs/config.yaml
    python scripts/preprocess.py --dry-run  # Test with dummy data
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.feature_groups import FeatureGroups
from src.data.imbalance import ImbalanceHandler
from src.data.preprocessing import CICIDSPreprocessor


def create_dummy_data(config: dict) -> None:
    """Create dummy CSV data for pipeline testing."""
    import pandas as pd

    fg = FeatureGroups.from_dict(config)
    raw_path = config["paths"]["raw_data"]
    os.makedirs(raw_path, exist_ok=True)

    n_samples = 2000
    n_features = fg.total_features

    np.random.seed(config["project"]["seed"])

    # Create feature data
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    # Add some inf values to test cleaning
    data[0, 0] = np.inf
    data[1, 1] = -np.inf

    df = pd.DataFrame(data, columns=fg.all_features)

    # Add identifier columns (will be dropped)
    df["Flow ID"] = range(n_samples)
    df["Source IP"] = "192.168.1.1"
    df["Source Port"] = 80
    df["Destination IP"] = "10.0.0.1"
    df["Destination Port"] = 443
    df["Timestamp"] = "2017-07-04 09:00:00"

    # Add labels with realistic imbalance
    label_probs = {
        "BENIGN": 0.50,
        "DDoS": 0.12,
        "PortScan": 0.10,
        "DoS Hulk": 0.08,
        "DoS GoldenEye": 0.04,
        "DoS slowloris": 0.03,
        "DoS Slowhttptest": 0.03,
        "FTP-Patator": 0.02,
        "SSH-Patator": 0.02,
        "Bot": 0.015,
        "Web Attack \u2013 Brute Force": 0.01,
        "Web Attack \u2013 XSS": 0.01,
        "Infiltration": 0.01,
        "Web Attack \u2013 Sql Injection": 0.005,
        "Heartbleed": 0.005,
    }

    probs = np.array(list(label_probs.values()))
    probs = probs / probs.sum()  # Ensure exact sum to 1.0
    labels = np.random.choice(
        list(label_probs.keys()),
        size=n_samples,
        p=probs,
    )
    df["Label"] = labels

    # Add a few duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    csv_path = os.path.join(raw_path, "dummy_monday.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Created dummy data: {csv_path} ({len(df)} rows)")


def main() -> None:
    """Main preprocessing entry point."""
    parser = argparse.ArgumentParser(
        description="HiT-IDS Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preprocess.py --config configs/config.yaml
  python scripts/preprocess.py --dry-run
  python scripts/preprocess.py --config configs/config.yaml --apply-smote
        """,
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Create dummy data and run pipeline (for testing)",
    )
    parser.add_argument(
        "--apply-smote", action="store_true",
        help="Apply SMOTE+Tomek resampling to training data",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save processed data (for debugging)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"  HiT-IDS Preprocessing Pipeline")
    print(f"  Config: {args.config}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Apply SMOTE: {args.apply_smote}")
    print("=" * 60)

    # Create dummy data if dry-run
    if args.dry_run:
        create_dummy_data(config)

    # Run preprocessing
    start_time = time.time()
    preprocessor = CICIDSPreprocessor.from_config(args.config)
    results = preprocessor.run(save=not args.no_save)

    elapsed = time.time() - start_time
    print(f"\nPreprocessing time: {elapsed:.1f}s")

    # Print summary
    for split_name in ["train", "val", "test"]:
        split = results[split_name]
        print(f"\n{split_name.upper()} split:")
        print(f"  X shape: {split['X'].shape}")
        print(f"  y shape: {split['y'].shape}")
        for gname, gtensor in split["groups"].items():
            print(f"  {gname}: {gtensor.shape}")

    # Optional: Apply SMOTE+Tomek
    if args.apply_smote:
        print("\n" + "=" * 60)
        print("Applying SMOTE+Tomek to training data...")
        print("=" * 60)

        handler = ImbalanceHandler.from_dict(config)

        X_train = results["train"]["X"].numpy()
        y_train = results["train"]["y"].numpy()

        group_indices = results["metadata"]["group_indices"]
        X_res, y_res, group_data_res = handler.resample_grouped(
            X_train, y_train, group_indices
        )

        # Update training data
        results["train"]["X"] = torch.tensor(X_res, dtype=torch.float32)
        results["train"]["y"] = torch.tensor(y_res, dtype=torch.long)
        for name, data in group_data_res.items():
            results["train"]["groups"][name] = torch.tensor(
                data, dtype=torch.float32
            )

        # Re-save training data
        if not args.no_save:
            splits_path = config["paths"]["splits"]
            os.makedirs(splits_path, exist_ok=True)
            torch.save(results["train"], os.path.join(splits_path, "train.pt"))
            print(f"✓ Saved resampled training data")

        print(f"\nResampled training set:")
        print(f"  X shape: {results['train']['X'].shape}")
        print(f"  y shape: {results['train']['y'].shape}")

    print(f"\n✓ Pipeline complete! Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()

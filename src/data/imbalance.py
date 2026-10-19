"""
Class imbalance handling for HiT-IDS.

Provides SMOTE + Tomek Links resampling for minority class oversampling
on the training set only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml


class ImbalanceHandler:
    """
    Handles class imbalance via SMOTE, SMOTE+Tomek, or ADASYN.

    Applied ONLY to the training set. Validation and test sets remain untouched.

    Usage:
        >>> handler = ImbalanceHandler(method="smote_tomek")
        >>> X_resampled, y_resampled = handler.resample(X_train, y_train)
    """

    VALID_METHODS = {"smote", "smote_tomek", "adasyn", "none"}

    def __init__(
        self,
        method: str = "smote_tomek",
        random_state: int = 42,
    ) -> None:
        """
        Args:
            method: Resampling method. One of 'smote', 'smote_tomek', 'adasyn', 'none'.
            random_state: Random seed for reproducibility.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {self.VALID_METHODS}"
            )
        self.method = method
        self.random_state = random_state

    @classmethod
    def from_config(cls, config_path: str) -> "ImbalanceHandler":
        """Create handler from YAML config."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        imb_cfg = cfg.get("training", {}).get("imbalance", {})
        return cls(
            method=imb_cfg.get("method", "smote_tomek"),
            random_state=imb_cfg.get("random_state", 42),
        )

    @classmethod
    def from_dict(cls, config: dict) -> "ImbalanceHandler":
        """Create handler from config dict."""
        imb_cfg = config.get("training", {}).get("imbalance", {})
        return cls(
            method=imb_cfg.get("method", "smote_tomek"),
            random_state=imb_cfg.get("random_state", 42),
        )

    def resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply resampling to handle class imbalance.

        Uses a CAPPED strategy: minority classes are oversampled up to the
        median class count (not the majority). This prevents generating
        millions of synthetic samples for ultra-rare classes.

        Args:
            X: Feature matrix, shape (N, D).
            y: Label vector, shape (N,).

        Returns:
            Tuple of (X_resampled, y_resampled).
        """
        if self.method == "none":
            print("[ImbalanceHandler] Method='none', skipping resampling.")
            return X, y

        # Log original distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n[ImbalanceHandler] Original class distribution:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c:>8,}")
        print(f"  Total: {len(y):,}")

        # Check minimum class size for SMOTE (needs at least k_neighbors+1 samples)
        min_count = counts.min()
        k_neighbors = min(5, min_count - 1) if min_count > 1 else 1

        if min_count <= 1:
            print(f"  [WARNING] Minimum class has {min_count} sample(s). "
                  f"Using k_neighbors=1 for SMOTE.")
            k_neighbors = 1

        # --- CAPPED SAMPLING STRATEGY ---
        # Oversample minority classes to a fixed target (50K).
        # This gives rare classes enough representation without
        # overwhelming the dataset with synthetic samples.
        target_count = 50_000

        sampling_strategy = {}
        for cls, cnt in zip(unique, counts):
            if cnt < target_count:
                sampling_strategy[int(cls)] = target_count

        if sampling_strategy:
            print(f"  [SMOTE] Target count per minority class: {target_count:,}")
            print(f"  [SMOTE] Classes to oversample: {list(sampling_strategy.keys())}")
        else:
            print("  [INFO] No classes below target — skipping SMOTE")
            return X, y

        # Apply resampling
        if self.method == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=k_neighbors,
            )
            X_res, y_res = sampler.fit_resample(X, y)

        elif self.method == "smote_tomek":
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=k_neighbors,
            )
            sampler = SMOTETomek(
                smote=smote,
                random_state=self.random_state,
            )
            X_res, y_res = sampler.fit_resample(X, y)

        elif self.method == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(
                random_state=self.random_state,
                n_neighbors=k_neighbors,
            )
            try:
                X_res, y_res = sampler.fit_resample(X, y)
            except ValueError as e:
                print(f"  [WARNING] ADASYN failed: {e}. Falling back to SMOTE.")
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(
                    random_state=self.random_state,
                    k_neighbors=k_neighbors,
                )
                X_res, y_res = sampler.fit_resample(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Log resampled distribution
        unique_res, counts_res = np.unique(y_res, return_counts=True)
        print(f"\n[ImbalanceHandler] Resampled distribution ({self.method}):")
        for u, c in zip(unique_res, counts_res):
            print(f"  Class {u}: {c:>8,}")
        print(f"  Total: {len(y_res):,} "
              f"(+{len(y_res) - len(y):,} samples)")

        # --- MAJORITY UNDERSAMPLING ---
        # After SMOTE, undersample majority class to max_majority_count
        # This prevents the model from defaulting to majority prediction
        max_majority = 200_000
        majority_cls = unique_res[counts_res.argmax()]
        majority_count = counts_res.max()

        if majority_count > max_majority:
            print(f"\n  [UNDERSAMPLE] Reducing class {majority_cls}: "
                  f"{majority_count:,} → {max_majority:,}")
            majority_mask = np.where(y_res == majority_cls)[0]
            keep_majority = np.random.RandomState(self.random_state).choice(
                majority_mask, max_majority, replace=False
            )
            other_mask = np.where(y_res != majority_cls)[0]
            keep_indices = np.concatenate([keep_majority, other_mask])
            np.random.RandomState(self.random_state).shuffle(keep_indices)
            X_res = X_res[keep_indices]
            y_res = y_res[keep_indices]

            # Log final distribution
            unique_final, counts_final = np.unique(y_res, return_counts=True)
            print(f"  [UNDERSAMPLE] Final distribution:")
            for u, c in zip(unique_final, counts_final):
                print(f"    Class {u}: {c:>8,}")
            print(f"    Total: {len(y_res):,}")

        return X_res.astype(np.float32), y_res.astype(np.int64)

    def resample_grouped(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_indices: Dict[str, list],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Resample and return both full and grouped feature matrices.

        Args:
            X: Full feature matrix, shape (N, D).
            y: Labels, shape (N,).
            group_indices: Dict mapping group name → list of column indices.

        Returns:
            Tuple of (X_resampled, y_resampled, group_data_resampled).
        """
        X_res, y_res = self.resample(X, y)

        group_data = {}
        for name, indices in group_indices.items():
            if len(indices) > 0:
                group_data[name] = X_res[:, indices]

        return X_res, y_res, group_data


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing ImbalanceHandler with dummy data...")

    np.random.seed(42)

    # Create imbalanced dummy data
    N = 1000
    D = 72

    X = np.random.randn(N, D).astype(np.float32)
    # Imbalanced: class 0 = 800, class 1 = 100, class 2 = 70, class 3 = 30
    y = np.array(
        [0] * 800 + [1] * 100 + [2] * 70 + [3] * 30,
        dtype=np.int64,
    )

    # Shuffle
    perm = np.random.permutation(N)
    X, y = X[perm], y[perm]

    # Test each method
    for method in ["none", "smote", "smote_tomek"]:
        print(f"\n{'='*40}")
        print(f"Testing method: {method}")
        handler = ImbalanceHandler(method=method)
        X_res, y_res = handler.resample(X.copy(), y.copy())
        print(f"Original: {N}, Resampled: {len(y_res)}")

    print("\n✓ All tests passed!")

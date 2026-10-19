"""
Leak-free data access for cross-validation experiments.

Background
----------
`data/splits/train.pt` is NOT raw data: it is the SMOTE-oversampled and
BENIGN-undersampled training split produced by `preprocess.py --apply-smote`.
The original `evaluate_cv.py` / `run_ablation.py` concatenated train+val+test
and then ran StratifiedKFold, so every CV test fold contained synthetic
minority samples (e.g. ~10,000 "Heartbleed" rows per fold while the whole real
dataset has 11). All CV numbers produced that way are invalid.

This module provides the correct pipeline:

    raw CSVs  →  clean (drop ids, inf→NaN→drop, dedup)  →  15-class labels
              →  UNSCALED feature matrix X (N, 72), cached on disk
    per fold:    RobustScaler fit on the training fold only
              →  SMOTE / undersampling on the training fold only
              →  evaluate on REAL test-fold rows only

Nothing here touches `data/splits/*.pt`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.preprocessing import RobustScaler

from .preprocessing import CICIDSPreprocessor

# ── 13-class merge (Web Attack sub-types → single class) ─────────────────
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

DEFAULT_CACHE = "data/processed/clean_full.npz"
DEFAULT_COUNTS = "data/processed/clean_full_class_counts.json"


def merge_labels_np(y: np.ndarray) -> np.ndarray:
    lut = np.zeros(max(MERGE_MAP) + 1, dtype=np.int64)
    for k, v in MERGE_MAP.items():
        lut[k] = v
    return lut[y.astype(np.int64)]


def _group_indices_from_config(config: dict, feature_cols: List[str]) -> Dict[str, List[int]]:
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    out = {}
    for gname, feats in config["feature_groups"].items():
        out[gname] = [col_to_idx[f] for f in feats if f in col_to_idx]
    return out


def load_clean_dataset(
    config_path: str = "configs/config.yaml",
    cache_path: str = DEFAULT_CACHE,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """Return the cleaned, UNSCALED, real-only CIC-IDS2017 dataset.

    Returns:
        X            : float32 array (N, 72), columns in feature-group order
                       (temporal | volume | protocol | behavioral), unscaled.
        y15          : int64 array (N,), original 15-class labels.
        feature_cols : list of 72 column names (same order as X).
        group_indices: dict group_name -> list of column indices (contiguous).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cache = Path(cache_path)
    meta_path = cache.with_suffix(".json")

    if cache.exists() and meta_path.exists() and not force_rebuild:
        if verbose:
            print(f"  [cv_data] Loading cached clean dataset: {cache}")
        d = np.load(cache)
        X, y15 = d["X"], d["y15"]
        meta = json.loads(meta_path.read_text())
        feature_cols = meta["feature_columns"]
        group_indices = {k: list(v) for k, v in meta["group_indices"].items()}
    else:
        if verbose:
            print(f"  [cv_data] Building clean dataset from raw CSVs "
                  f"(one-off, cached to {cache})")
        pre = CICIDSPreprocessor.from_config(config_path)
        df = pre._load_csvs()
        df = pre._clean_data(df)
        df, y15 = pre._encode_labels(df)
        feature_cols = [c for c in df.columns if c != pre.label_column]
        X = df[feature_cols].values.astype(np.float32)
        y15 = np.asarray(y15, dtype=np.int64)
        group_indices = _group_indices_from_config(config, feature_cols)

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, y15=y15)
        meta = {
            "n_rows": int(len(y15)),
            "n_features": int(X.shape[1]),
            "feature_columns": feature_cols,
            "group_indices": group_indices,
            "group_sizes": {k: len(v) for k, v in group_indices.items()},
            "label_map": {str(k): int(v) for k, v in pre.label_map.items()},
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        _write_class_counts(y15, pre.label_map, DEFAULT_COUNTS)

    # Sanity: groups must be contiguous blocks in feature-group order so that
    # evaluate_cv.make_hit_groups (offset slicing) is valid.
    offset = 0
    for gname in config["feature_groups"]:
        idx = group_indices[gname]
        assert idx == list(range(offset, offset + len(idx))), (
            f"Group '{gname}' is not a contiguous block starting at {offset}: {idx[:5]}...")
        offset += len(idx)
    assert offset == X.shape[1], f"Grouped features {offset} != matrix width {X.shape[1]}"

    if verbose:
        print(f"  [cv_data] Real rows: {len(y15):,} | features: {X.shape[1]} | "
              f"groups: {[len(group_indices[g]) for g in config['feature_groups']]}")
    return X, y15, feature_cols, group_indices


def _write_class_counts(y15: np.ndarray, label_map: Dict[str, int], path: str) -> None:
    inv = {v: k for k, v in label_map.items()}
    u, c = np.unique(y15, return_counts=True)
    counts15 = {inv.get(int(k), str(k)): int(n) for k, n in zip(u, c)}
    y13 = merge_labels_np(y15)
    u13, c13 = np.unique(y13, return_counts=True)
    counts13 = {MERGED_NAMES[int(k)]: int(n) for k, n in zip(u13, c13)}
    out = {
        "total_rows": int(len(y15)),
        "note": "Real rows only (after inf/NaN drop and de-duplication). "
                "No SMOTE, no undersampling.",
        "counts_15_class": counts15,
        "counts_13_class_merged": counts13,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))


def scale_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    group_indices: Dict[str, List[int]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, RobustScaler]]:
    """Per-group RobustScaler fitted on the training fold ONLY.

    Mirrors CICIDSPreprocessor._scale_per_group, but per CV fold.
    """
    Xtr = X_train.astype(np.float32, copy=True)
    Xte = X_test.astype(np.float32, copy=True)
    scalers: Dict[str, RobustScaler] = {}
    for gname, idx in group_indices.items():
        if not idx:
            continue
        sc = RobustScaler()
        Xtr[:, idx] = sc.fit_transform(X_train[:, idx])
        Xte[:, idx] = sc.transform(X_test[:, idx])
        scalers[gname] = sc
    return Xtr, Xte, scalers


def smoke_subsample(
    X: np.ndarray,
    y: np.ndarray,
    frac: float = 0.03,
    keep_all_below: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Small stratified subset for pipeline smoke tests.

    Keeps ALL rows of classes with fewer than `keep_all_below` samples and a
    `frac` fraction of the larger classes, so rare classes still appear.
    """
    rng = np.random.RandomState(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < keep_all_below:
            keep.append(idx)
        else:
            n = max(int(len(idx) * frac), keep_all_below)
            keep.append(rng.choice(idx, n, replace=False))
    sel = np.sort(np.concatenate(keep))
    return X[sel], y[sel]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build / inspect the clean CV dataset cache")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--force", action="store_true", help="Rebuild cache from raw CSVs")
    a = p.parse_args()
    X, y15, cols, gi = load_clean_dataset(a.config, force_rebuild=a.force)
    y13 = merge_labels_np(y15)
    print("\n  13-class (merged) real counts:")
    for k, name in enumerate(MERGED_NAMES):
        print(f"    {k:2d} {name:18s} {int((y13 == k).sum()):>10,}")
    print(f"\n  Counts written to {DEFAULT_COUNTS}")

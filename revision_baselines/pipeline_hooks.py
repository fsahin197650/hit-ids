#!/usr/bin/env python3
"""
pipeline_hooks.py  —  Single wiring point for revision baseline experiments.

All scripts (00–04) import this module.  If anything about the existing repo's
data paths, import paths, or conventions changes, fix it HERE and only here.

═══════════════════════════════════════════════════════════════════════════════
ASSUMPTIONS (verified by reading the repo — listed so the user can sanity-check)
═══════════════════════════════════════════════════════════════════════════════
A1. (FIXED 2026-09-29) Data comes from src/data/cv_data.load_clean_dataset():
    REAL, UNSCALED rows rebuilt from the raw CSVs. The old data/splits/train.pt
    was already SMOTE'd, so folding train+val+test leaked synthetic rows into
    every test fold. Scripts must call scale_fold(X_train, X_test) inside each
    fold (RobustScaler fit on the training fold only).
A2. Original 15-class labels are merged to 13 classes via MERGE_MAP below
    (Web Attack subtypes 12,13,14 → 12).
A3. 5-fold StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on the
    REAL rows — identical to the fixed evaluate_cv.py.
A4. Per-fold SMOTE uses  apply_smote_fold()  with  target_count=20_000  and
    random_state=42+fold_index.
A5. Majority undersampling caps BENIGN at 200,000 within each fold.
A6. BiLSTM baseline is  BiLSTMClassifier(num_features=72, num_classes=13,
    hidden_size=128, num_layers=2, dropout=0.3).
A7. PyTorch baselines train with  train_pytorch_baseline()  from
    src.models.baselines — AdamW, lr=1e-3, CosineAnnealingLR, 30 epochs.
A8. Config file: configs/config.yaml
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# ── Repo root on sys.path ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Existing repo imports ────────────────────────────────────────────────
# If these fail, the user must fix the paths above.
try:
    from src.models.baselines import (
        BiLSTMClassifier,
        create_baseline,
        train_pytorch_baseline,
    )
except ImportError as e:
    raise ImportError(
        f"Cannot import from src.models.baselines — is REPO_ROOT correct?\n"
        f"  REPO_ROOT = {REPO_ROOT}\n  Error: {e}"
    )

# ── Constants (must match evaluate_cv.py exactly) ────────────────────────

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
def _num_features() -> int:
    meta = REPO_ROOT / "data" / "processed" / "clean_full.json"
    if meta.exists():
        return int(json.loads(meta.read_text())["n_features"])
    return 72


NUM_FEATURES = _num_features()

SPLITS_DIR = str(REPO_ROOT / "data" / "splits")
CONFIG_PATH = str(REPO_ROOT / "configs" / "config.yaml")
RESULTS_DIR = str(REPO_ROOT / "revision_results")

CV_TAG = os.environ.get("HIT_CV_TAG", "clean")   # e.g. HIT_CV_TAG=clean_rv
CLEAN_CV_PATH = str(REPO_ROOT / "results" / "tables" / f"cv_evaluation_{CV_TAG}.json")


def get_hit_ids_per_fold_f1() -> List[float]:
    """Per-fold macro-F1 of HiT-IDS (Cascade) from the CLEAN CV run.

    The old hard-coded values (0.9891 mean) came from the contaminated CV and
    must not be used.
    """
    if not os.path.exists(CLEAN_CV_PATH):
        raise FileNotFoundError(
            f"{CLEAN_CV_PATH} not found — run scripts/evaluate_cv.py first "
            f"(clean protocol) before comparing baselines against HiT-IDS.")
    with open(CLEAN_CV_PATH) as f:
        d = json.load(f)
    return list(d["summary"]["HiT-IDS (Cascade)"]["per_fold_f1"])


# Kept for backward compatibility; None until the clean CV run exists.
try:
    HIT_IDS_PER_FOLD_F1: Optional[List[float]] = get_hit_ids_per_fold_f1()
except FileNotFoundError:
    HIT_IDS_PER_FOLD_F1 = None


# ════════════════════════════════════════════════════════════════════════
# Data Loading & Folding
# ════════════════════════════════════════════════════════════════════════


def merge_labels_np(y: np.ndarray) -> np.ndarray:
    """Merge 15-class labels to 13-class (Web Attack subtypes → single)."""
    return np.array([MERGE_MAP[int(v)] for v in y])


GROUP_INDICES: Optional[Dict[str, List[int]]] = None


def load_all_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the REAL, UNSCALED CIC-IDS2017 rows (no synthetic samples).

    Returns:
        X_all : np.ndarray, shape (N, 72), float32, UNSCALED
        y_all : np.ndarray, shape (N,), int64  (13-class)

    Call scale_fold(X_train, X_test) inside every CV fold.
    """
    global GROUP_INDICES
    from src.data.cv_data import load_clean_dataset
    X_all, y15, _, GROUP_INDICES = load_clean_dataset(CONFIG_PATH)
    return X_all.astype(np.float32), merge_labels_np(y15).astype(np.int64)


def scale_fold(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-group RobustScaler fitted on the training fold only."""
    from src.data.cv_data import scale_fold as _scale
    assert GROUP_INDICES is not None, "call load_all_data() first"
    Xtr, Xte, _ = _scale(X_train, X_test, GROUP_INDICES)
    return Xtr, Xte


def get_fold_splitter(n_splits: int = 5) -> StratifiedKFold:
    """Return the SAME fold splitter used by evaluate_cv.py."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# ════════════════════════════════════════════════════════════════════════
# SMOTE  (copied from evaluate_cv.py — apply_smote_fold)
# ════════════════════════════════════════════════════════════════════════


def apply_smote_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_count: int = 20_000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply adaptive SMOTE + majority undersample to a fold's training data.

    Adaptive strategy (from evaluate_cv.py):
      - Classes with <50 real samples  → target = max(real*3, 200)
      - Classes with 50–500 samples    → target = 5_000
      - Classes with 500–5000 samples  → target = target_count
      - Classes with >5000             → no oversampling
    After SMOTE: majority class capped at 200,000.
    """
    from imblearn.over_sampling import SMOTE

    unique, counts = np.unique(y_train, return_counts=True)
    cls_count = dict(zip(unique.astype(int), counts.astype(int)))

    sampling_strategy = {}
    for cls, cnt in cls_count.items():
        if cnt < 50:
            tgt = max(cnt * 3, 200)
        elif cnt < 500:
            tgt = 5_000
        elif cnt < 5_000:
            tgt = target_count
        else:
            continue
        if tgt > cnt:
            sampling_strategy[cls] = tgt

    if sampling_strategy:
        min_cnt = min(counts[counts > 1])
        k = min(5, int(min_cnt) - 1)
        k = max(1, k)
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k,
        )
        try:
            X_res, y_res = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"    [SMOTE warning] {e}, skipping SMOTE")
            X_res, y_res = X_train, y_train
    else:
        X_res, y_res = X_train, y_train

    # Majority undersample: cap largest class at 200K
    unique_r, counts_r = np.unique(y_res, return_counts=True)
    max_majority = 200_000
    maj_cls = unique_r[counts_r.argmax()]
    if counts_r.max() > max_majority:
        maj_idx = np.where(y_res == maj_cls)[0]
        keep = np.random.RandomState(random_state).choice(
            maj_idx, max_majority, replace=False
        )
        other = np.where(y_res != maj_cls)[0]
        indices = np.concatenate([keep, other])
        np.random.RandomState(random_state).shuffle(indices)
        X_res, y_res = X_res[indices], y_res[indices]

    return X_res.astype(np.float32), y_res.astype(np.int64)


# ════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ════════════════════════════════════════════════════════════════════════


def compute_fold_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Compute metrics for one fold (matching evaluate_cv.py output)."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def compute_full_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str] = None
) -> Dict[str, Any]:
    """Compute full metrics including per-class."""
    if label_names is None:
        label_names = MERGED_NAMES
    n_cls = len(label_names)
    base = compute_fold_metrics(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        labels=list(range(n_cls)),
        zero_division=0,
        output_dict=True,
    )
    base["per_class"] = {}
    for name in label_names:
        if name in report:
            base["per_class"][name] = {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1-score": report[name]["f1-score"],
                "support": report[name]["support"],
            }
    return base


def aggregate_folds(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean ± std across folds."""
    keys = ["macro_f1", "weighted_f1", "accuracy"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in fold_results]
        agg[f"mean_{k}"] = float(np.mean(vals))
        agg[f"std_{k}"] = float(np.std(vals))
    agg["per_fold"] = fold_results
    return agg


def set_seed(seed: int):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_results_dir():
    """Create revision_results/ if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(data: Any, filename: str):
    """Save dict to revision_results/<filename>."""
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved: {path}")
    return path

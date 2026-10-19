#!/usr/bin/env python3
"""
Shared helpers for E1-E8 supplementary experiment scripts.

Provides:
    - Model loading (CIC-IDS2017, UNSW-NB15)
    - Config loading
    - Common constants
    - Attention / gradient importance computation
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.hit_ids import HiTIDS
from src.xai.attention_rollout import AttentionRollout

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# CIC-IDS2017 config & paths
CIC_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/config.yaml")
CIC_BINARY_CKPT = os.path.join(PROJECT_ROOT, "results/checkpoints/hit_ids_binary.pt")
CIC_ATTACK_CKPT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results/checkpoints/hit_ids_attack_merged_v2.pt"),
    os.path.join(PROJECT_ROOT, "results/checkpoints/hit_ids_attack_merged.pt"),
    os.path.join(PROJECT_ROOT, "results/checkpoints/hit_ids_attack.pt"),
]
CIC_TEST_DATA = os.path.join(PROJECT_ROOT, "data/splits/test.pt")

# UNSW-NB15 config & paths
UNSW_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/config_unsw_binary.yaml")
UNSW_BINARY_CKPT = os.path.join(
    PROJECT_ROOT, "results/unsw_nb15/checkpoints/hit_ids_binary_unsw.pt"
)

# Aligned data paths for cross-dataset
ALIGNED_NB15_FOR_CIC_MODEL = os.path.join(
    PROJECT_ROOT, "data/unsw_nb15/aligned_for_2017_model.pt"
)
ALIGNED_CIC_FOR_NB15_MODEL = os.path.join(
    PROJECT_ROOT, "data/splits/aligned_for_nb15_model.pt"
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load a YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_group_info(config: dict) -> Tuple[List[str], List[int]]:
    """Extract group names and sizes from config."""
    fg = config["feature_groups"]
    names = list(fg.keys())
    sizes = [len(fg[n]) for n in names]
    return names, sizes


def load_model(
    config_path: str,
    checkpoint_path: str,
    num_classes: int,
    device: str = DEVICE,
) -> HiTIDS:
    """Load a trained HiT-IDS model from config and checkpoint."""
    config = load_config(config_path)
    group_names, group_sizes = get_group_info(config)
    mcfg = config["model"]

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Adjust group sizes from checkpoint (handles dynamic one-hot)
    for i, gn in enumerate(group_names):
        key = f"embeddings.{gn}.feature_weight"
        if key in state_dict:
            actual = state_dict[key].shape[0]
            if actual != group_sizes[i]:
                group_sizes[i] = actual

    model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=mcfg["d_emb"], nhead=mcfg["nhead"],
        intra_layers=mcfg["intra_group_layers"],
        inter_layers=mcfg["inter_group_layers"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=0.0, num_classes=num_classes,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_cic_binary_model(device: str = DEVICE) -> HiTIDS:
    """Load CIC-IDS2017 binary model."""
    return load_model(CIC_CONFIG_PATH, CIC_BINARY_CKPT, num_classes=2, device=device)


def load_cic_attack_model(device: str = DEVICE) -> Tuple[HiTIDS, int]:
    """Load CIC-IDS2017 attack model. Returns (model, num_classes)."""
    ckpt_path = None
    for c in CIC_ATTACK_CKPT_CANDIDATES:
        if os.path.exists(c):
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No attack checkpoint found in {CIC_ATTACK_CKPT_CANDIDATES}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt["model_state_dict"]["classification_head.head.3.bias"].shape[0]
    model = load_model(CIC_CONFIG_PATH, ckpt_path, num_classes=num_classes, device=device)
    return model, num_classes


def load_unsw_binary_model(device: str = DEVICE) -> HiTIDS:
    """Load UNSW-NB15 binary model (fold-0 or best fold checkpoint)."""
    return load_model(UNSW_CONFIG_PATH, UNSW_BINARY_CKPT, num_classes=2, device=device)


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_attention_importance(
    model: HiTIDS, X: torch.Tensor, group_names: List[str],
    batch_size: int = 256, device: str = DEVICE
) -> np.ndarray:
    """Compute attention rollout importance for all samples."""
    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")
    all_imp = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = X[start:end].to(device)
        with torch.no_grad():
            _, expl = model(xb, return_explanations=True)
        hi = rollout.hierarchical_importance(
            expl["raw_intra_attn"], expl["raw_inter_attn"], group_names
        )
        imp = hi["global_flat"].cpu().numpy()
        all_imp.append(imp)

    imp_all = np.vstack(all_imp)
    row_sums = imp_all.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return imp_all / row_sums


def compute_gradient_importance(
    model: HiTIDS, X: torch.Tensor, y: torch.Tensor,
    batch_size: int = 128, device: str = DEVICE
) -> np.ndarray:
    """Compute input gradient (|∂L/∂x|) importance for all samples."""
    all_imp = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = X[start:end].clone().to(device).requires_grad_(True)
        yb = y[start:end].to(device)

        logits = model(xb, return_explanations=False)
        scores = logits.gather(1, yb.unsqueeze(1)).sum()
        scores.backward()

        imp = xb.grad.abs().cpu().numpy()
        all_imp.append(imp)
        model.zero_grad()

    imp_all = np.vstack(all_imp)
    row_sums = imp_all.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    return imp_all / row_sums


def compute_random_importance(N: int, D: int, seed: int = SEED) -> np.ndarray:
    """Random baseline importance."""
    rng = np.random.RandomState(seed)
    imp = rng.rand(N, D)
    return imp / imp.sum(axis=1, keepdims=True)


def set_seed(seed: int = SEED):
    """Set reproducibility seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(*paths):
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_json(data: dict, path: str):
    """Save dict to JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  ✓ Saved: {path}")

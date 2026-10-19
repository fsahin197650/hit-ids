"""
Shared loader for the single "deployment" HiT-IDS cascade used by every XAI /
error-analysis experiment, so that all of them refer to ONE clearly defined
model:

    binary stage : results/checkpoints/hit_ids_binary.pt
    attack stage : results/checkpoints/hit_ids_attack_merged.pt   (12 classes)
    trained on   : data/splits/train.pt (70% split, SMOTE'd)
    threshold    : calibrated on data/splits/val.pt (real rows) by
                   scripts/deployment_eval.py → results/tables/deployment_eval.json
    evaluated on : data/splits/test.pt (real rows, 377,423)

Nothing here touches the CV protocol.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from ..models.hit_ids import HiTIDS
from ..data.cv_data import merge_labels_np, MERGED_NAMES, NUM_MERGED  # noqa: F401

BINARY_CKPT = "results/checkpoints/hit_ids_binary.pt"
ATTACK_CKPT = "results/checkpoints/hit_ids_attack_merged.pt"
EVAL_JSON = "results/tables/deployment_eval.json"


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build(config: dict, num_classes: int, device: str, dropout: float = 0.0) -> HiTIDS:
    fg = config["feature_groups"]
    names = list(fg.keys())
    mc = config["model"]
    return HiTIDS(group_sizes=[len(fg[g]) for g in names], group_names=names,
                  d_emb=mc["d_emb"], nhead=mc["nhead"], intra_layers=mc["intra_group_layers"],
                  inter_layers=mc["inter_group_layers"], dim_feedforward=mc["dim_feedforward"],
                  dropout=dropout, num_classes=num_classes).to(device).eval()


def load_deployment_models(config: dict, device: str,
                           binary_ckpt: str = BINARY_CKPT,
                           attack_ckpt: str = ATTACK_CKPT) -> Tuple[HiTIDS, HiTIDS, List[str]]:
    b = torch.load(binary_ckpt, map_location=device, weights_only=False)
    a = torch.load(attack_ckpt, map_location=device, weights_only=False)
    bs, as_ = b["model_state_dict"], a["model_state_dict"]
    nc = as_["classification_head.head.3.bias"].shape[0]
    assert nc == NUM_MERGED - 1, f"attack checkpoint has {nc} classes, expected {NUM_MERGED - 1}"
    bin_m, atk_m = _build(config, 2, device), _build(config, nc, device)
    bin_m.load_state_dict(bs)
    atk_m.load_state_dict(as_)
    return bin_m, atk_m, list(config["feature_groups"].keys())


def load_split(name: str, splits_dir: str = "data/splits") -> Tuple[torch.Tensor, np.ndarray]:
    d = torch.load(os.path.join(splits_dir, f"{name}.pt"), weights_only=False)
    return d["X"].float(), merge_labels_np(d["y"].numpy())


@torch.no_grad()
def cascade_probs(bin_m: HiTIDS, atk_m: HiTIDS, X: torch.Tensor, device: str,
                  bs: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    pb, pa = [], []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        pb.append(torch.softmax(bin_m(xb, return_explanations=False), 1).cpu().numpy())
        pa.append(torch.softmax(atk_m(xb, return_explanations=False), 1).cpu().numpy())
    return np.concatenate(pb), np.concatenate(pa)


def cascade_predict(p_bin: np.ndarray, p_atk: np.ndarray, threshold: float) -> np.ndarray:
    pred = np.zeros(len(p_bin), dtype=np.int64)
    m = p_bin[:, 1] >= threshold
    pred[m] = p_atk[m].argmax(1) + 1
    return pred


def combined_proba(p_bin: np.ndarray, p_atk: np.ndarray) -> np.ndarray:
    return np.concatenate([p_bin[:, :1], p_bin[:, 1:2] * p_atk], axis=1)


def read_threshold(path: str = EVAL_JSON) -> float:
    """Calibrated Stage-1 threshold; 0.5 (argmax) if calibration has not run."""
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f)["stage1_threshold"])
    print(f"  [deployment] {path} not found — using argmax (threshold 0.5)")
    return 0.5

#!/usr/bin/env python3
"""
05_faithfulness_h2h.py — Head-to-head faithfulness comparison.

    HiT-IDS attention rollout  vs  Deep-IDS GradientSHAP

Each explanation is evaluated on its OWN model (the methodologically correct
way: faithfulness asks whether THIS model's explanation reflects what THIS
model relies on).

Track 1  — Feature-level:   identical 11-metric battery from Section 5.12
Track 2  — Group-level:     4-group comprehensiveness/sufficiency
                            (the structured-explanation claim)

Reuses:
    - src.xai.faithfulness.ComprehensiveFaithfulness  (the exact battery)
    - src.xai.attention_rollout.AttentionRollout      (rollout explainer)
    - src.models.hit_ids.HiTIDS                       (HiT-IDS model)
    - revision_baselines.01_deepids.DeepIDSModel      (Deep-IDS model)
    - captum.attr.GradientShap                        (SHAP explainer)

Writes:
    revision_results/faithfulness_h2h.json
    revision_results/faithfulness_h2h.md

Usage:
    python revision_baselines/05_faithfulness_h2h.py
    python revision_baselines/05_faithfulness_h2h.py --n-samples 200   # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import stats as sp_stats

# ── Repo root on sys.path ────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# revision_baselines/ on sys.path (for 01_deepids import)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — Everything the user might need to verify or tweak
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Eval-set selection
    "n_samples": 500,           # Same as Section 5.12 (496 after correct-only filter)
    "seed": 42,                 # Same as Section 5.12

    # Faithfulness battery parameters (must match Section 5.12 exactly)
    "masking_value": 0.0,       # Zero-masking, same as ComprehensiveFaithfulness default
    "k_values": [5, 10, 15, 20],  # Same as Section 5.12
    "steps": 20,                # Deletion/insertion curve steps (same as 5.12)

    # Statistical test
    "alpha": 0.05,              # Significance level

    # Deep-IDS architecture (must match 01_deepids.py training config)
    "deepids_lstm_units": 64,
    "deepids_lstm_layers": 1,
    "deepids_dropout": 0.0,     # 0 for explanation (no stochasticity)
    "deepids_num_classes": 13,  # 13-class merged

    # GradientSHAP config
    "shap_n_background": 50,    # Background samples for GradientSHAP
    "shap_n_stdevs": 0.0,      # Standard GradientSHAP (no noise tunnel)

    # Paths
    "config_path": os.path.join(REPO_ROOT, "configs", "config.yaml"),
    "results_dir": os.path.join(REPO_ROOT, "revision_results"),
}

# ═══════════════════════════════════════════════════════════════════════════
# ASSUMPTIONS (documented for user verification)
# ═══════════════════════════════════════════════════════════════════════════
#
# A1. HiT-IDS attack model checkpoint exists at one of the paths checked
#     by load_attack_model() (same logic as run_faithfulness.py).
# A2. Deep-IDS was trained by 01_deepids.py with 5-fold CV. For this script,
#     we RE-TRAIN a fresh Deep-IDS on the full training data (train+val) or
#     we use a "representative fold" model. Since we need a SINGLE model to
#     evaluate faithfulness on a fixed eval set (not 5 separate fold models),
#     we train one from scratch on train+val.
# A3. masking_value = 0.0 matches the existing battery (line 257 of
#     run_faithfulness.py: ComprehensiveFaithfulness(model, masking_value=0.0)).
# A4. Eval samples: attack-only from test.pt, randomly subsampled (N=500,
#     seed=42), filtered to correctly classified only — same as 5.12.
# A5. The 11 metrics and their "better" directions:
#       deletion_auc       → lower  = more faithful
#       insertion_auc      → higher = more faithful
#       aopc               → higher = more faithful
#       comprehensiveness_k5/10/15/20  → higher = more faithful
#       sufficiency_k5/10/15/20        → lower  = more faithful
# A6. Feature groups (temporal/volume/protocol/behavioral) and their sizes
#     come from configs/config.yaml, matching HiT-IDS architecture.
# A7. For group-level Track 2: HiT-IDS uses native inter-group attention.
#     Deep-IDS groups are formed by summing |GradientSHAP_attr| per group.
#     Group comprehensiveness/sufficiency mask ALL features in a group.


# ═══════════════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS (reused from Section 5.12 — not redefined)
# ═══════════════════════════════════════════════════════════════════════════

METRIC_DEFS = {
    "deletion_auc":          {"direction": "lower",  "better_label": "↓"},
    "insertion_auc":         {"direction": "higher", "better_label": "↑"},
    "aopc":                  {"direction": "higher", "better_label": "↑"},
    "comprehensiveness_k5":  {"direction": "higher", "better_label": "↑"},
    "comprehensiveness_k10": {"direction": "higher", "better_label": "↑"},
    "comprehensiveness_k15": {"direction": "higher", "better_label": "↑"},
    "comprehensiveness_k20": {"direction": "higher", "better_label": "↑"},
    "sufficiency_k5":        {"direction": "lower",  "better_label": "↓"},
    "sufficiency_k10":       {"direction": "lower",  "better_label": "↓"},
    "sufficiency_k15":       {"direction": "lower",  "better_label": "↓"},
    "sufficiency_k20":       {"direction": "lower",  "better_label": "↓"},
}

ALL_METRIC_NAMES = list(METRIC_DEFS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_hitids_model(device: str) -> Tuple[nn.Module, List[str], List[int]]:
    """Load HiT-IDS attack model (same as run_faithfulness.py)."""
    from src.models.hit_ids import HiTIDS

    with open(CONFIG["config_path"]) as f:
        config = yaml.safe_load(f)

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[n]) for n in group_names]
    model_cfg = config["model"]

    ckpt_candidates = [
        os.path.join(REPO_ROOT, "results/checkpoints/hit_ids_attack_merged_v2.pt"),
        os.path.join(REPO_ROOT, "results/checkpoints/hit_ids_attack_merged.pt"),
        os.path.join(REPO_ROOT, "results/checkpoints/hit_ids_attack_v2.pt"),
        os.path.join(REPO_ROOT, "results/checkpoints/hit_ids_attack.pt"),
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No attack checkpoint found in {ckpt_candidates}")

    print(f"  Loading HiT-IDS: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt["model_state_dict"]["classification_head.head.3.bias"].shape[0]

    model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=0.0, num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  HiT-IDS: {num_classes} classes, {n_params:,} params")
    return model, group_names, group_sizes


def train_and_load_deepids(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    device: str,
) -> nn.Module:
    """
    Train a fresh Deep-IDS on train data and return the model.

    We need a single trained model for faithfulness evaluation.
    Training from scratch ensures the model is properly fitted.
    """
    from importlib import import_module
    deepids_mod = import_module("01_deepids")
    DeepIDSModel = deepids_mod.DeepIDSModel
    train_deepids_fn = deepids_mod.train_deepids

    from pipeline_hooks import set_seed, apply_smote_fold

    set_seed(CONFIG["seed"])

    # Apply SMOTE to training data
    X_tr_s, y_tr_s = apply_smote_fold(
        X_train, y_train,
        target_count=20_000,
        random_state=CONFIG["seed"],
    )
    print(f"  Deep-IDS SMOTE: {len(X_train):,} → {len(X_tr_s):,}")

    model = DeepIDSModel(
        num_features=X_train.shape[1],
        num_classes=CONFIG["deepids_num_classes"],
        lstm_units=CONFIG["deepids_lstm_units"],
        lstm_layers=CONFIG["deepids_lstm_layers"],
        dropout=0.3,  # Training dropout (will be set to eval mode after)
    )

    from torch.utils.data import DataLoader, TensorDataset
    train_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_s, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val[:20_000], dtype=torch.float32),
        torch.tensor(y_val[:20_000], dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    print("  Training Deep-IDS (30 epochs)...")
    train_deepids_fn(model, train_loader, val_loader, num_epochs=30,
                     lr=1e-3, weight_decay=1e-4, device=device)

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Deep-IDS: {CONFIG['deepids_num_classes']} classes, {n_params:,} params")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load CIC-IDS2017 test data (attack-only), same as run_faithfulness.py."""
    train_data = torch.load(
        os.path.join(REPO_ROOT, "data/splits/train.pt"), weights_only=False)
    val_data = torch.load(
        os.path.join(REPO_ROOT, "data/splits/val.pt"), weights_only=False)
    test_data = torch.load(
        os.path.join(REPO_ROOT, "data/splits/test.pt"), weights_only=False)

    X_test = test_data["X"]
    y_test = test_data["y"]

    # Attack-only for eval
    atk_mask = y_test > 0
    X_atk = X_test[atk_mask]
    y_atk = y_test[atk_mask] - 1  # 0-indexed for attack model

    # Train+val for Deep-IDS training (merge labels first)
    from pipeline_hooks import merge_labels_np
    X_train_all = torch.cat([train_data["X"], val_data["X"]]).numpy().astype(np.float32)
    y_train_all = torch.cat([train_data["y"], val_data["y"]]).numpy().astype(np.int64)
    y_train_all = merge_labels_np(y_train_all)

    return X_atk, y_atk, X_train_all, y_train_all


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_rollout_importance(
    model: nn.Module, X: torch.Tensor, group_names: List[str], device: str,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HiT-IDS attention rollout importance.
    Returns:
        flat_imp: (N, 72) feature-level importance (normalized)
        group_imp: (N, 4) group-level importance (from inter-group attention)
    """
    from src.xai.attention_rollout import AttentionRollout

    rollout = AttentionRollout(residual_weight=0.5, head_fusion="mean")
    all_flat = []
    all_group = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        xb = X[start:end].to(device)
        with torch.no_grad():
            _, expl = model(xb, return_explanations=True)
        hi = rollout.hierarchical_importance(
            expl["raw_intra_attn"], expl["raw_inter_attn"], group_names)
        all_flat.append(hi["global_flat"].cpu().numpy())
        all_group.append(hi["group"].cpu().numpy())

    flat_imp = np.vstack(all_flat)
    group_imp = np.vstack(all_group)

    # Normalize feature-level per sample
    row_sums = flat_imp.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    flat_imp = flat_imp / row_sums

    return flat_imp, group_imp


def compute_shap_importance(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor,
    X_background: torch.Tensor,
    group_sizes: List[int],
    device: str,
    n_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GradientSHAP importance for Deep-IDS.
    Returns:
        flat_imp: (N, 72) feature-level |attribution| (normalized)
        group_imp: (N, 4) group-level aggregated importance
    """
    from captum.attr import GradientShap

    # IMPORTANT: model must be in train mode for cuDNN LSTM backward
    model.train()
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    explainer = GradientShap(model)

    X_bg = X_background[:CONFIG["shap_n_background"]].to(device)
    all_attr = []

    for i in range(len(X)):
        x_single = X[i:i+1].to(device)
        with torch.no_grad():
            target = model(x_single).argmax(1)

        attr = explainer.attribute(x_single, baselines=X_bg, target=target)
        all_attr.append(attr.detach().cpu().numpy())

        if (i + 1) % 100 == 0:
            print(f"    GradientSHAP: {i+1}/{len(X)}")

    torch.backends.cudnn.enabled = prev_cudnn
    model.eval()

    attr_all = np.vstack(all_attr)
    # Take absolute value (faithfulness uses |attribution| as importance)
    flat_imp = np.abs(attr_all)

    # Normalize per sample
    row_sums = flat_imp.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    flat_imp = flat_imp / row_sums

    # Group-level: sum |attr| per group
    group_imp = np.zeros((len(X), len(group_sizes)))
    offset = 0
    for g, gs in enumerate(group_sizes):
        group_imp[:, g] = flat_imp[:, offset:offset + gs].sum(axis=1)
        offset += gs

    return flat_imp, group_imp


# ═══════════════════════════════════════════════════════════════════════════
# FAITHFULNESS EVALUATORS (reuse ComprehensiveFaithfulness)
# ═══════════════════════════════════════════════════════════════════════════

class ModelWrapper(nn.Module):
    """Wrap Deep-IDS to match the interface ComprehensiveFaithfulness expects.

    ComprehensiveFaithfulness calls: model(X, return_explanations=False)
    Deep-IDS only takes: model(X)
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, return_explanations=False):
        return self.model(x)

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self


def run_feature_level_battery(
    model: nn.Module, X: torch.Tensor, importances: np.ndarray,
    targets: torch.Tensor, device: str,
) -> Dict[str, Any]:
    """Run the identical 11-metric battery from Section 5.12."""
    from src.xai.faithfulness import ComprehensiveFaithfulness

    evaluator = ComprehensiveFaithfulness(
        model, device=device, masking_value=CONFIG["masking_value"])
    result = evaluator.evaluate_all(
        X, importances, targets,
        k_values=CONFIG["k_values"], steps=CONFIG["steps"])
    return result


def run_group_level_battery(
    model: nn.Module, X: torch.Tensor,
    group_importances: np.ndarray,  # (N, 4)
    group_sizes: List[int],
    targets: torch.Tensor,
    device: str,
) -> Dict[str, Any]:
    """
    Group-level comprehensiveness & sufficiency.

    For each group (ranked by group importance): mask ALL features in
    that group → measure prediction drop.

    This tests whether the group-level explanation is meaningful.
    """
    from src.xai.faithfulness import ComprehensiveFaithfulness

    N = len(X)
    n_groups = len(group_sizes)

    evaluator = ComprehensiveFaithfulness(
        model, device=device, masking_value=CONFIG["masking_value"])

    # Compute base probabilities
    targets_t = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets, dtype=torch.long)
    base_probs = evaluator._get_proba(X, targets_t)

    # Sort groups by importance (descending) per sample
    group_order = np.argsort(-group_importances, axis=1)  # (N, n_groups)

    # Group boundaries (cumulative offsets)
    group_offsets = np.cumsum([0] + group_sizes)

    # Comprehensiveness: mask top-g groups, g=1..n_groups
    comp_results = {}
    suff_results = {}

    for g in range(1, n_groups + 1):
        # Comprehensiveness: mask top-g groups
        X_comp = X.clone()
        for i in range(N):
            for gi in range(g):
                grp_idx = group_order[i, gi]
                start = group_offsets[grp_idx]
                end = group_offsets[grp_idx + 1]
                X_comp[i, start:end] = CONFIG["masking_value"]

        comp_probs = evaluator._get_proba(X_comp, targets_t)
        comp_vals = base_probs - comp_probs  # higher = more faithful

        # Sufficiency: keep ONLY top-g groups
        X_suff = torch.full_like(X, CONFIG["masking_value"])
        for i in range(N):
            for gi in range(g):
                grp_idx = group_order[i, gi]
                start = group_offsets[grp_idx]
                end = group_offsets[grp_idx + 1]
                X_suff[i, start:end] = X[i, start:end]

        suff_probs = evaluator._get_proba(X_suff, targets_t)
        suff_vals = base_probs - suff_probs  # lower = more faithful

        comp_results[f"group_comp_top{g}"] = {
            "per_sample": comp_vals.tolist(),
            "mean": float(comp_vals.mean()),
            "std": float(comp_vals.std()),
        }
        suff_results[f"group_suff_top{g}"] = {
            "per_sample": suff_vals.tolist(),
            "mean": float(suff_vals.mean()),
            "std": float(suff_vals.std()),
        }

    # Group AOPC: mean comprehensiveness across g=1..4
    aopc_vals = np.mean(
        [np.array(comp_results[f"group_comp_top{g}"]["per_sample"])
         for g in range(1, n_groups + 1)],
        axis=0
    )
    group_aopc = {
        "per_sample": aopc_vals.tolist(),
        "mean": float(aopc_vals.mean()),
        "std": float(aopc_vals.std()),
    }

    return {**comp_results, **suff_results, "group_aopc": group_aopc}


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def compare_metrics(
    rollout_results: Dict, shap_results: Dict,
    metric_name: str, direction: str,
) -> Dict[str, Any]:
    """
    Paired comparison of a single metric between rollout and SHAP.

    Uses Wilcoxon signed-rank test (primary) and paired t-test (secondary).
    Returns who wins (rollout, shap, or tie at alpha=0.05).
    """
    r_vals = np.array(rollout_results[metric_name]["per_sample"])
    s_vals = np.array(shap_results[metric_name]["per_sample"])

    r_mean = float(r_vals.mean())
    r_std = float(r_vals.std())
    s_mean = float(s_vals.mean())
    s_std = float(s_vals.std())

    delta = r_vals - s_vals  # positive = rollout has higher value

    # Determine which direction "wins" for rollout
    # "higher" metrics: rollout wins if delta > 0
    # "lower"  metrics: rollout wins if delta < 0
    if direction == "higher":
        # rollout wins if its value is higher
        rollout_advantage = delta.mean() > 0
    else:
        # rollout wins if its value is lower
        rollout_advantage = delta.mean() < 0

    # Wilcoxon signed-rank test (two-sided)
    try:
        w_stat, w_p = sp_stats.wilcoxon(r_vals, s_vals, alternative="two-sided")
    except Exception:
        w_stat, w_p = float("nan"), 1.0

    # Paired t-test (two-sided)
    t_stat, t_p = sp_stats.ttest_rel(r_vals, s_vals)

    # Cohen's d (paired)
    d_diff = delta
    d_std = d_diff.std(ddof=1)
    cohens_d = float(d_diff.mean() / d_std) if d_std > 0 else 0.0

    # Winner determination
    alpha = CONFIG["alpha"]
    if w_p < alpha:
        winner = "rollout" if rollout_advantage else "shap"
    else:
        winner = "tie"

    return {
        "metric": metric_name,
        "direction": direction,
        "rollout_mean": r_mean,
        "rollout_std": r_std,
        "shap_mean": s_mean,
        "shap_std": s_std,
        "delta_mean": float(delta.mean()),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "cohens_d": cohens_d,
        "winner": winner,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MARKDOWN OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def generate_markdown(
    feature_comparisons: List[Dict],
    group_comparisons: List[Dict],
    n_samples: int,
    runtime_s: float,
) -> str:
    """Generate ready-to-paste markdown tables."""
    lines = [
        "# Faithfulness Head-to-Head: HiT-IDS Rollout vs Deep-IDS GradientSHAP\n",
        f"**Eval samples:** {n_samples} attack samples (correctly classified by each model)",
        f"**Masking:** zero-masking (value = {CONFIG['masking_value']})",
        f"**GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"**Runtime:** {runtime_s:.0f}s\n",
        "---\n",
    ]

    # ── Track 1: Feature-level ───────────────────────────────────────────
    lines.append("## Track 1 — Feature-Level Faithfulness (11 Metrics)\n")
    lines.append("Each explanation evaluated on **its own model** (rollout → HiT-IDS, SHAP → Deep-IDS).\n")
    lines.append("| Metric | Better | HiT-IDS Rollout | Deep-IDS SHAP | Δ | p (Wilcoxon) | Winner |")
    lines.append("|--------|:------:|:---------------:|:-------------:|:---:|:------------:|:------:|")

    wins_r, wins_s, ties = 0, 0, 0
    for c in feature_comparisons:
        d = METRIC_DEFS[c["metric"]]
        winner_icon = {"rollout": "🟢 Rollout", "shap": "🔴 SHAP", "tie": "⚪ Tie"}[c["winner"]]
        p_str = f"{c['wilcoxon_p']:.2e}" if c["wilcoxon_p"] < 0.001 else f"{c['wilcoxon_p']:.4f}"
        lines.append(
            f"| {c['metric']} | {d['better_label']} | "
            f"{c['rollout_mean']:.4f}±{c['rollout_std']:.4f} | "
            f"{c['shap_mean']:.4f}±{c['shap_std']:.4f} | "
            f"{c['delta_mean']:+.4f} | {p_str} | {winner_icon} |"
        )
        if c["winner"] == "rollout":
            wins_r += 1
        elif c["winner"] == "shap":
            wins_s += 1
        else:
            ties += 1

    lines.append(f"\n**Feature-level tally: Rollout wins {wins_r}/11, SHAP wins {wins_s}/11, Ties {ties}/11 (α={CONFIG['alpha']})**\n")

    # ── Track 2: Group-level ─────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Track 2 — Group-Level Faithfulness (Structured Explanation)\n")
    lines.append("HiT-IDS: native inter-group attention. Deep-IDS: GradientSHAP aggregated to 4 groups.\n")
    lines.append("| Metric | Better | HiT-IDS Rollout | Deep-IDS SHAP | Δ | p (Wilcoxon) | Winner |")
    lines.append("|--------|:------:|:---------------:|:-------------:|:---:|:------------:|:------:|")

    g_wins_r, g_wins_s, g_ties = 0, 0, 0
    for c in group_comparisons:
        if "comp" in c["metric"]:
            better = "↑"
        elif "suff" in c["metric"]:
            better = "↓"
        else:
            better = "↑"  # aopc
        winner_icon = {"rollout": "🟢 Rollout", "shap": "🔴 SHAP", "tie": "⚪ Tie"}[c["winner"]]
        p_str = f"{c['wilcoxon_p']:.2e}" if c["wilcoxon_p"] < 0.001 else f"{c['wilcoxon_p']:.4f}"
        lines.append(
            f"| {c['metric']} | {better} | "
            f"{c['rollout_mean']:.4f}±{c['rollout_std']:.4f} | "
            f"{c['shap_mean']:.4f}±{c['shap_std']:.4f} | "
            f"{c['delta_mean']:+.4f} | {p_str} | {winner_icon} |"
        )
        if c["winner"] == "rollout":
            g_wins_r += 1
        elif c["winner"] == "shap":
            g_wins_s += 1
        else:
            g_ties += 1

    n_group_metrics = len(group_comparisons)
    lines.append(f"\n**Group-level tally: Rollout wins {g_wins_r}/{n_group_metrics}, SHAP wins {g_wins_s}/{n_group_metrics}, Ties {g_ties}/{n_group_metrics} (α={CONFIG['alpha']})**\n")

    # ── Interpretation ───────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Interpretation\n")

    if wins_r >= 7 and g_wins_r >= n_group_metrics // 2 + 1:
        case = "STRONG"
        lines.append("**Case: STRONG** — Rollout wins feature-level majority AND group-level majority.")
        lines.append("→ The paper's thesis holds: intrinsic explanation is more faithful AND uniquely structured.")
    elif g_wins_r >= n_group_metrics // 2 + 1:
        case = "PARTIAL"
        lines.append("**Case: PARTIAL** — Rollout wins group-level but not feature-level majority.")
        lines.append("→ Narrow thesis to structured, group-level faithfulness. Feature-level is competitive but not dominant.")
    else:
        case = "WEAK"
        lines.append("**Case: WEAK** — Rollout does not win either level convincingly.")
        lines.append("→ Intrinsic-explainability advantage is not empirically supported vs a real competitor.")

    lines.append(f"\n**Summary: Rollout wins {wins_r}/11 feature-level, {g_wins_r}/{n_group_metrics} group-level (α={CONFIG['alpha']})**")

    return "\n".join(lines) + "\n", case


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Faithfulness H2H: HiT-IDS rollout vs Deep-IDS GradientSHAP")
    parser.add_argument("--n-samples", type=int, default=CONFIG["n_samples"])
    parser.add_argument("--steps", type=int, default=CONFIG["steps"])
    args = parser.parse_args()

    CONFIG["n_samples"] = args.n_samples
    CONFIG["steps"] = args.steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    t_start = time.time()

    print("=" * 70)
    print("  FAITHFULNESS HEAD-TO-HEAD")
    print("  HiT-IDS Attention Rollout  vs  Deep-IDS GradientSHAP")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Samples: {CONFIG['n_samples']}  |  Steps: {CONFIG['steps']}")

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading data...")
    X_atk, y_atk, X_train_all, y_train_all = load_data()
    print(f"  Attack test samples: {len(X_atk):,}")

    # ── Subsample eval set (same as Section 5.12) ────────────────────────
    N = min(CONFIG["n_samples"], len(X_atk))
    idx = np.random.choice(len(X_atk), N, replace=False)
    X_eval = X_atk[idx]
    y_eval = y_atk[idx]

    # ── Load HiT-IDS ─────────────────────────────────────────────────────
    print("\n  Loading HiT-IDS...")
    hitids_model, group_names, group_sizes = load_hitids_model(device)

    # Filter to correctly classified by HiT-IDS
    with torch.no_grad():
        logits = hitids_model(X_eval.to(device), return_explanations=False)
        hitids_preds = logits.argmax(1).cpu()
    hitids_correct = (hitids_preds == y_eval)
    print(f"  HiT-IDS accuracy on eval: {hitids_correct.float().mean():.4f}")

    # ── Train Deep-IDS ───────────────────────────────────────────────────
    print("\n  Training Deep-IDS...")
    deepids_model = train_and_load_deepids(
        X_train_all, y_train_all,
        X_eval.numpy(), y_eval.numpy(),  # used only for validation
        device,
    )

    # Deep-IDS predicts 13-class (merged). y_eval is attack-only (0-indexed).
    # We need to map: y_eval (attack 0-indexed = class 1..12 minus 1) to
    # Deep-IDS's 13-class labels. Attack classes in merged = 1..12, so:
    deepids_y = y_eval + 1  # map back to merged 13-class

    # Filter to correctly classified by Deep-IDS
    deepids_model.eval()
    with torch.no_grad():
        logits = deepids_model(X_eval.to(device))
        deepids_preds = logits.argmax(1).cpu()
    deepids_correct = (deepids_preds == deepids_y)
    print(f"  Deep-IDS accuracy on eval: {deepids_correct.float().mean():.4f}")

    # Use samples correctly classified by BOTH models (conservative)
    both_correct = hitids_correct & deepids_correct
    X_shared = X_eval[both_correct]
    y_hitids = y_eval[both_correct]         # 0-indexed attack labels
    y_deepids = deepids_y[both_correct]     # 13-class merged labels
    n_eval = len(X_shared)
    print(f"\n  Shared correct samples: {n_eval} "
          f"(both models got these right)")

    if n_eval < 50:
        print("  ❌ Too few shared correct samples. Check models.")
        sys.exit(1)

    # ── Compute importances ──────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  COMPUTING IMPORTANCES")
    print(f"{'═' * 70}")

    print("\n  HiT-IDS attention rollout...")
    t0 = time.time()
    rollout_flat, rollout_group = compute_rollout_importance(
        hitids_model, X_shared, group_names, device)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("\n  Deep-IDS GradientSHAP...")
    t0 = time.time()
    # Background = BENIGN samples from test set
    benign_mask = torch.load(
        os.path.join(REPO_ROOT, "data/splits/test.pt"),
        weights_only=False)["y"] == 0
    X_test_all = torch.load(
        os.path.join(REPO_ROOT, "data/splits/test.pt"),
        weights_only=False)["X"]
    X_bg = X_test_all[benign_mask][:CONFIG["shap_n_background"]]

    shap_flat, shap_group = compute_shap_importance(
        deepids_model, X_shared, y_deepids, X_bg, group_sizes, device)
    print(f"    Done ({time.time()-t0:.1f}s)")

    # ── Sanity check: rollout vs random (must pass ~11/11 like 5.12) ─────
    print(f"\n{'═' * 70}")
    print("  SANITY CHECK: Rollout vs Random (should match Section 5.12)")
    print(f"{'═' * 70}")

    random_imp = np.random.rand(n_eval, X_eval.shape[1])
    random_imp = random_imp / random_imp.sum(axis=1, keepdims=True)

    rollout_results = run_feature_level_battery(
        hitids_model, X_shared, rollout_flat, y_hitids, device)
    random_results = run_feature_level_battery(
        hitids_model, X_shared, random_imp, y_hitids, device)

    sanity_wins = 0
    for m_name, m_def in METRIC_DEFS.items():
        r_vals = np.array(rollout_results[m_name]["per_sample"])
        rand_vals = np.array(random_results[m_name]["per_sample"])
        _, p = sp_stats.wilcoxon(r_vals, rand_vals, alternative="two-sided")
        if m_def["direction"] == "higher":
            won = r_vals.mean() > rand_vals.mean() and p < 0.05
        else:
            won = r_vals.mean() < rand_vals.mean() and p < 0.05
        if won:
            sanity_wins += 1
        status = "✓" if won else "✗"
        print(f"  {m_name:30s}: rollout={r_vals.mean():.4f} random={rand_vals.mean():.4f} {status}")

    print(f"\n  Sanity: rollout beats random on {sanity_wins}/11 metrics")
    if sanity_wins < 8:
        print("  ⚠️ WARNING: Rollout doesn't clearly beat random. Battery may not be wired correctly.")

    # ── Track 1: Feature-level H2H ───────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  TRACK 1: FEATURE-LEVEL FAITHFULNESS (11 metrics)")
    print(f"{'═' * 70}")

    # rollout_results already computed above
    # Now evaluate SHAP on Deep-IDS
    deepids_wrapped = ModelWrapper(deepids_model)
    shap_results = run_feature_level_battery(
        deepids_wrapped, X_shared, shap_flat, y_deepids, device)

    feature_comparisons = []
    for m_name, m_def in METRIC_DEFS.items():
        comp = compare_metrics(rollout_results, shap_results, m_name, m_def["direction"])
        feature_comparisons.append(comp)
        winner_str = {"rollout": "ROLLOUT", "shap": "SHAP", "tie": "TIE"}[comp["winner"]]
        print(f"  {m_name:30s}: rollout={comp['rollout_mean']:.4f} "
              f"shap={comp['shap_mean']:.4f} Δ={comp['delta_mean']:+.4f} "
              f"p={comp['wilcoxon_p']:.4f} → {winner_str}")

    # ── Track 2: Group-level H2H ─────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  TRACK 2: GROUP-LEVEL FAITHFULNESS")
    print(f"{'═' * 70}")

    rollout_group_results = run_group_level_battery(
        hitids_model, X_shared, rollout_group, group_sizes, y_hitids, device)
    shap_group_results = run_group_level_battery(
        deepids_wrapped, X_shared, shap_group, group_sizes, y_deepids, device)

    # Group metric definitions
    group_metric_defs = {}
    for g in range(1, len(group_sizes) + 1):
        group_metric_defs[f"group_comp_top{g}"] = "higher"
        group_metric_defs[f"group_suff_top{g}"] = "lower"
    group_metric_defs["group_aopc"] = "higher"

    group_comparisons = []
    for m_name, direction in group_metric_defs.items():
        comp = compare_metrics(rollout_group_results, shap_group_results, m_name, direction)
        group_comparisons.append(comp)
        winner_str = {"rollout": "ROLLOUT", "shap": "SHAP", "tie": "TIE"}[comp["winner"]]
        print(f"  {m_name:25s}: rollout={comp['rollout_mean']:.4f} "
              f"shap={comp['shap_mean']:.4f} → {winner_str}")

    # ── Generate outputs ─────────────────────────────────────────────────
    runtime = time.time() - t_start

    # JSON
    output = {
        "n_samples": n_eval,
        "n_features": int(X_eval.shape[1]),
        "n_groups": len(group_sizes),
        "group_names": group_names,
        "group_sizes": group_sizes,
        "config": {k: v for k, v in CONFIG.items()
                   if not k.startswith("config") and not k.startswith("results")},
        "sanity_check_rollout_vs_random": f"{sanity_wins}/11",
        "track1_feature_level": feature_comparisons,
        "track2_group_level": group_comparisons,
        "runtime_seconds": runtime,
    }

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    json_path = os.path.join(CONFIG["results_dir"], "faithfulness_h2h.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ JSON saved: {json_path}")

    # Markdown
    md_content, case = generate_markdown(
        feature_comparisons, group_comparisons, n_eval, runtime)
    md_path = os.path.join(CONFIG["results_dir"], "faithfulness_h2h.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"  ✓ Markdown saved: {md_path}")

    # ── Final summary ────────────────────────────────────────────────────
    f_wins = sum(1 for c in feature_comparisons if c["winner"] == "rollout")
    f_losses = sum(1 for c in feature_comparisons if c["winner"] == "shap")
    g_wins = sum(1 for c in group_comparisons if c["winner"] == "rollout")
    g_losses = sum(1 for c in group_comparisons if c["winner"] == "shap")

    print(f"\n{'═' * 70}")
    print(f"  RESULT: {case} CASE")
    print(f"{'═' * 70}")
    print(f"  Feature-level: Rollout {f_wins}W / {f_losses}L / {11-f_wins-f_losses}T  (of 11)")
    print(f"  Group-level:   Rollout {g_wins}W / {g_losses}L / {len(group_comparisons)-g_wins-g_losses}T  (of {len(group_comparisons)})")
    print(f"  Runtime:       {runtime:.0f}s")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

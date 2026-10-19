#!/usr/bin/env python3
"""
Ablation Study for HiT-IDS Manuscript Revision.

Tests 6 ablation variants using 5-fold stratified CV:
    A1: No hierarchical grouping (flat 72-token sequence)
    A2: No cascade (direct 13-class classification)
    A3: No Phase 2 attention regularization
    A4: ReLU instead of GELU
    A5: No SMOTE (class weights only)
    A6: Single attention head (h=1)

Reuses the same data pipeline and evaluation protocol from evaluate_cv.py.

Usage:
    python scripts/run_ablation.py --config configs/config.yaml
    python scripts/run_ablation.py --config configs/config.yaml --ablations A1 A4
    python scripts/run_ablation.py --config configs/config.yaml --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.models.hit_ids import HiTIDS, compute_attention_entropy
from src.training.focal_loss import FocalLoss, compute_class_weights

# Import shared utilities from evaluate_cv
import scripts.evaluate_cv as evcv
from scripts.evaluate_cv import (
    merge_labels_np, set_seed, apply_smote_fold,
    make_hit_groups, train_hit_model, phase2_refine,
    eval_cascade_on_test, MERGED_NAMES, NUM_MERGED,
)
from src.data.cv_data import load_clean_dataset, scale_fold, smoke_subsample


# ── Flat Transformer (A1: no grouping) ──────────────────────────────────

class FlatTransformerIDS(nn.Module):
    """Vanilla Transformer treating all 72 features as a flat sequence.
    Same capacity as HiT-IDS but without hierarchical grouping."""

    def __init__(self, num_features, d_emb, nhead, num_layers,
                 dim_feedforward, dropout, num_classes):
        super().__init__()
        self.num_features = num_features
        self.d_emb = d_emb

        # Per-feature embeddings (same as HiT-IDS)
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_emb) for _ in range(num_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_emb) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, num_classes),
        )

    def forward(self, x, return_explanations=False):
        B = x.shape[0]
        tokens = [emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)]
        tokens = torch.stack(tokens, dim=1)  # (B, 72, d_emb)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 73, d_emb)
        encoded = self.encoder(tokens)
        logits = self.head(encoded[:, 0])
        return logits


# ── ReLU variant (A4) ───────────────────────────────────────────────────

def patch_gelu_to_relu(model):
    """Replace all GELU activations with ReLU in a model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            setattr(parent, child_name, nn.ReLU())
    # Also patch TransformerEncoderLayer activation
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            if hasattr(module, 'activation') and callable(module.activation):
                module.activation = nn.functional.relu
    return model


# ── Ablation Configurations ─────────────────────────────────────────────

ABLATION_CONFIGS = {
    "A1": {"name": "No Grouping (Flat)", "desc": "Flat 72-token Transformer, direct 13-class"},
    "A2": {"name": "No Cascade (Direct 13-class)", "desc": "Single model, direct 13-class"},
    "A3": {"name": "No Phase 2", "desc": "Skip attention entropy regularization"},
    "A4": {"name": "ReLU (not GELU)", "desc": "Replace GELU with ReLU activation"},
    "A5": {"name": "No SMOTE", "desc": "No SMOTE, only class weights + Focal Loss"},
    "A6": {"name": "Single Head (h=1)", "desc": "Single attention head instead of 8"},
}


def create_ablation_model(ablation_id, config, num_classes, device):
    """Create the model variant for a given ablation."""
    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    model_cfg = config["model"]
    total_features = sum(group_sizes)

    if ablation_id == "A1":
        # Single-group HiT-IDS: all 72 features in one group.
        # Uses the EXACT same architecture/training as HiT-IDS,
        # only difference = no hierarchical grouping.
        model = HiTIDS(
            group_sizes=[total_features],
            group_names=["all_features"],
            d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
            intra_layers=model_cfg["intra_group_layers"],
            inter_layers=model_cfg["inter_group_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"], num_classes=num_classes,
        )
        # Resize inter-group pos_embedding: 5 (4 groups + CLS) → 2 (1 group + CLS)
        old_pe = model.inter_transformer.pos_embedding  # (1, 5, d_emb)
        new_pe = nn.Parameter(old_pe.data[:, :2, :].clone())  # (1, 2, d_emb)
        model.inter_transformer.pos_embedding = new_pe
        return model.to(device), "hit"

    elif ablation_id == "A4":
        model = HiTIDS(
            group_sizes=group_sizes, group_names=group_names,
            d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
            intra_layers=model_cfg["intra_group_layers"],
            inter_layers=model_cfg["inter_group_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"], num_classes=num_classes,
        )
        model = patch_gelu_to_relu(model)
        return model.to(device), "hit"

    elif ablation_id == "A6":
        model = HiTIDS(
            group_sizes=group_sizes, group_names=group_names,
            d_emb=model_cfg["d_emb"], nhead=1,  # Single head
            intra_layers=model_cfg["intra_group_layers"],
            inter_layers=model_cfg["inter_group_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"], num_classes=num_classes,
        )
        return model.to(device), "hit"

    else:
        # A2, A3, A5 use standard HiT-IDS architecture
        model = HiTIDS(
            group_sizes=group_sizes, group_names=group_names,
            d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
            intra_layers=model_cfg["intra_group_layers"],
            inter_layers=model_cfg["inter_group_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"], num_classes=num_classes,
        )
        return model.to(device), "hit"


def train_flat_model(model, X_train, y_train, X_val, y_val,
                     config, device, max_epochs, patience, num_classes):
    """Train flat transformer model (no groups).

    Uses standard CrossEntropy + class weights instead of Focal Loss,
    because the Optuna-tuned gamma=4.5 suppresses gradients for the
    larger flat attention matrices (73×73 vs HiT-IDS's 26×26 max).
    """
    batch_size = config["training"]["phase1"]["batch_size"]
    lr = 1e-4  # Conservative LR for flat model (Optuna's 4.5e-4 is too aggressive)
    wd = config["training"]["phase1"]["weight_decay"]

    y_t = torch.tensor(y_train, dtype=torch.long)
    y_v = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_t)
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_v)

    # Weighted sampler
    class_counts = torch.bincount(y_t, minlength=num_classes).float().clamp(min=1)
    sw = 1.0 / class_counts[y_t]
    sampler = torch.utils.data.WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Standard CE with class weights (instead of Focal Loss with gamma=4.5)
    alpha = compute_class_weights(y_t, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=alpha.to(device), label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Warmup + cosine schedule
    total_steps = len(train_loader) * max_epochs
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, wait, best_state = 0.0, 0, None

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            loss = criterion(model(xb.to(device)), yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        ps, ls_ = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                ps.append(model(xb.to(device)).argmax(1).cpu())
                ls_.append(yb)
        vf = f1_score(torch.cat(ls_).numpy(), torch.cat(ps).numpy(),
                      average="macro", zero_division=0)

        if vf > best_f1:
            best_f1 = vf
            wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_f1


def run_ablation_fold(ablation_id, X_train, y_train, X_test, y_test,
                      config, device, seed, X_val=None, y_val=None):
    """Run one ablation on one fold and return macro-F1."""
    set_seed(seed)

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())

    if X_val is not None:
        # REAL validation rows carved from the training fold before resampling
        X_tr, y_tr, X_vl, y_vl = X_train, y_train, X_val, y_val
    else:
        # Legacy: train/val split (90/10) from the resampled set
        n = len(y_train)
        idx = np.random.RandomState(seed).permutation(n)
        val_size = max(1, n // 10)
        val_idx, tr_idx = idx[:val_size], idx[val_size:]
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_vl, y_vl = X_train[val_idx], y_train[val_idx]

    def make_groups(X):
        groups = {}
        offset = 0
        for name in group_names:
            size = len(fg_cfg[name])
            groups[name] = torch.tensor(X[:, offset:offset+size], dtype=torch.float32)
            offset += size
        return groups

    groups_tr = make_groups(X_tr)
    groups_vl = make_groups(X_vl)
    groups_test = make_groups(X_test)

    # ── A2: No cascade → direct 13-class ──
    if ablation_id == "A2":
        model, _ = create_ablation_model("A2", config, NUM_MERGED, device)
        train_hit_model(model, X_tr, y_tr, X_vl, y_vl,
                        groups_tr, groups_vl, config, device,
                        max_epochs=50, patience=10, num_classes=NUM_MERGED)
        phase2_refine(model, X_tr, y_tr, groups_tr, config, device,
                      num_classes=NUM_MERGED, max_epochs=10, patience=5)
        # Evaluate directly
        model.eval()
        test_ds = HiTIDSDataset(
            X=torch.tensor(X_test, dtype=torch.float32),
            y=torch.tensor(y_test, dtype=torch.long),
            group_data=groups_test)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                                 collate_fn=collate_fn, num_workers=4)
        ps, ls_ = [], []
        with torch.no_grad():
            for gd, lb in test_loader:
                X = torch.cat(list(gd.values()), dim=1).to(device)
                ps.append(model(X).argmax(1).cpu())
                ls_.append(lb)
        macro_f1 = f1_score(torch.cat(ls_).numpy(), torch.cat(ps).numpy(),
                            average="macro", zero_division=0)
        return {"macro_f1": float(macro_f1)}

    # ── A1: Single-group HiT-IDS (no hierarchical grouping) ──
    # Uses the full HiT-IDS cascade pipeline but with all 72 features
    # in a single group, isolating the effect of hierarchical grouping.
    if ablation_id == "A1":
        # Override groups: single group with all features
        def make_single_group(X):
            return {"all_features": torch.tensor(X, dtype=torch.float32)}
        groups_tr = make_single_group(X_tr)
        groups_vl = make_single_group(X_vl)
        groups_test = make_single_group(X_test)

        # Binary stage
        y_tr_bin = (y_tr > 0).astype(np.int64)
        y_vl_bin = (y_vl > 0).astype(np.int64)
        bin_model, _ = create_ablation_model("A1", config, 2, device)
        train_hit_model(bin_model, X_tr, y_tr_bin, X_vl, y_vl_bin,
                        groups_tr, groups_vl, config, device,
                        max_epochs=50, patience=10, num_classes=2)
        phase2_refine(bin_model, X_tr, y_tr_bin, groups_tr, config, device,
                      num_classes=2, max_epochs=10, patience=5)

        # Attack stage
        atk_mask_tr = y_tr > 0
        atk_mask_vl = y_vl > 0
        if atk_mask_tr.sum() == 0 or atk_mask_vl.sum() == 0:
            return None
        X_tr_a, y_tr_a = X_tr[atk_mask_tr], y_tr[atk_mask_tr] - 1
        X_vl_a, y_vl_a = X_vl[atk_mask_vl], y_vl[atk_mask_vl] - 1
        g_tr_a = {k: v[atk_mask_tr] for k, v in groups_tr.items()}
        g_vl_a = {k: v[atk_mask_vl] for k, v in groups_vl.items()}

        num_atk = NUM_MERGED - 1
        atk_model, _ = create_ablation_model("A1", config, num_atk, device)
        train_hit_model(atk_model, X_tr_a, y_tr_a, X_vl_a, y_vl_a,
                        g_tr_a, g_vl_a, config, device,
                        max_epochs=100, patience=15, num_classes=num_atk)
        phase2_refine(atk_model, X_tr_a, y_tr_a, g_tr_a, config, device,
                      num_classes=num_atk, max_epochs=10, patience=5)

        # Cascade evaluation
        g_test = {k: v for k, v in groups_test.items()}
        batch_size = config["training"]["phase1"]["batch_size"]
        y_true, y_pred = eval_cascade_on_test(
            bin_model, atk_model, X_test, y_test, g_test, batch_size, device)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return {"macro_f1": float(macro_f1)}

    # ── A3, A4, A5, A6: Standard cascade with modifications ──
    # Binary model
    y_tr_bin = (y_tr > 0).astype(np.int64)
    y_vl_bin = (y_vl > 0).astype(np.int64)
    bin_model, _ = create_ablation_model(ablation_id, config, 2, device)
    train_hit_model(bin_model, X_tr, y_tr_bin, X_vl, y_vl_bin,
                    groups_tr, groups_vl, config, device,
                    max_epochs=50, patience=10, num_classes=2)

    if ablation_id != "A3":
        phase2_refine(bin_model, X_tr, y_tr_bin, groups_tr, config, device,
                      num_classes=2, max_epochs=10, patience=5)

    # Attack model
    atk_mask_tr = y_tr > 0
    atk_mask_vl = y_vl > 0
    if atk_mask_tr.sum() == 0 or atk_mask_vl.sum() == 0:
        return None

    X_tr_a, y_tr_a = X_tr[atk_mask_tr], y_tr[atk_mask_tr] - 1
    X_vl_a, y_vl_a = X_vl[atk_mask_vl], y_vl[atk_mask_vl] - 1
    g_tr_a = {k: v[atk_mask_tr] for k, v in groups_tr.items()}
    g_vl_a = {k: v[atk_mask_vl] for k, v in groups_vl.items()}

    num_atk = NUM_MERGED - 1
    atk_model, _ = create_ablation_model(ablation_id, config, num_atk, device)
    train_hit_model(atk_model, X_tr_a, y_tr_a, X_vl_a, y_vl_a,
                    g_tr_a, g_vl_a, config, device,
                    max_epochs=100, patience=15, num_classes=num_atk)

    if ablation_id != "A3":
        phase2_refine(atk_model, X_tr_a, y_tr_a, g_tr_a, config, device,
                      num_classes=num_atk, max_epochs=10, patience=5)

    # Cascade evaluation
    batch_size = config["training"]["phase1"]["batch_size"]
    y_true, y_pred = eval_cascade_on_test(
        bin_model, atk_model, X_test, y_test, groups_test, batch_size, device)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"macro_f1": float(macro_f1)}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HiT-IDS Ablation Study")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--smote-target", type=int, default=20000)
    parser.add_argument("--ablations", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                        help="Which ablations to run (e.g., A1 A4)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick pipeline check: 3%% subsample, 2 folds, 2 epochs")
    parser.add_argument("--tag", default="clean")
    parser.add_argument("--val-source", choices=["real", "smote"], default="real")
    parser.add_argument("--no-phase2", action="store_true")
    args = parser.parse_args()
    if args.no_phase2:
        evcv.PHASE2 = False

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = args.folds
    tables_dir = config["paths"]["tables"]
    os.makedirs(tables_dir, exist_ok=True)

    tag = args.tag + ("_smoke" if args.smoke else "")
    if args.smoke:
        K = 2
        evcv.EPOCH_CAP = 2
    progress_path = os.path.join(tables_dir, f"ablation_progress_{tag}.json")

    # Resume support
    if args.resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            prev = json.load(f)
        results = prev.get("results", {})
        completed = set(tuple(c) for c in prev.get("completed", []))
        print(f"\n  ✓ Resuming: {len(completed)} steps completed")
    else:
        results = {}
        completed = set()

    print("=" * 70)
    print("  HiT-IDS ABLATION STUDY")
    print(f"  Ablations: {args.ablations}")
    print(f"  Folds: {K}")
    print("=" * 70)

    # Load REAL, UNSCALED rows (leak-free; see src/data/cv_data.py)
    X_all, y_all_orig, _, group_indices = load_clean_dataset(args.config)
    y_all = merge_labels_np(y_all_orig)
    if args.smoke:
        X_all, y_all = smoke_subsample(X_all, y_all, frac=0.03)
        print(f"  [smoke] subsampled to {len(y_all):,} rows")

    print(f"  Total REAL samples: {len(y_all):,}")
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    for ablation_id in args.ablations:
        cfg = ABLATION_CONFIGS[ablation_id]
        results.setdefault(ablation_id, [])
        print(f"\n{'═' * 70}")
        print(f"  {ablation_id}: {cfg['name']} — {cfg['desc']}")
        print(f"{'═' * 70}")

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
            step_key = (ablation_id, fold_i)
            if step_key in completed:
                print(f"  Fold {fold_i+1}: SKIPPED (completed)")
                continue

            print(f"\n  Fold {fold_i+1}/{K}")
            t0 = time.time()

            X_fold_train, y_fold_train = X_all[train_idx], y_all[train_idx]
            X_fold_test, y_fold_test = X_all[test_idx], y_all[test_idx]

            # Scale with RobustScaler fitted on the training fold only
            X_fold_train, X_fold_test, _ = scale_fold(
                X_fold_train, X_fold_test, group_indices)

            # Real validation rows for early stopping (before resampling)
            X_val_real = y_val_real = None
            if args.val_source == "real":
                from sklearn.model_selection import train_test_split
                tr_i, va_i = train_test_split(
                    np.arange(len(y_fold_train)), test_size=0.10,
                    stratify=y_fold_train, random_state=42 + fold_i)
                X_val_real, y_val_real = X_fold_train[va_i], y_fold_train[va_i]
                X_fold_train, y_fold_train = X_fold_train[tr_i], y_fold_train[tr_i]

            # Apply SMOTE (except A5)
            if ablation_id == "A5":
                X_train_s, y_train_s = X_fold_train.astype(np.float32), y_fold_train.astype(np.int64)
                print(f"    No SMOTE (A5): {len(X_train_s):,} samples")
            else:
                X_train_s, y_train_s = apply_smote_fold(
                    X_fold_train, y_fold_train,
                    target_count=args.smote_target, random_state=42 + fold_i)
                print(f"    SMOTE: {len(X_fold_train):,} → {len(X_train_s):,}")

            try:
                result = run_ablation_fold(
                    ablation_id, X_train_s, y_train_s,
                    X_fold_test.astype(np.float32), y_fold_test.astype(np.int64),
                    config, device, seed=42 + fold_i,
                    X_val=X_val_real, y_val=y_val_real)

                if result:
                    results[ablation_id].append(result)
                    completed.add(step_key)
                    # Save progress
                    with open(progress_path, "w") as f:
                        json.dump({"results": results,
                                   "completed": [list(c) for c in completed]}, f, indent=2)
                    print(f"    → Macro-F1: {result['macro_f1']:.4f} [{time.time()-t0:.0f}s] ✓")
                else:
                    print(f"    → FAILED (no result)")
            except Exception as e:
                print(f"    → ERROR: {e}")
                import traceback; traceback.print_exc()

    # ── Summary ──
    print("\n\n" + "=" * 70)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 70)

    # Reference: Full HiT-IDS from the CLEAN CV run with the same tag
    ref_path = os.path.join(tables_dir, f"cv_evaluation_{tag}.json")
    ref_f1 = float("nan")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)
        ref_f1 = ref.get("summary", {}).get("HiT-IDS (Cascade)", {}).get("mean_macro_f1", ref_f1)
    else:
        print(f"  [WARN] {ref_path} not found — ΔF1 will be NaN until the clean CV run exists")

    print(f"\n  {'Ablation':<35s} {'Mean F1':>8s} {'±Std':>8s} {'ΔF1':>8s}")
    print("  " + "-" * 65)
    print(f"  {'Full HiT-IDS (reference)':<35s} {ref_f1:>8.4f} {'—':>8s} {'—':>8s}")

    summary = {"reference": {"mean_macro_f1": ref_f1}}
    for abl_id in args.ablations:
        f1s = [r["macro_f1"] for r in results.get(abl_id, [])]
        if f1s:
            arr = np.array(f1s)
            delta = arr.mean() - ref_f1
            summary[abl_id] = {
                "name": ABLATION_CONFIGS[abl_id]["name"],
                "desc": ABLATION_CONFIGS[abl_id]["desc"],
                "mean_macro_f1": float(arr.mean()),
                "std_macro_f1": float(arr.std()),
                "delta_f1": float(delta),
                "per_fold_f1": f1s,
            }
            print(f"  {abl_id}: {ABLATION_CONFIGS[abl_id]['name']:<30s} "
                  f"{arr.mean():>8.4f} ±{arr.std():>6.4f} {delta:>+8.4f}")

    # Save final results
    summary["protocol"] = {
        "data_source": "raw CSVs → clean → REAL rows only (src/data/cv_data.py)",
        "scaling": "RobustScaler per feature group, fit on training fold only",
        "resampling": "adaptive SMOTE + BENIGN cap 200K, training fold only (A5: none)",
        "smoke": bool(args.smoke), "epoch_cap": evcv.EPOCH_CAP, "k_folds": K,
        "validation_for_early_stopping": args.val_source,
    }
    out_path = os.path.join(tables_dir, f"ablation_results_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 70)
    print("  ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

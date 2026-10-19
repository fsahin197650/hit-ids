#!/usr/bin/env python3
"""
5-Fold Stratified Cross-Validation with Statistical Significance Tests.

All models (HiT-IDS Cascade + baselines) trained on the SAME folds.
Each fold produces paired observations → valid Wilcoxon signed-rank test.

DATA SOURCE (fixed 2026-09-29):
    The previous version concatenated data/splits/{train,val,test}.pt and folded
    the result. train.pt is already SMOTE-oversampled + BENIGN-undersampled, so
    synthetic rows leaked into every test fold and all reported CV numbers were
    inflated. This version loads the REAL, UNSCALED rows from the raw CSVs
    (src/data/cv_data.py), and fits the RobustScaler + SMOTE inside each
    training fold only. Outputs go to *_clean.json so the old (contaminated)
    files remain as an audit trail.

Pipeline per fold:
    1. Stratified split of REAL rows → train/test
    2. RobustScaler (per feature group) fit on train fold only
    3. SMOTE + majority undersample on train fold only
    4. Train ALL models on same train set
    5. Evaluate ALL models on the same REAL test fold
    6. Collect macro-F1, per-class P/R/F1, confusion matrix, ROC/PR-AUC,
       predictions + probabilities (npz) per model per fold

After all folds:
    - Wilcoxon signed-rank test (5 paired observations)
    - Cohen's d effect size
    - 95% CI

Usage:
    python scripts/evaluate_cv.py --config configs/config.yaml
    python scripts/evaluate_cv.py --config configs/config.yaml --resume  # Continue after power outage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, precision_score, recall_score)
from scipy import stats
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HiTIDSDataset, collate_fn
from src.data.cv_data import load_clean_dataset, scale_fold, smoke_subsample
from src.models.hit_ids import HiTIDS
from src.models.baselines import SklearnBaseline, create_baseline, train_pytorch_baseline
from src.training.focal_loss import FocalLoss, compute_class_weights

# ── Label Merging ────────────────────────────────────────────────────────
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

# Set by --smoke to cap epochs for a quick end-to-end pipeline check.
EPOCH_CAP = None
# Set False (--no-phase2) to skip the attention-entropy fine-tuning stage.
PHASE2 = True


def _cap(max_epochs: int) -> int:
    return max_epochs if EPOCH_CAP is None else min(max_epochs, EPOCH_CAP)


def merge_labels_np(y: np.ndarray) -> np.ndarray:
    return np.array([MERGE_MAP[int(v)] for v in y])


def fold_metrics(y_true, y_pred, proba=None):
    """Full per-fold metric bundle on REAL test rows.

    proba: (N, NUM_MERGED) class probabilities or None.
    """
    labels = list(range(NUM_MERGED))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    present = [c for c in labels if s[c] > 0]
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1_present_only": float(np.mean([f[c] for c in present])) if present else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": {
            MERGED_NAMES[c]: {"precision": float(p[c]), "recall": float(r[c]),
                              "f1": float(f[c]), "support": int(s[c])}
            for c in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    # Binary view (attack vs benign) — what a SOC actually sees first
    yt_b, yp_b = (y_true > 0).astype(int), (y_pred > 0).astype(int)
    out["binary"] = {
        "f1": float(f1_score(yt_b, yp_b, zero_division=0)),
        "precision": float(precision_score(yt_b, yp_b, zero_division=0)),
        "recall": float(recall_score(yt_b, yp_b, zero_division=0)),
        "fpr": float(((yt_b == 0) & (yp_b == 1)).sum() / max((yt_b == 0).sum(), 1)),
    }
    if proba is not None:
        proba = np.asarray(proba, dtype=np.float64)
        proba = proba / np.clip(proba.sum(1, keepdims=True), 1e-12, None)
        p_attack = 1.0 - proba[:, 0]
        try:
            out["binary"]["roc_auc"] = float(roc_auc_score(yt_b, p_attack))
            out["binary"]["pr_auc"] = float(average_precision_score(yt_b, p_attack))
        except ValueError:
            pass
        per_auc, per_ap = {}, {}
        for c in present:
            yc = (y_true == c).astype(int)
            if 0 < yc.sum() < len(yc):
                per_auc[MERGED_NAMES[c]] = float(roc_auc_score(yc, proba[:, c]))
                per_ap[MERGED_NAMES[c]] = float(average_precision_score(yc, proba[:, c]))
        out["per_class_roc_auc"] = per_auc
        out["per_class_pr_auc"] = per_ap
        if per_auc:
            out["macro_roc_auc_ovr"] = float(np.mean(list(per_auc.values())))
            out["macro_pr_auc"] = float(np.mean(list(per_ap.values())))
    return out


def save_fold_predictions(out_dir, fold_i, model_name, y_true, y_pred, proba=None):
    """Persist predictions so confusion matrices / ROC curves can be re-plotted."""
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(out_dir, f"fold{fold_i}_{safe}.npz")
    kw = {"y_true": y_true.astype(np.int16), "y_pred": y_pred.astype(np.int16)}
    if proba is not None:
        kw["proba"] = np.asarray(proba, dtype=np.float16)
    np.savez_compressed(path, **kw)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0


def apply_smote_fold(X_train, y_train, target_count=20_000, random_state=42):
    """Apply adaptive SMOTE + majority undersample to a fold's training data.

    Adaptive strategy:
      - Classes with <50 real samples  → target = max(real*3, 200)
      - Classes with 50-500 samples    → target = 5_000
      - Classes with 500-5000 samples  → target = target_count
      - Classes with >5000             → no oversampling
    """
    from imblearn.over_sampling import SMOTE

    unique, counts = np.unique(y_train, return_counts=True)
    cls_count = dict(zip(unique.astype(int), counts.astype(int)))

    # Adaptive per-class targets
    sampling_strategy = {}
    for cls, cnt in cls_count.items():
        if cnt < 50:
            tgt = max(cnt * 3, 200)  # conservative for extreme minority
        elif cnt < 500:
            tgt = 5_000
        elif cnt < 5_000:
            tgt = target_count
        else:
            continue  # already large enough
        if tgt > cnt:
            sampling_strategy[cls] = tgt

    if sampling_strategy:
        # k_neighbors must be < smallest class count
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
        keep = np.random.RandomState(random_state).choice(maj_idx, max_majority, replace=False)
        other = np.where(y_res != maj_cls)[0]
        indices = np.concatenate([keep, other])
        np.random.RandomState(random_state).shuffle(indices)
        X_res, y_res = X_res[indices], y_res[indices]

    return X_res.astype(np.float32), y_res.astype(np.int64)


# ── HiT-IDS Training ────────────────────────────────────────────────────

def make_hit_groups(X, config):
    """Split flat features into group dict for HiT-IDS."""
    fg_cfg = config["feature_groups"]
    groups = {}
    offset = 0
    for name in fg_cfg:
        size = len(fg_cfg[name])
        groups[name] = torch.tensor(X[:, offset:offset+size], dtype=torch.float32)
        offset += size
    return groups


def train_hit_model(model, X_train, y_train, X_val, y_val, groups_train, groups_val,
                    config, device, max_epochs, patience, num_classes,
                    rare_boost=True):
    """Train HiT-IDS model (Phase 1) and return best val F1."""
    batch_size = config["training"]["phase1"]["batch_size"]
    lr = config["training"]["phase1"]["learning_rate"]
    wd = config["training"]["phase1"]["weight_decay"]
    gamma = config["training"]["focal_loss"]["gamma"]
    ls = config["training"]["focal_loss"].get("label_smoothing", 0.0)

    y_t = torch.tensor(y_train, dtype=torch.long)
    y_v = torch.tensor(y_val, dtype=torch.long)

    train_ds = HiTIDSDataset(X=torch.tensor(X_train, dtype=torch.float32),
                             y=y_t, group_data=groups_train)
    val_ds = HiTIDSDataset(X=torch.tensor(X_val, dtype=torch.float32),
                           y=y_v, group_data=groups_val)

    # Weighted sampling with rare-class boost
    class_counts = torch.bincount(y_t, minlength=num_classes).float().clamp(min=1)
    sw = 1.0 / class_counts[y_t]
    if rare_boost:
        # Extra 3x weight for classes with <500 samples
        for cls_id in range(num_classes):
            if class_counts[cls_id] < 500:
                mask = (y_t == cls_id)
                sw[mask] *= 3.0
    sampler = torch.utils.data.WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    alpha = compute_class_weights(y_t, num_classes=num_classes)
    criterion = FocalLoss(alpha=alpha.to(device), gamma=gamma, label_smoothing=ls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_f1, wait, best_state = 0.0, 0, None

    for epoch in range(_cap(max_epochs)):
        model.train()
        for gd, lb in train_loader:
            g = {k: v.to(device) for k, v in gd.items()}
            X = torch.cat(list(g.values()), dim=1)
            loss = criterion(model(X, return_explanations=False), lb.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        ps, ls_ = [], []
        with torch.no_grad():
            for gd, lb in val_loader:
                X = torch.cat(list(gd.values()), dim=1).to(device)
                ps.append(model(X, return_explanations=False).argmax(1).cpu())
                ls_.append(lb)
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


def phase2_refine(model, X_train, y_train, groups_train, config, device,
                  num_classes, max_epochs=10, patience=5):
    """Phase 2: Attention entropy regularisation to sharpen attention patterns.

    L_total = L_focal + lambda * L_entropy
    Lower LR, shorter training, focus on explanation quality.
    """
    from src.models.hit_ids import compute_attention_entropy

    if not PHASE2:
        print("      Phase 2 skipped (--no-phase2)")
        return
    lam = config["training"]["phase2"].get("attention_entropy_lambda", 0.01)
    lr = config["training"]["phase2"].get("learning_rate", 1e-5)
    batch_size = config["training"]["phase1"]["batch_size"]
    gamma = config["training"]["focal_loss"]["gamma"]
    ls = config["training"]["focal_loss"].get("label_smoothing", 0.0)

    y_t = torch.tensor(y_train, dtype=torch.long)
    train_ds = HiTIDSDataset(X=torch.tensor(X_train, dtype=torch.float32),
                             y=y_t, group_data=groups_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    alpha = compute_class_weights(y_t, num_classes=num_classes)
    criterion = FocalLoss(alpha=alpha.to(device), gamma=gamma, label_smoothing=ls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss, wait, best_state = float('inf'), 0, None
    avg_loss, epoch = float('nan'), -1

    for epoch in range(_cap(max_epochs)):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for gd, lb in train_loader:
            g = {k: v.to(device) for k, v in gd.items()}
            X = torch.cat(list(g.values()), dim=1)
            logits, explanations = model(X, return_explanations=True)

            l_cls = criterion(logits, lb.to(device))

            # Attention entropy from all attention weights
            attn_weights = []
            # raw_intra_attn is a dict: group_name -> list of attn tensors
            intra = explanations.get('raw_intra_attn', {})
            if isinstance(intra, dict):
                for attn_list in intra.values():
                    attn_weights.extend(attn_list)
            # raw_inter_attn is a list of attn tensors
            inter = explanations.get('raw_inter_attn', [])
            if isinstance(inter, list):
                attn_weights.extend(inter)
            if attn_weights:
                l_ent = compute_attention_entropy(attn_weights)
            else:
                l_ent = torch.tensor(0.0, device=device)

            loss = l_cls + lam * l_ent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"      Phase 2 done: {epoch+1} epochs, final loss={avg_loss:.4f}")


def eval_cascade_on_test_full(bin_model, atk_model, X_test, y_test, groups_test,
                              batch_size, device):
    """Evaluate cascade on test fold; also return combined class probabilities.

    Combined probability: p(BENIGN) = p_bin[0];  p(attack k) = p_bin[1] * p_atk[k-1].
    The hard prediction follows the deployed cascade rule exactly (argmax of the
    binary stage gates the attack stage), NOT the argmax of the combined vector.
    """
    y_t = torch.tensor(y_test, dtype=torch.long)
    test_ds = HiTIDSDataset(X=torch.tensor(X_test, dtype=torch.float32),
                            y=y_t, group_data=groups_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)

    bin_model.eval()
    atk_model.eval()
    all_preds, all_labels, all_proba = [], [], []

    with torch.no_grad():
        for gd, lb in test_loader:
            X = torch.cat(list(gd.values()), dim=1).to(device)
            p_bin = torch.softmax(bin_model(X, return_explanations=False), dim=1)
            p_atk = torch.softmax(atk_model(X, return_explanations=False), dim=1)
            proba = torch.cat([p_bin[:, :1], p_bin[:, 1:2] * p_atk], dim=1)

            bp = p_bin.argmax(1)
            final = torch.zeros(len(lb), dtype=torch.long)
            mask = bp == 1
            if mask.any():
                final[mask] = p_atk[mask].argmax(1).cpu() + 1
            all_preds.append(final)
            all_labels.append(lb)
            all_proba.append(proba.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    proba = torch.cat(all_proba).numpy()
    return y_true, y_pred, proba


def eval_cascade_on_test(bin_model, atk_model, X_test, y_test, groups_test,
                         batch_size, device):
    """Backward-compatible wrapper (used by run_ablation.py)."""
    y_true, y_pred, _ = eval_cascade_on_test_full(
        bin_model, atk_model, X_test, y_test, groups_test, batch_size, device)
    return y_true, y_pred


def train_eval_cascade_fold(X_train, y_train, X_test, y_test,
                            groups_train, groups_val, groups_test,
                            config, device, seed,
                            fold_i=None, pred_dir=None, ckpt_dir=None,
                            X_val=None, y_val=None):
    """Train cascade (binary + attack) and evaluate on test fold.

    X_val/y_val: REAL (un-resampled) rows carved from the training fold, used
    only for early stopping / model selection. If None, falls back to the old
    behaviour of splitting 10% off the (SMOTE'd) training set.
    """
    set_seed(seed)

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    group_sizes = [len(fg_cfg[name]) for name in group_names]
    model_cfg = config["model"]
    batch_size = config["training"]["phase1"]["batch_size"]

    if X_val is not None:
        X_tr, y_tr = X_train, y_train
        X_vl, y_vl = X_val, y_val
        g_tr = groups_train
        g_vl = make_hit_groups(X_val, config)
    else:
        # Legacy: split train into train/val (90/10) from the resampled set
        n = len(y_train)
        idx = np.random.RandomState(seed).permutation(n)
        val_size = max(1, n // 10)
        val_idx, tr_idx = idx[:val_size], idx[val_size:]
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_vl, y_vl = X_train[val_idx], y_train[val_idx]
        g_tr = {k: v[tr_idx] for k, v in groups_train.items()}
        g_vl = {k: v[val_idx] for k, v in groups_train.items()}

    # ── BINARY ──
    y_tr_bin = (y_tr > 0).astype(np.int64)
    y_vl_bin = (y_vl > 0).astype(np.int64)

    bin_model = HiTIDS(
        group_sizes=group_sizes, group_names=group_names,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"], num_classes=2,
    ).to(device)

    g_tr_t = {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v
              for k, v in g_tr.items()}
    g_vl_t = {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v
              for k, v in g_vl.items()}

    bin_f1 = train_hit_model(
        bin_model, X_tr, y_tr_bin, X_vl, y_vl_bin, g_tr_t, g_vl_t,
        config, device, max_epochs=50, patience=10, num_classes=2,
        rare_boost=False)
    print(f"      Binary F1 (Phase 1): {bin_f1:.4f}")

    # Phase 2 refinement for binary model
    print(f"      Phase 2 refinement (binary)...")
    phase2_refine(bin_model, X_tr, y_tr_bin, g_tr_t, config, device,
                  num_classes=2, max_epochs=10, patience=5)

    # ── ATTACK ──
    atk_mask_tr = y_tr > 0
    atk_mask_vl = y_vl > 0

    if atk_mask_tr.sum() > 0 and atk_mask_vl.sum() > 0:
        X_tr_a, y_tr_a = X_tr[atk_mask_tr], y_tr[atk_mask_tr] - 1
        X_vl_a, y_vl_a = X_vl[atk_mask_vl], y_vl[atk_mask_vl] - 1
        g_tr_a = {k: v[atk_mask_tr] for k, v in g_tr_t.items()}
        g_vl_a = {k: v[atk_mask_vl] for k, v in g_vl_t.items()}

        num_atk = NUM_MERGED - 1
        atk_model = HiTIDS(
            group_sizes=group_sizes, group_names=group_names,
            d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
            intra_layers=model_cfg["intra_group_layers"],
            inter_layers=model_cfg["inter_group_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"], num_classes=num_atk,
        ).to(device)

        atk_f1 = train_hit_model(
            atk_model, X_tr_a, y_tr_a, X_vl_a, y_vl_a, g_tr_a, g_vl_a,
            config, device, max_epochs=100, patience=15, num_classes=num_atk,
            rare_boost=True)
        print(f"      Attack F1 (Phase 1): {atk_f1:.4f}")

        # Phase 2 refinement for attack model
        print(f"      Phase 2 refinement (attack)...")
        phase2_refine(atk_model, X_tr_a, y_tr_a, g_tr_a, config, device,
                      num_classes=num_atk, max_epochs=10, patience=5)
    else:
        print("      [WARNING] No attack samples in val split")
        return None

    # ── Cascade Test (REAL rows only) ──
    y_true, y_pred, proba = eval_cascade_on_test_full(
        bin_model, atk_model, X_test, y_test, groups_test, batch_size, device)

    result = fold_metrics(y_true, y_pred, proba)
    # Validation-split (SMOTE'd) stage scores — kept for reference only,
    # they are NOT test-fold numbers.
    result["val_binary_f1_phase1"] = float(bin_f1)
    result["val_attack_f1_phase1"] = float(atk_f1)

    if fold_i is not None:
        save_fold_predictions(pred_dir, fold_i, "HiT-IDS (Cascade)", y_true, y_pred, proba)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(bin_model.state_dict(), os.path.join(ckpt_dir, f"fold{fold_i}_binary.pt"))
            torch.save(atk_model.state_dict(), os.path.join(ckpt_dir, f"fold{fold_i}_attack.pt"))
    return result


def _sk_proba(baseline, X, num_classes):
    raw = baseline.predict_proba(X)
    classes = getattr(baseline.model, "classes_", None)
    proba = np.zeros((len(X), num_classes), dtype=np.float32)
    if classes is not None and len(classes) == raw.shape[1]:
        for j, c in enumerate(classes):
            proba[:, int(c)] = raw[:, j]
    else:
        proba[:, :raw.shape[1]] = raw
    return proba


def _torch_proba(model, X, device, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(torch.tensor(X[i:i+bs], dtype=torch.float32).to(device)), 1).cpu())
    return torch.cat(out).numpy()


def train_eval_baseline_fold(model_type, X_train, y_train, X_test, y_test,
                             num_features, num_classes, device, seed,
                             fold_i=None, pred_dir=None, X_val=None, y_val=None, ckpt_dir=None):
    """Train and eval a baseline on one fold (REAL test rows only).

    Also stores the validation-split probabilities (for threshold calibration)
    and the fitted model (for attribution-stability analyses) when fold_i is set.
    """
    set_seed(seed)
    proba = None
    proba_val = None

    if model_type in ("random_forest", "xgboost"):
        baseline = SklearnBaseline(model_type=model_type)
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        try:
            raw = baseline.predict_proba(X_test)
            classes = getattr(baseline.model, "classes_", None)
            proba = np.zeros((len(X_test), num_classes), dtype=np.float32)
            if classes is not None and len(classes) == raw.shape[1]:
                for j, c in enumerate(classes):
                    proba[:, int(c)] = raw[:, j]
            else:
                proba[:, :raw.shape[1]] = raw
        except Exception as e:  # pragma: no cover
            print(f"      [proba warning] {e}")
        if X_val is not None:
            proba_val = _sk_proba(baseline, X_val, num_classes)
        if fold_i is not None and ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            import pickle
            with open(os.path.join(ckpt_dir, f"fold{fold_i}_{model_type}.pkl"), "wb") as f:
                pickle.dump(baseline.model, f)
    else:
        model = create_baseline(model_type, num_features, num_classes)
        if X_val is not None:
            Xtr_, ytr_, Xva_, yva_ = X_train, y_train, X_val, y_val
        else:
            # Legacy: hold out 10% of the (SMOTE'd) training fold
            n = len(y_train)
            idx = np.random.RandomState(seed).permutation(n)
            n_val = max(1, n // 10)
            va, tr = idx[:n_val], idx[n_val:]
            Xtr_, ytr_, Xva_, yva_ = X_train[tr], y_train[tr], X_train[va], y_train[va]
        train_ds = TensorDataset(
            torch.tensor(Xtr_, dtype=torch.float32),
            torch.tensor(ytr_, dtype=torch.long))
        val_ds = TensorDataset(
            torch.tensor(Xva_, dtype=torch.float32),
            torch.tensor(yva_, dtype=torch.long))
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
        train_pytorch_baseline(model, train_loader, val_loader,
                               num_epochs=_cap(30), lr=1e-3, device=device)
        model.eval()
        preds, probs = [], []
        with torch.no_grad():
            for xb, _ in test_loader:
                logits = model(xb.to(device))
                preds.append(logits.argmax(1).cpu())
                probs.append(torch.softmax(logits, dim=1).cpu())
        y_pred = torch.cat(preds).numpy()
        proba = torch.cat(probs).numpy()
        if X_val is not None:
            proba_val = _torch_proba(model, X_val, device)
        if fold_i is not None and ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            extra = {"d_emb": 64, "nhead": 4, "num_layers": 4, "dim_feedforward": 256} if model_type == "vanilla_transformer" else {}
            torch.save({"model_state_dict": model.state_dict(), "num_features": num_features,
                        "num_classes": num_classes, **extra}, os.path.join(ckpt_dir, f"fold{fold_i}_{model_type}.pt"))

    result = fold_metrics(y_test, y_pred, proba)
    if fold_i is not None:
        save_fold_predictions(pred_dir, fold_i, model_type, y_test, y_pred, proba)
        if proba_val is not None:
            save_fold_predictions(pred_dir, fold_i, f"{model_type}_val", y_val, proba_val.argmax(1), proba_val)
    return result


# ── Main ─────────────────────────────────────────────────────────────────

def save_progress(progress_path, results, completed):
    """Save progress to disk after each model completes."""
    data = {"completed": completed, "results": results}
    # Write to temp then rename for atomicity (crash-safe)
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, progress_path)


def load_progress(progress_path):
    """Load progress from previous run."""
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            data = json.load(f)
        return data.get("results", {}), set(tuple(c) for c in data.get("completed", []))
    return {}, set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--smote-target", type=int, default=20000,
                        help="SMOTE target per minority class per fold")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (power outage safe)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick pipeline check: ~3%% stratified subsample, "
                             "2 folds, epochs capped at 2, separate output files")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild the clean dataset cache from raw CSVs")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to run (default: HiT-IDS + config baselines)")
    parser.add_argument("--tag", default="clean",
                        help="Suffix for output files (default: clean)")
    parser.add_argument("--no-phase2", action="store_true",
                        help="skip the attention-entropy fine-tuning (Phase 2) for HiT-IDS")
    parser.add_argument("--val-source", choices=["real", "smote"], default="real",
                        help="Early-stopping validation set: 'real' = 10%% of the REAL "
                             "training-fold rows carved out BEFORE resampling (default); "
                             "'smote' = legacy 10%% of the resampled set")
    args = parser.parse_args()

    global EPOCH_CAP, PHASE2
    if args.no_phase2:
        PHASE2 = False
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = args.folds
    tables_dir = config["paths"]["tables"]
    os.makedirs(tables_dir, exist_ok=True)

    tag = args.tag + ("_smoke" if args.smoke else "")
    if args.smoke:
        K = 2
        EPOCH_CAP = 2
    progress_path = os.path.join(tables_dir, f"cv_progress_{tag}.json")
    out_path = os.path.join(tables_dir, f"cv_evaluation_{tag}.json")
    pred_dir = os.path.join(tables_dir, f"cv_preds_{tag}")
    ckpt_dir = os.path.join(config["paths"]["checkpoints"], f"cv_{tag}")

    baselines = config.get("evaluation", {}).get("baselines", [])
    all_models = ["HiT-IDS (Cascade)"] + baselines
    if args.models:
        all_models = [m for m in all_models if m in args.models]
        baselines = [b for b in baselines if b in all_models]

    # ── Resume support ──
    if args.resume:
        prev_results, completed = load_progress(progress_path)
        results = {model: prev_results.get(model, []) for model in all_models}
        print(f"\n  ✓ Resuming from checkpoint: {len(completed)} steps completed")
    else:
        results = {model: [] for model in all_models}
        completed = set()

    print("=" * 70)
    print(f"  {K}-FOLD STRATIFIED CV WITH STATISTICAL TESTS")
    print(f"  Models: {all_models}")
    print(f"  Classes: {NUM_MERGED} (merged)")
    print(f"  SMOTE target: {args.smote_target:,} per fold")
    print(f"  Resume: {args.resume} | Smoke: {args.smoke} | Output tag: {tag}")
    print("=" * 70)

    # ── Load REAL, UNSCALED rows (no synthetic samples anywhere) ──
    X_all, y_all_orig, feature_cols, group_indices = load_clean_dataset(
        args.config, force_rebuild=args.force_rebuild)
    y_all = merge_labels_np(y_all_orig)
    if args.smoke:
        X_all, y_all = smoke_subsample(X_all, y_all, frac=0.03)
        print(f"  [smoke] subsampled to {len(y_all):,} rows")

    total_features = X_all.shape[1]
    print(f"\n  Total REAL samples: {len(y_all):,}")
    print(f"  Features: {total_features}")
    print(f"  Class distribution (real):")
    for cls_id in range(NUM_MERGED):
        cnt = (y_all == cls_id).sum()
        print(f"    {MERGED_NAMES[cls_id]}: {cnt:,}")

    # ── K-Fold CV ──
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    fg_cfg = config["feature_groups"]
    group_names = list(fg_cfg.keys())
    fold_info = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n{'═' * 70}")
        print(f"  FOLD {fold_i+1}/{K}")
        print(f"  Train: {len(train_idx):,} | Test: {len(test_idx):,}")
        print(f"{'═' * 70}")

        X_fold_train, y_fold_train = X_all[train_idx], y_all[train_idx]
        X_fold_test, y_fold_test = X_all[test_idx], y_all[test_idx]

        # Scale: RobustScaler per feature group, fit on the TRAINING fold only
        X_fold_train, X_fold_test, _ = scale_fold(X_fold_train, X_fold_test, group_indices)
        fold_info.append({
            "fold": fold_i,
            "n_train_real": int(len(train_idx)),
            "n_test_real": int(len(test_idx)),
            "test_class_counts": {MERGED_NAMES[c]: int((y_fold_test == c).sum())
                                  for c in range(NUM_MERGED)},
        })

        # Validation for early stopping: REAL rows carved out BEFORE resampling
        X_val_real = y_val_real = None
        if args.val_source == "real":
            from sklearn.model_selection import train_test_split
            tr_i, va_i = train_test_split(
                np.arange(len(y_fold_train)), test_size=0.10,
                stratify=y_fold_train, random_state=42 + fold_i)
            X_val_real, y_val_real = X_fold_train[va_i], y_fold_train[va_i]
            X_fold_train, y_fold_train = X_fold_train[tr_i], y_fold_train[tr_i]
            fold_info[-1]["n_val_real"] = int(len(va_i))
            fold_info[-1]["n_train_real_after_val"] = int(len(tr_i))
            print(f"  Real validation split: {len(va_i):,} rows "
                  f"(train fold now {len(tr_i):,})")

        # Apply SMOTE to training fold only
        print(f"  Applying SMOTE (target={args.smote_target:,})...")
        t0 = time.time()
        X_fold_train_s, y_fold_train_s = apply_smote_fold(
            X_fold_train, y_fold_train,
            target_count=args.smote_target,
            random_state=42 + fold_i,
        )
        print(f"  SMOTE done: {len(X_fold_train):,} → {len(X_fold_train_s):,} "
              f"({time.time()-t0:.0f}s)")
        fold_info[-1]["n_train_after_smote"] = int(len(y_fold_train_s))
        fold_info[-1]["train_class_counts_after_smote"] = {
            MERGED_NAMES[c]: int((y_fold_train_s == c).sum()) for c in range(NUM_MERGED)}

        # Create group tensors for test set
        groups_test = {}
        offset = 0
        for name in group_names:
            size = len(fg_cfg[name])
            groups_test[name] = torch.tensor(X_fold_test[:, offset:offset+size],
                                             dtype=torch.float32)
            offset += size

        # Create group tensors for SMOTE'd train set
        groups_train = {}
        offset = 0
        for name in group_names:
            size = len(fg_cfg[name])
            groups_train[name] = torch.tensor(X_fold_train_s[:, offset:offset+size],
                                              dtype=torch.float32)
            offset += size

        # ── HiT-IDS Cascade ──
        step_key = (fold_i, "HiT-IDS (Cascade)")
        if "HiT-IDS (Cascade)" not in all_models:
            pass
        elif step_key in completed:
            print(f"\n  [HiT-IDS (Cascade)] ✓ SKIPPED (already completed)")
        else:
            print(f"\n  [HiT-IDS (Cascade)]")
            t0 = time.time()
            try:
                hit_result = train_eval_cascade_fold(
                    X_fold_train_s, y_fold_train_s,
                    X_fold_test, y_fold_test,
                    groups_train, None, groups_test,
                    config, device, seed=42+fold_i,
                    fold_i=fold_i, pred_dir=pred_dir, ckpt_dir=ckpt_dir,
                    X_val=X_val_real, y_val=y_val_real)
                if hit_result:
                    results["HiT-IDS (Cascade)"].append(hit_result)
                    completed.add(step_key)
                    save_progress(progress_path, results, [list(c) for c in completed])
                    print(f"    → Macro-F1: {hit_result['macro_f1']:.4f} [{time.time()-t0:.0f}s] ✓ saved")
                else:
                    print(f"    → FAILED")
            except Exception as e:
                print(f"    → ERROR: {e}")
                import traceback; traceback.print_exc()

        # ── Baselines ──
        for bl in baselines:
            step_key = (fold_i, bl)
            if step_key in completed:
                print(f"\n  [{bl}] ✓ SKIPPED (already completed)")
                continue

            print(f"\n  [{bl}]")
            t0 = time.time()
            try:
                bl_result = train_eval_baseline_fold(
                    bl, X_fold_train_s, y_fold_train_s,
                    X_fold_test, y_fold_test,
                    total_features, NUM_MERGED, device, seed=42+fold_i,
                    fold_i=fold_i, pred_dir=pred_dir,
                    X_val=X_val_real, y_val=y_val_real, ckpt_dir=ckpt_dir)
                results[bl].append(bl_result)
                completed.add(step_key)
                save_progress(progress_path, results, [list(c) for c in completed])
                print(f"    → Macro-F1: {bl_result['macro_f1']:.4f} [{time.time()-t0:.0f}s] ✓ saved")
            except Exception as e:
                print(f"    → FAILED: {e}")

    # ════════════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  STATISTICAL ANALYSIS")
    print("=" * 70)

    # Summary Table
    print(f"\n  {'Model':<30s} {'Mean F1':>8s} {'±Std':>8s} {'Min':>6s} {'Max':>6s}")
    print("  " + "-" * 58)
    summary = {}
    for model in all_models:
        f1s = [r["macro_f1"] for r in results[model]]
        if f1s:
            arr = np.array(f1s)
            summary[model] = {
                "mean_macro_f1": float(arr.mean()),
                "std_macro_f1": float(arr.std()),
                "min_macro_f1": float(arr.min()),
                "max_macro_f1": float(arr.max()),
                "per_fold_f1": f1s,
                "mean_accuracy": float(np.mean([r["accuracy"] for r in results[model]])),
                "mean_weighted_f1": float(np.mean([r["weighted_f1"] for r in results[model]])),
            }
            # Per-class mean ± std across folds (real test rows)
            per_class = {}
            for cname in MERGED_NAMES:
                vals = {k: [r["per_class"][cname][k] for r in results[model]
                            if "per_class" in r]
                        for k in ("precision", "recall", "f1")}
                sup = [r["per_class"][cname]["support"] for r in results[model]
                       if "per_class" in r]
                if vals["f1"]:
                    per_class[cname] = {
                        **{f"{k}_mean": float(np.mean(v)) for k, v in vals.items()},
                        **{f"{k}_std": float(np.std(v)) for k, v in vals.items()},
                        "support_per_fold": sup,
                    }
            summary[model]["per_class"] = per_class
            for key in ("macro_roc_auc_ovr", "macro_pr_auc"):
                v = [r[key] for r in results[model] if key in r]
                if v:
                    summary[model][f"mean_{key}"] = float(np.mean(v))
            bins = [r["binary"] for r in results[model] if "binary" in r]
            if bins:
                summary[model]["binary"] = {
                    k: {"mean": float(np.mean([b[k] for b in bins if k in b])),
                        "std": float(np.std([b[k] for b in bins if k in b]))}
                    for k in ("f1", "precision", "recall", "fpr", "roc_auc", "pr_auc")
                    if any(k in b for b in bins)
                }
            # Summed confusion matrix over folds
            cms = [np.array(r["confusion_matrix"]) for r in results[model]
                   if "confusion_matrix" in r]
            if cms:
                summary[model]["confusion_matrix_sum"] = np.sum(cms, axis=0).tolist()
            print(f"  {model:<30s} {arr.mean():>8.4f} ±{arr.std():>6.4f} "
                  f"{arr.min():>6.4f} {arr.max():>6.4f}")

    # Pairwise Tests
    hit_f1s = np.array([r["macro_f1"] for r in results.get("HiT-IDS (Cascade)", [])])

    stat_tests = {}
    if len(hit_f1s) >= K:
        print(f"\n  {'─' * 60}")
        print(f"  Pairwise: HiT-IDS (Cascade) vs Baselines")
        print(f"  {'─' * 60}")

        for bl in baselines:
            bl_f1s = np.array([r["macro_f1"] for r in results.get(bl, [])])
            if len(bl_f1s) < K:
                continue

            delta = hit_f1s - bl_f1s
            d = cohens_d(hit_f1s, bl_f1s)

            if K >= 6:
                try:
                    stat, p = stats.wilcoxon(hit_f1s, bl_f1s, alternative="greater")
                    test_name = "Wilcoxon"
                except Exception:
                    stat, p = float("nan"), float("nan")
                    test_name = "Wilcoxon (failed)"
            else:
                stat, p = stats.ttest_rel(hit_f1s, bl_f1s)
                p = p / 2  # one-sided
                test_name = "Paired t-test"

            ci_lo = delta.mean() - 1.96 * delta.std(ddof=1) / np.sqrt(K)
            ci_hi = delta.mean() + 1.96 * delta.std(ddof=1) / np.sqrt(K)

            effect = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
            sig = "✓ (p<0.05)" if p < 0.05 else "✗ (ns)"

            stat_tests[bl] = {
                "test": test_name,
                "delta_mean": float(delta.mean()),
                "delta_ci": [float(ci_lo), float(ci_hi)],
                "cohens_d": float(d),
                "effect_size": effect,
                "p_value": float(p),
                "significant": bool(p < 0.05),
            }

            direction = "BETTER" if delta.mean() > 0 else "WORSE"
            print(f"\n  vs {bl}:")
            print(f"    ΔF1 = {delta.mean():+.4f} (HiT-IDS {direction})")
            print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
            print(f"    Cohen's d = {d:.3f} ({effect})")
            print(f"    {test_name}: p = {p:.4f} {sig}")

    # ── Save ──
    output = {
        "protocol": {
            "data_source": "raw CSVs → clean → REAL rows only (src/data/cv_data.py)",
            "n_real_rows": int(len(y_all)),
            "scaling": "RobustScaler per feature group, fit on training fold only",
            "resampling": "adaptive SMOTE + BENIGN cap 200K, training fold only",
            "test_fold": "real rows only, no synthetic samples",
            "phase2_entropy_finetuning": bool(PHASE2),
            "validation_for_early_stopping": (
                "10% of REAL training-fold rows, stratified, carved out before resampling"
                if args.val_source == "real" else "10% of the resampled training set (legacy)"),
            "smoke": bool(args.smoke),
            "epoch_cap": EPOCH_CAP,
        },
        "k_folds": K,
        "num_classes": NUM_MERGED,
        "class_names": MERGED_NAMES,
        "smote_target": args.smote_target,
        "folds": fold_info,
        "summary": summary,
        "statistical_tests": stat_tests,
        "per_fold_results": {
            model: results[model] for model in all_models
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {out_path}")
    print(f"  Per-fold predictions: {pred_dir}/")
    print(f"  Per-fold checkpoints: {ckpt_dir}/")

    # Final table
    print("\n" + "=" * 70)
    print("  FINAL TABLE (for paper)")
    print("=" * 70)
    print(f"\n  {'Model':<30s} {'Macro-F1':>12s} {'Accuracy':>10s} {'W-F1':>10s}")
    print("  " + "-" * 64)
    for model in all_models:
        s = summary.get(model)
        if s:
            f1_str = f"{s['mean_macro_f1']:.4f}±{s['std_macro_f1']:.4f}"
            print(f"  {model:<30s} {f1_str:>12s} "
                  f"{s['mean_accuracy']:>10.4f} {s['mean_weighted_f1']:>10.4f}")

    print("\n" + "=" * 70)
    print("  CV EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

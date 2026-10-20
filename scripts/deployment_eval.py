#!/usr/bin/env python3
"""
Deployment-model evaluation (single 70/15/15 split) with a validation-calibrated
Stage-1 threshold. This is the model every XAI / error-analysis section uses, so
its numbers must be stated once, cleanly:

    threshold t*  : chosen on data/splits/val.pt (real rows) by maximising
                    13-class cascade macro-F1
    reported      : data/splits/test.pt (real rows) at t* (and at 0.5 for reference)

Writes
    results/tables/deployment_eval.json
    results/tables/deployment_preds.npz
    results/figures/clean_rv_thr/threshold_sweep.{png,pdf}

Usage:
    python scripts/deployment_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.deployment import (load_config, load_deployment_models, load_split,
                                       cascade_probs, cascade_predict, combined_proba,
                                       BINARY_CKPT, ATTACK_CKPT, EVAL_JSON)
from scripts.evaluate_cv import fold_metrics, MERGED_NAMES


def sweep(p_bin, p_atk, y, grid):
    f1s, fprs = [], []
    for t in grid:
        pred = cascade_predict(p_bin, p_atk, t)
        f1s.append(f1_score(y, pred, average="macro", zero_division=0))
        fprs.append(((y == 0) & (pred > 0)).sum() / max((y == 0).sum(), 1))
    return np.array(f1s), np.array(fprs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--grid", type=int, default=199)
    ap.add_argument("--fig-dir", default="results/figures/clean_rv_thr")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    bin_m, atk_m, _ = load_deployment_models(config, device)

    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    pb_va, pa_va = cascade_probs(bin_m, atk_m, Xva, device)
    pb_te, pa_te = cascade_probs(bin_m, atk_m, Xte, device)

    grid = np.linspace(0.005, 0.995, args.grid)
    f_va, fpr_va = sweep(pb_va, pa_va, yva, grid)
    t_star = float(grid[int(np.argmax(f_va))])
    f_te, fpr_te = sweep(pb_te, pa_te, yte, grid)      # for the figure only

    res = {}
    for label, t in (("argmax_0.5", 0.5), ("calibrated", t_star)):
        pred = cascade_predict(pb_te, pa_te, t)
        res[label] = fold_metrics(yte, pred, combined_proba(pb_te, pa_te))
        res[label]["threshold"] = t
    out = {
        "model": {"binary": BINARY_CKPT, "attack": ATTACK_CKPT,
                  "train": "data/splits/train.pt (70%, SMOTE'd)",
                  "val": "data/splits/val.pt (15%, real)", "test": "data/splits/test.pt (15%, real)"},
        "stage1_threshold": t_star,
        "val_macro_f1_at_0.5": float(f_va[np.argmin(np.abs(grid - 0.5))]),
        "val_macro_f1_at_threshold": float(f_va.max()),
        "n_val": int(len(yva)), "n_test": int(len(yte)),
        "test": res,
        "sweep": {"grid": grid.tolist(), "val_macro_f1": f_va.tolist(), "val_fpr": fpr_va.tolist(),
                  "test_macro_f1": f_te.tolist(), "test_fpr": fpr_te.tolist()},
    }
    os.makedirs(os.path.dirname(EVAL_JSON), exist_ok=True)
    with open(EVAL_JSON, "w") as f:
        json.dump(out, f, indent=2)
    pred = cascade_predict(pb_te, pa_te, t_star)
    np.savez_compressed("results/tables/deployment_preds.npz", y_true=yte.astype(np.int16),
                        y_pred=pred.astype(np.int16), p_attack=pb_te[:, 1].astype(np.float32),
                        proba=combined_proba(pb_te, pa_te).astype(np.float16))

    # ── Figure: threshold sweep ───────────────────────────────────────────
    os.makedirs(args.fig_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(grid, f_va, label="macro-F1 (validation)", color="C0")
    ax1.plot(grid, f_te, label="macro-F1 (test)", color="C0", ls="--")
    ax1.axvline(t_star, color="k", lw=0.8, ls=":")
    ax1.axvline(0.5, color="grey", lw=0.8, ls=":")
    ax1.set_xlabel("Stage-1 threshold on p(attack)")
    ax1.set_ylabel("13-class macro-F1")
    ax2 = ax1.twinx()
    ax2.plot(grid, fpr_va * 100, label="FPR % (validation)", color="C3")
    ax2.plot(grid, fpr_te * 100, label="FPR % (test)", color="C3", ls="--")
    ax2.set_ylabel("false-positive rate (%)")
    ax2.set_yscale("log")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center")
    ax1.set_title(f"Stage-1 threshold sweep (t* = {t_star:.3f} chosen on validation)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.fig_dir, f"threshold_sweep.{ext}"), dpi=300, bbox_inches="tight")

    a, c = res["argmax_0.5"], res["calibrated"]
    print(f"\n  t* = {t_star:.3f}   val macro-F1 {out['val_macro_f1_at_0.5']:.4f} -> {out['val_macro_f1_at_threshold']:.4f}")
    print(f"  TEST macro-F1 {a['macro_f1']:.4f} -> {c['macro_f1']:.4f} | binary F1 {a['binary']['f1']:.4f} -> {c['binary']['f1']:.4f}"
          f" | FPR {a['binary']['fpr']*100:.2f}% -> {c['binary']['fpr']*100:.2f}%")
    for cl in ("Bot", "Infiltration", "Web Attack", "SSH-Patator"):
        print(f"    {cl:14s} F1 {a['per_class'][cl]['f1']:.3f} -> {c['per_class'][cl]['f1']:.3f}")
    print(f"  wrote {EVAL_JSON}")


if __name__ == "__main__":
    main()

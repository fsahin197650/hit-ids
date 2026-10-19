#!/usr/bin/env python3
"""
Unified explanation-latency benchmark (replaces the inconsistent 0.55 ms /
3.43 ms / 478 ms / 870× numbers, which came from three different protocols).

ONE protocol for every method:
    * same 100 attack rows from the real held-out test split
    * same 100-row SHAP background
    * two regimes, both reported:
        - single : batch size 1, loop over the N rows   (SOC per-alert latency)
        - batched: one call on all N rows               (amortised throughput)
    * 2 warm-up calls, then R timed repeats; median / min / max reported
    * CUDA synchronised before/after every timing window

Methods
    HiT-IDS  prediction only (forward, no explanation)   → baseline cost
    HiT-IDS  attention rollout (intrinsic)               → forward + rollout
    HiT-IDS  input×gradient (1 backward pass, post-hoc)
    HiT-IDS  GradientSHAP (post-hoc)
    RF       TreeSHAP (per-sample)
    XGBoost  TreeSHAP (per-sample)      [falls back to a note if shap fails]
    BiLSTM   GradientSHAP
    Vanilla Transformer GradientSHAP

Writes results/tables/explanation_latency_clean.{json,md}

Usage:
    python scripts/explanation_latency.py                 # full (N=100, R=5)
    python scripts/explanation_latency.py --n 4 --repeats 1   # quick check
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hit_ids import HiTIDS
from src.models.baselines import BiLSTMClassifier, VanillaTransformer
from src.xai.attention_rollout import AttentionRollout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "results/checkpoints"


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def timed(fn, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(min(ts)), float(max(ts))


def bench(name, kind, single_fn, batch_fn, n, repeats, results):
    """single_fn(i) explains row i; batch_fn() explains all n rows."""
    print(f"\n  [{name}]")
    row = {"type": kind}
    try:
        med, lo, hi = timed(lambda: [single_fn(i) for i in range(n)], repeats=repeats)
        row["single_ms_per_sample"] = {"median": med / n * 1e3, "min": lo / n * 1e3, "max": hi / n * 1e3}
        print(f"    single  : {med/n*1e3:9.3f} ms/sample  (min {lo/n*1e3:.3f}, max {hi/n*1e3:.3f})")
    except Exception as e:
        row["single_error"] = str(e)[:300]
        print(f"    single  : FAILED ({e})")
    try:
        med, lo, hi = timed(batch_fn, repeats=repeats)
        row["batched_ms_per_sample"] = {"median": med / n * 1e3, "min": lo / n * 1e3, "max": hi / n * 1e3}
        print(f"    batched : {med/n*1e3:9.3f} ms/sample  (min {lo/n*1e3:.3f}, max {hi/n*1e3:.3f})")
    except Exception as e:
        row["batched_error"] = str(e)[:300]
        print(f"    batched : FAILED ({e})")
    results[name] = row


class FlatWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x, return_explanations=False)


def load_hit_attack(config, path=None):
    p = path or os.path.join(CKPT, "hit_ids_attack_merged.pt")   # the deployment model
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    fg = config["feature_groups"]
    names = list(fg.keys())
    sizes = [len(fg[g]) for g in names]
    mc = config["model"]
    nc = ck["model_state_dict"]["classification_head.head.3.bias"].shape[0]
    m = HiTIDS(group_sizes=sizes, group_names=names, d_emb=mc["d_emb"], nhead=mc["nhead"],
               intra_layers=mc["intra_group_layers"], inter_layers=mc["inter_group_layers"],
               dim_feedforward=mc["dim_feedforward"], dropout=0.0, num_classes=nc)
    m.load_state_dict(ck["model_state_dict"])
    return m.to(DEVICE).eval(), names, p


def unwrap(obj):
    return getattr(obj, "model", obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/tables/explanation_latency_clean")
    ap.add_argument("--attack-ckpt", default=None, help="default: results/checkpoints/hit_ids_attack_merged.pt")
    args = ap.parse_args()
    n, R = args.n, args.repeats
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Data: real held-out test split (already scaled with train-fit scaler) ──
    test = torch.load("data/splits/test.pt", weights_only=False)
    X, y = test["X"].numpy(), test["y"].numpy()
    rng = np.random.RandomState(42)
    atk_idx = rng.choice(np.where(y > 0)[0], n, replace=False)
    bg_idx = rng.choice(len(y), 100, replace=False)
    Xs = torch.tensor(X[atk_idx], dtype=torch.float32).to(DEVICE)
    Xbg = torch.tensor(X[bg_idx], dtype=torch.float32).to(DEVICE)
    Xs_np, Xbg_np = X[atk_idx], X[bg_idx]

    results = {}
    import shap

    # ── HiT-IDS ──────────────────────────────────────────────────────────
    hit, gnames, hit_path = load_hit_attack(config, args.attack_ckpt)
    rollout = AttentionRollout(residual_weight=config["xai"]["rollout"].get("residual_weight", 0.5),
                               head_fusion=config["xai"]["rollout"].get("head_fusion", "mean"))

    @torch.no_grad()
    def hit_pred(x):
        return hit(x, return_explanations=False)

    @torch.no_grad()
    def hit_rollout(x):
        _, ex = hit(x, return_explanations=True)
        r = rollout.hierarchical_importance(ex["raw_intra_attn"], ex["raw_inter_attn"], gnames)
        return r["global_flat"]

    def hit_inputgrad(x):
        x = x.clone().requires_grad_(True)
        logits = hit(x, return_explanations=False)
        logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
        return x.grad.abs()

    bench("HiT-IDS prediction only (no explanation)", "reference",
          lambda i: hit_pred(Xs[i:i+1]), lambda: hit_pred(Xs), n, R, results)
    bench("HiT-IDS attention rollout", "intrinsic",
          lambda i: hit_rollout(Xs[i:i+1]), lambda: hit_rollout(Xs), n, R, results)
    bench("HiT-IDS input x gradient", "post-hoc (1 backward)",
          lambda i: hit_inputgrad(Xs[i:i+1]), lambda: hit_inputgrad(Xs), n, R, results)
    hit_exp = shap.GradientExplainer(FlatWrapper(hit).to(DEVICE).eval(), Xbg)
    bench("HiT-IDS GradientSHAP", "post-hoc",
          lambda i: hit_exp.shap_values(Xs[i:i+1]), lambda: hit_exp.shap_values(Xs), n, R, results)

    # ── Tree models ──────────────────────────────────────────────────────
    for name, fn in [("Random Forest TreeSHAP", "baseline_rf.pkl"), ("XGBoost TreeSHAP", "baseline_xgb.pkl")]:
        p = os.path.join(CKPT, fn)
        if not os.path.exists(p):
            results[name] = {"type": "post-hoc", "error": "checkpoint missing"}
            continue
        with open(p, "rb") as f:
            model = unwrap(pickle.load(f))
        try:
            if "XGBoost" in name:
                # shap.TreeExplainer cannot parse xgboost>=2 multi-class base_score;
                # use XGBoost's built-in TreeSHAP (pred_contribs) instead — same algorithm.
                import xgboost as xgb
                booster = model.get_booster()
                contrib = lambda A: booster.predict(xgb.DMatrix(A), pred_contribs=True)
                bench(name, "post-hoc", lambda i: contrib(Xs_np[i:i+1]),
                      lambda: contrib(Xs_np), n, R, results)
                results[name]["note"] = "xgboost native TreeSHAP (pred_contribs=True)"
            else:
                ex = shap.TreeExplainer(model)
                bench(name, "post-hoc", lambda i, ex=ex: ex.shap_values(Xs_np[i:i+1]),
                      lambda ex=ex: ex.shap_values(Xs_np), n, R, results)
        except Exception as e:
            results[name] = {"type": "post-hoc", "error": str(e)[:300]}
            print(f"\n  [{name}] FAILED: {e}")

    # ── BiLSTM ───────────────────────────────────────────────────────────
    p = os.path.join(CKPT, "baseline_bilstm.pt")
    if os.path.exists(p):
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        bilstm = BiLSTMClassifier(num_features=ck["num_features"], num_classes=ck["num_classes"])
        bilstm.load_state_dict(ck["model_state_dict"])
        bilstm.to(DEVICE)
        cudnn_was = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False   # shap needs non-cuDNN RNN backward
        bilstm.train()                          # cuDNN RNN backward requires train mode
        ex = shap.GradientExplainer(bilstm, Xbg)
        bench("BiLSTM GradientSHAP", "post-hoc",
              lambda i: ex.shap_values(Xs[i:i+1]), lambda: ex.shap_values(Xs), n, R, results)
        torch.backends.cudnn.enabled = cudnn_was

    # ── Vanilla Transformer ──────────────────────────────────────────────
    p = os.path.join(CKPT, "baseline_vt.pt")
    if os.path.exists(p):
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        vt = VanillaTransformer(num_features=ck["num_features"], num_classes=ck["num_classes"],
                                d_emb=ck["d_emb"], nhead=ck["nhead"], num_layers=ck["num_layers"],
                                dim_feedforward=ck["dim_feedforward"])
        vt.load_state_dict(ck["model_state_dict"])
        vt.to(DEVICE).eval()
        ex = shap.GradientExplainer(vt, Xbg)
        bench("Vanilla Transformer GradientSHAP", "post-hoc",
              lambda i: ex.shap_values(Xs[i:i+1]), lambda: ex.shap_values(Xs), n, R, results)

    # ── Derived speed-ups (attention rollout vs each post-hoc, same regime) ──
    att = results.get("HiT-IDS attention rollout", {})
    speed = {}
    for k, v in results.items():
        if v.get("type", "").startswith("post-hoc"):
            for reg in ("single_ms_per_sample", "batched_ms_per_sample"):
                if reg in v and reg in att and att[reg]["median"] > 0:
                    speed.setdefault(k, {})[reg.split("_")[0]] = v[reg]["median"] / att[reg]["median"]

    out = {
        "protocol": {"n_explain_samples": n, "n_background": 100, "repeats": R, "warmup": 2,
                     "statistic": "median over repeats", "data": "data/splits/test.pt attack rows (real)",
                     "hit_ids_checkpoint": hit_path},
        "hardware": {"device": DEVICE,
                     "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
                     "cpu": platform.processor() or platform.machine(),
                     "torch": torch.__version__, "shap": shap.__version__},
        "results_ms_per_sample": results,
        "speedup_of_attention_rollout_vs": speed,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=2)

    lines = ["| Method | Type | Single (batch=1) ms/sample | Batched (N=%d) ms/sample |" % n,
             "|---|---|---:|---:|"]
    for k, v in results.items():
        s = v.get("single_ms_per_sample", {}).get("median")
        b = v.get("batched_ms_per_sample", {}).get("median")
        lines.append(f"| {k} | {v.get('type','')} | {s:.3f} | {b:.3f} |" if s is not None and b is not None
                     else f"| {k} | {v.get('type','')} | {v.get('error', v.get('single_error','—'))} | |")
    lines.append("")
    lines.append("Speed-up of attention rollout (same regime): " +
                 "; ".join(f"{k}: {v.get('single', float('nan')):.1f}× single / {v.get('batched', float('nan')):.1f}× batched"
                           for k, v in speed.items()))
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n  wrote {args.out}.json / .md")


if __name__ == "__main__":
    main()

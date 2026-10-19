#!/usr/bin/env python3
"""
E5 — Efficiency table: parameter count + training time + peak memory.

Collects efficiency metrics for HiT-IDS and all baselines:
    - Trainable parameter count
    - Training time (from existing logs if available)
    - Peak GPU/CPU memory
    - Corrects the 870× speedup claim

Prerequisite artefacts:
    - results/checkpoints/*.pt, *.pkl
    - results/tables/throughput_results.json
    - (optional) results/logs/ for training time extraction

Output:
    results/tables/efficiency.json

Usage:
    python scripts/e5_efficiency.py
    python scripts/e5_efficiency.py --measure-memory
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common.helpers import (
    DEVICE, PROJECT_ROOT, SEED,
    CIC_CONFIG_PATH, CIC_BINARY_CKPT,
    UNSW_CONFIG_PATH, UNSW_BINARY_CKPT,
    load_model, load_config, get_group_info, set_seed, save_json, ensure_dir,
)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters (total, trainable, per-component)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    component_counts = {}
    if hasattr(model, "count_parameters"):
        component_counts = model.count_parameters()

    return {
        "total": total,
        "trainable": trainable,
        "component_breakdown": component_counts,
    }


def count_sklearn_params(model_path: str) -> int | None:
    """Estimate parameter count for sklearn models."""
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)

        if hasattr(clf, "estimators_"):
            # Random Forest: sum of nodes across all trees
            total = sum(
                est.tree_.node_count for est in clf.estimators_
            )
            return total
        elif hasattr(clf, "get_booster"):
            # XGBoost
            booster = clf.get_booster()
            try:
                trees = booster.get_dump()
                # Rough estimate: count lines (each ≈ one node)
                total = sum(len(t.strip().split("\n")) for t in trees)
                return total
            except Exception:
                return None
        return None
    except Exception:
        return None


def count_bilstm_params(ckpt_path: str) -> int | None:
    """Count BiLSTM parameters from checkpoint."""
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        total = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        return total
    except Exception:
        return None


def measure_peak_memory(model, X_sample, batch_size=256) -> dict | None:
    """Measure peak GPU memory during forward pass."""
    if not torch.cuda.is_available():
        return None

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval()
    dummy = X_sample[:batch_size].to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy, return_explanations=False)

    torch.cuda.reset_peak_memory_stats()

    # Measure
    with torch.no_grad():
        for _ in range(10):
            model(dummy, return_explanations=True)

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "peak_gpu_mb": round(peak_mb, 1),
        "batch_size": batch_size,
    }


def extract_training_time_from_logs(log_dir: str) -> dict:
    """Try to extract training times from log files."""
    times = {}
    if not os.path.isdir(log_dir):
        return times

    for fname in os.listdir(log_dir):
        fpath = os.path.join(log_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath) as f:
                content = f.read()
            # Look for common training time patterns
            for line in content.split("\n"):
                line_lower = line.lower()
                if "total training time" in line_lower or "elapsed" in line_lower:
                    # Extract numbers
                    import re
                    nums = re.findall(r"[\d.]+", line)
                    if nums:
                        times[fname] = {
                            "raw_line": line.strip(),
                            "extracted_seconds": float(nums[-1]) if len(nums) == 1 else None,
                        }
        except Exception:
            pass

    return times


def main():
    parser = argparse.ArgumentParser(
        description="E5: Efficiency table (params + training time + memory)"
    )
    parser.add_argument("--measure-memory", action="store_true",
                        help="Run live GPU memory measurement")
    parser.add_argument("--out", type=str,
                        default="results/tables/efficiency.json")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 70)
    print("  E5: EFFICIENCY TABLE")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    models_info = {}

    # ── HiT-IDS (CIC-IDS2017 Binary) ──
    print("\n  ── HiT-IDS (CIC-IDS2017 Binary) ──")
    model_cic = load_model(CIC_CONFIG_PATH, CIC_BINARY_CKPT, num_classes=2)
    params_cic = count_parameters(model_cic)
    models_info["HiT-IDS (Binary)"] = {
        "parameters": params_cic,
        "model_type": "Hierarchical Transformer",
    }
    print(f"    Params: {params_cic['trainable']:,}")

    # ── HiT-IDS Attack model ──
    print("\n  ── HiT-IDS (Attack) ──")
    for ckpt_name in [
        "results/checkpoints/hit_ids_attack_merged_v2.pt",
        "results/checkpoints/hit_ids_attack_merged.pt",
    ]:
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            nc = ckpt["model_state_dict"]["classification_head.head.3.bias"].shape[0]
            model_atk = load_model(CIC_CONFIG_PATH, ckpt_path, num_classes=nc)
            params_atk = count_parameters(model_atk)
            models_info["HiT-IDS (Attack)"] = {
                "parameters": params_atk,
                "model_type": "Hierarchical Transformer",
            }
            print(f"    Params: {params_atk['trainable']:,}")
            break

    # ── HiT-IDS (UNSW-NB15 Binary) ──
    unsw_ckpt = os.path.join(PROJECT_ROOT, UNSW_BINARY_CKPT)
    if os.path.exists(unsw_ckpt):
        print("\n  ── HiT-IDS (UNSW-NB15 Binary) ──")
        model_unsw = load_model(UNSW_CONFIG_PATH, unsw_ckpt, num_classes=2)
        params_unsw = count_parameters(model_unsw)
        models_info["HiT-IDS (UNSW Binary)"] = {
            "parameters": params_unsw,
            "model_type": "Hierarchical Transformer",
        }
        print(f"    Params: {params_unsw['trainable']:,}")

    # ── Vanilla Transformer ──
    vt_ckpt = os.path.join(PROJECT_ROOT, "results/checkpoints/baseline_vt.pt")
    if os.path.exists(vt_ckpt):
        print("\n  ── Vanilla Transformer ──")
        n_params_vt = count_bilstm_params(vt_ckpt)
        models_info["Vanilla Transformer"] = {
            "parameters": {"trainable": n_params_vt},
            "model_type": "Standard Transformer",
        }
        if n_params_vt:
            print(f"    Params: {n_params_vt:,}")

    # ── BiLSTM ──
    bilstm_ckpt = os.path.join(PROJECT_ROOT, "results/checkpoints/baseline_bilstm.pt")
    if os.path.exists(bilstm_ckpt):
        print("\n  ── BiLSTM ──")
        n_params_bilstm = count_bilstm_params(bilstm_ckpt)
        models_info["BiLSTM"] = {
            "parameters": {"trainable": n_params_bilstm},
            "model_type": "Bidirectional LSTM",
        }
        if n_params_bilstm:
            print(f"    Params: {n_params_bilstm:,}")

    # ── Random Forest ──
    rf_path = os.path.join(PROJECT_ROOT, "results/checkpoints/baseline_rf.pkl")
    if os.path.exists(rf_path):
        print("\n  ── Random Forest ──")
        n_nodes = count_sklearn_params(rf_path)
        file_size_mb = os.path.getsize(rf_path) / (1024 ** 2)
        models_info["Random Forest"] = {
            "parameters": {"tree_nodes": n_nodes},
            "model_type": "Ensemble (100 trees)",
            "file_size_mb": round(file_size_mb, 1),
        }
        if n_nodes:
            print(f"    Nodes: {n_nodes:,}  |  File: {file_size_mb:.1f} MB")

    # ── XGBoost ──
    xgb_path = os.path.join(PROJECT_ROOT, "results/checkpoints/baseline_xgb.pkl")
    if os.path.exists(xgb_path):
        print("\n  ── XGBoost ──")
        n_nodes = count_sklearn_params(xgb_path)
        file_size_mb = os.path.getsize(xgb_path) / (1024 ** 2)
        models_info["XGBoost"] = {
            "parameters": {"tree_nodes": n_nodes},
            "model_type": "Gradient Boosting",
            "file_size_mb": round(file_size_mb, 1),
        }
        if n_nodes:
            print(f"    Nodes: {n_nodes:,}  |  File: {file_size_mb:.1f} MB")

    # ── GPU Memory measurement ──
    if args.measure_memory and torch.cuda.is_available():
        print("\n  ── Memory Measurement ──")
        test_data = torch.load(
            os.path.join(PROJECT_ROOT, "data/splits/test.pt"), weights_only=False
        )
        X_sample = test_data["X"]

        mem = measure_peak_memory(model_cic, X_sample)
        if mem:
            models_info["HiT-IDS (Binary)"]["memory"] = mem
            print(f"    HiT-IDS peak GPU: {mem['peak_gpu_mb']:.1f} MB (batch={mem['batch_size']})")

    # ── Training time from logs ──
    print("\n  ── Training Time Extraction ──")
    log_dirs = [
        os.path.join(PROJECT_ROOT, "results/logs"),
        os.path.join(PROJECT_ROOT, "results/unsw_nb15/logs"),
    ]
    for log_dir in log_dirs:
        times = extract_training_time_from_logs(log_dir)
        if times:
            for fname, info in times.items():
                print(f"    {fname}: {info.get('raw_line', 'N/A')}")

    # ── Load existing throughput data ──
    throughput_path = os.path.join(PROJECT_ROOT, "results/tables/throughput_results.json")
    speedup_correction = {}
    if os.path.exists(throughput_path):
        with open(throughput_path) as f:
            tp = json.load(f)
        costs = tp.get("explanation_costs", {})
        speedup_correction = {
            "attention_ms": costs.get("attention_ms_per_sample"),
            "gradient_shap_ms": costs.get("gradient_shap_ms_per_sample"),
            "tree_shap_ms": costs.get("tree_shap_rf_ms_per_sample"),
            "speedup_vs_gradient_shap": costs.get("speedup_vs_gradient_shap"),
            "speedup_vs_tree_shap": costs.get("speedup_vs_tree_shap"),
            "CORRECTED_claim": (
                f"Attention rollout is {costs.get('speedup_vs_gradient_shap', '?')}× "
                f"faster than GradientSHAP and {costs.get('speedup_vs_tree_shap', '?')}× "
                f"faster than TreeSHAP"
            ),
            "OLD_WRONG_claim": "≈870× speedup — THIS IS INCORRECT",
        }
        print(f"\n  Speedup correction:")
        print(f"    OLD:  ≈870×")
        print(f"    NEW:  {costs.get('speedup_vs_gradient_shap', '?')}× vs GradientSHAP, "
              f"{costs.get('speedup_vs_tree_shap', '?')}× vs TreeSHAP")

    # ── Summary table ──
    print(f"\n{'═' * 70}")
    print("  EFFICIENCY SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Model':30s} {'Params':>12s} {'Type':20s}")
    print(f"  {'─' * 65}")
    for name, info in models_info.items():
        p = info["parameters"]
        if "trainable" in p and p["trainable"]:
            pstr = f"{p['trainable']:>12,}"
        elif "tree_nodes" in p and p["tree_nodes"]:
            pstr = f"{p['tree_nodes']:>10,} n"
        else:
            pstr = "      N/A"
        print(f"  {name:30s} {pstr:>12s} {info['model_type']:20s}")

    # ── Save ──
    output = {
        "models": models_info,
        "speedup_correction": speedup_correction,
        "hardware": {
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
    }

    out_path = os.path.join(PROJECT_ROOT, args.out)
    save_json(output, out_path)

    print(f"\n{'═' * 70}")
    print(f"  E5 COMPLETE")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Throughput & Latency Measurement — Q1 Revision (S1/S3).

Measures:
    - Single-sample latency (ms)
    - Batch throughput (flows/sec) at various batch sizes
    - Attention extraction overhead
    - TreeSHAP vs GradientSHAP vs Attention timing
    - Hardware info

Output:
    results/tables/throughput_results.json

Usage:
    python scripts/measure_throughput.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hit_ids import HiTIDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = "configs/config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

fg_cfg = config["feature_groups"]
GROUP_NAMES = list(fg_cfg.keys())
GROUP_SIZES = [len(fg_cfg[n]) for n in GROUP_NAMES]
TOTAL_FEATURES = sum(GROUP_SIZES)
model_cfg = config["model"]


def get_hardware_info():
    """Collect hardware information."""
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        info["cuda_version"] = torch.version.cuda

    # Try lscpu
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n"):
            if "Model name" in line:
                info["cpu"] = line.split(":")[1].strip()
                break
    except Exception:
        pass

    return info


def load_models():
    """Load binary + attack cascade models."""
    binary_ckpt_path = "results/checkpoints/hit_ids_binary.pt"
    attack_candidates = [
        "results/checkpoints/hit_ids_attack_merged_v2.pt",
        "results/checkpoints/hit_ids_attack_merged.pt",
        "results/checkpoints/hit_ids_attack.pt",
    ]

    # Binary
    bin_ckpt = torch.load(binary_ckpt_path, map_location=DEVICE, weights_only=False)
    bin_model = HiTIDS(
        group_sizes=GROUP_SIZES, group_names=GROUP_NAMES,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=0.0, num_classes=2,
    )
    bin_model.load_state_dict(bin_ckpt["model_state_dict"])
    bin_model.to(DEVICE).eval()

    # Attack
    atk_ckpt_path = None
    for c in attack_candidates:
        if os.path.exists(c):
            atk_ckpt_path = c
            break
    atk_ckpt = torch.load(atk_ckpt_path, map_location=DEVICE, weights_only=False)
    num_atk = atk_ckpt["model_state_dict"]["classification_head.head.3.bias"].shape[0]

    atk_model = HiTIDS(
        group_sizes=GROUP_SIZES, group_names=GROUP_NAMES,
        d_emb=model_cfg["d_emb"], nhead=model_cfg["nhead"],
        intra_layers=model_cfg["intra_group_layers"],
        inter_layers=model_cfg["inter_group_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=0.0, num_classes=num_atk,
    )
    atk_model.load_state_dict(atk_ckpt["model_state_dict"])
    atk_model.to(DEVICE).eval()

    return bin_model, atk_model


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_single_latency(bin_model, atk_model, X_sample, n_warmup=100, n_iters=1000):
    """Single-sample latency for cascade (binary + attack if positive)."""
    x = X_sample.unsqueeze(0).to(DEVICE)

    # Warmup
    for _ in range(n_warmup):
        bin_model(x, return_explanations=False)
    sync()

    # Binary-only latency
    sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        bp = bin_model(x, return_explanations=False).argmax(1)
    sync()
    bin_latency = (time.perf_counter() - t0) / n_iters * 1000  # ms

    # Cascade latency (binary + attack)
    sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        bp = bin_model(x, return_explanations=False).argmax(1)
        if bp.item() == 1:
            atk_model(x, return_explanations=False)
    sync()
    cascade_latency = (time.perf_counter() - t0) / n_iters * 1000

    # Cascade + attention latency
    sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        bp = bin_model(x, return_explanations=True)
        if bp[0].argmax(1).item() == 1:
            atk_model(x, return_explanations=True)
    sync()
    cascade_attn_latency = (time.perf_counter() - t0) / n_iters * 1000

    return {
        "binary_only_ms": round(bin_latency, 3),
        "cascade_ms": round(cascade_latency, 3),
        "cascade_with_attention_ms": round(cascade_attn_latency, 3),
        "attention_overhead_ms": round(cascade_attn_latency - cascade_latency, 3),
    }


def measure_batch_throughput(bin_model, atk_model, batch_sizes, n_warmup=20, n_iters=100):
    """Batch throughput in flows/sec for various batch sizes."""
    results = {}
    for bs in batch_sizes:
        dummy = torch.randn(bs, TOTAL_FEATURES).to(DEVICE)

        # Warmup
        for _ in range(n_warmup):
            bin_model(dummy, return_explanations=False)
        sync()

        # Classification only
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            bp = bin_model(dummy, return_explanations=False).argmax(1)
            mask = bp == 1
            if mask.any():
                atk_model(dummy[mask], return_explanations=False)
        sync()
        elapsed = time.perf_counter() - t0
        throughput_cls = (bs * n_iters) / elapsed

        # Classification + attention
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            logits, _ = bin_model(dummy, return_explanations=True)
            bp = logits.argmax(1)
            mask = bp == 1
            if mask.any():
                atk_model(dummy[mask], return_explanations=True)
        sync()
        elapsed_attn = time.perf_counter() - t0
        throughput_attn = (bs * n_iters) / elapsed_attn

        results[f"batch_{bs}"] = {
            "batch_size": bs,
            "throughput_cls_flows_sec": round(throughput_cls, 0),
            "throughput_cls_attn_flows_sec": round(throughput_attn, 0),
            "latency_cls_ms": round(1000 / throughput_cls * bs, 3),
        }
        print(f"    Batch {bs:>4d}: {throughput_cls:>8.0f} flows/s (cls)  |  {throughput_attn:>8.0f} flows/s (cls+attn)")

    return results


def measure_shap_times(bin_model, X_samples):
    """Measure GradientSHAP and TreeSHAP explanation times."""
    results = {}

    # GradientSHAP on HiT-IDS
    try:
        import shap
        x_bg = X_samples[:50].to(DEVICE)
        x_test = X_samples[:10].to(DEVICE)

        # Wrap model for SHAP compatibility (needs nn.Module, not function)
        class SHAPWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x, return_explanations=False)

        wrapper = SHAPWrapper(bin_model).to(DEVICE).eval()
        explainer = shap.GradientExplainer(wrapper, x_bg)

        # Warmup
        _ = explainer.shap_values(x_test[:1])

        sync()
        t0 = time.perf_counter()
        n_samples_shap = min(50, len(X_samples))
        for i in range(n_samples_shap):
            _ = explainer.shap_values(x_test[i:i+1])
        sync()
        gradient_shap_ms = (time.perf_counter() - t0) / n_samples_shap * 1000
        results["gradient_shap_ms_per_sample"] = round(gradient_shap_ms, 2)
        print(f"    GradientSHAP: {gradient_shap_ms:.2f} ms/sample")
    except Exception as e:
        print(f"    GradientSHAP: SKIPPED ({e})")
        results["gradient_shap_ms_per_sample"] = None

    # TreeSHAP (Random Forest)
    try:
        import shap
        import pickle
        rf_path = "results/checkpoints/baseline_rf.pkl"
        if os.path.exists(rf_path):
            with open(rf_path, "rb") as f:
                rf_model = pickle.load(f)

            X_np = X_samples[:100].numpy()
            explainer_tree = shap.TreeExplainer(rf_model)

            # Warmup
            _ = explainer_tree.shap_values(X_np[:1])

            t0 = time.perf_counter()
            n_tree = min(100, len(X_np))
            _ = explainer_tree.shap_values(X_np[:n_tree])
            tree_shap_ms = (time.perf_counter() - t0) / n_tree * 1000
            results["tree_shap_rf_ms_per_sample"] = round(tree_shap_ms, 2)
            print(f"    TreeSHAP (RF): {tree_shap_ms:.2f} ms/sample")
        else:
            print(f"    TreeSHAP (RF): SKIPPED (no checkpoint)")
            results["tree_shap_rf_ms_per_sample"] = None
    except Exception as e:
        print(f"    TreeSHAP (RF): SKIPPED ({e})")
        results["tree_shap_rf_ms_per_sample"] = None

    # HiT-IDS attention time (for comparison)
    x_single = X_samples[0:1].to(DEVICE)
    sync()
    t0 = time.perf_counter()
    n_attn = 1000
    for _ in range(n_attn):
        bin_model(x_single, return_explanations=True)
    sync()
    attn_ms = (time.perf_counter() - t0) / n_attn * 1000
    results["attention_ms_per_sample"] = round(attn_ms, 3)
    print(f"    Attention rollout: {attn_ms:.3f} ms/sample")

    # Speedup ratio
    if results.get("gradient_shap_ms_per_sample"):
        results["speedup_vs_gradient_shap"] = round(
            results["gradient_shap_ms_per_sample"] / max(attn_ms, 0.001), 1)
    if results.get("tree_shap_rf_ms_per_sample"):
        results["speedup_vs_tree_shap"] = round(
            results["tree_shap_rf_ms_per_sample"] / max(attn_ms, 0.001), 1)

    return results


def main():
    print("=" * 70)
    print("  THROUGHPUT & LATENCY MEASUREMENT")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    os.makedirs("results/tables", exist_ok=True)

    # Hardware
    print("\n  Hardware:")
    hw = get_hardware_info()
    for k, v in hw.items():
        print(f"    {k}: {v}")

    # Load models
    print("\n  Loading models...")
    bin_model, atk_model = load_models()

    # Load sample data
    test_data = torch.load("data/splits/test.pt", weights_only=False)
    X_test = test_data["X"]

    # Single-sample latency
    print("\n  Single-sample latency (1000 iterations):")
    single = measure_single_latency(bin_model, atk_model, X_test[0])
    for k, v in single.items():
        print(f"    {k}: {v} ms")

    # Batch throughput
    print("\n  Batch throughput:")
    batch_sizes = [1, 32, 64, 128, 256, 512]
    batch_results = measure_batch_throughput(bin_model, atk_model, batch_sizes)

    # SHAP comparison
    print("\n  Explanation cost comparison:")
    shap_results = measure_shap_times(bin_model, X_test[:200])

    # Save
    output = {
        "hardware": hw,
        "single_sample_latency": single,
        "batch_throughput": batch_results,
        "explanation_costs": shap_results,
    }

    out_path = "results/tables/throughput_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Results saved: {out_path}")

    # Summary
    print(f"\n{'═' * 70}")
    print("  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Single-sample cascade latency: {single['cascade_ms']} ms")
    print(f"  Attention overhead:            {single['attention_overhead_ms']} ms")
    b128 = batch_results.get("batch_128", {})
    if b128:
        print(f"  Batch-128 throughput (cls):     {b128['throughput_cls_flows_sec']:.0f} flows/sec")
        print(f"  Batch-128 throughput (cls+attn):{b128['throughput_cls_attn_flows_sec']:.0f} flows/sec")
    if shap_results.get("speedup_vs_gradient_shap"):
        print(f"  Speedup vs GradientSHAP:       {shap_results['speedup_vs_gradient_shap']}×")
    if shap_results.get("speedup_vs_tree_shap"):
        print(f"  Speedup vs TreeSHAP:           {shap_results['speedup_vs_tree_shap']}×")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

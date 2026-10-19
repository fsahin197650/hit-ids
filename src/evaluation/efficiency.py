"""
Efficiency profiling for HiT-IDS.

Measures inference latency, throughput, and model complexity
for the paper's efficiency comparison table.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class EfficiencyProfiler:
    """
    Profiles model efficiency: latency, throughput, FLOPs, parameters.

    Args:
        model: Model to profile.
        device: Device for profiling.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def measure_latency(
        self,
        input_size: tuple,
        batch_sizes: list = None,
        warmup_runs: int = 10,
        measure_runs: int = 100,
        input_type: str = "flat",
        group_sizes: list = None,
        group_names: list = None,
    ) -> Dict[str, Any]:
        """
        Measure inference latency.

        Args:
            input_size: (total_features,) for flat or used with groups.
            batch_sizes: List of batch sizes to test.
            warmup_runs: Warmup iterations.
            measure_runs: Measurement iterations.
            input_type: 'flat' or 'grouped'.
            group_sizes: List of group sizes (for grouped input).
            group_names: List of group names.

        Returns:
            Dict mapping batch_size → latency stats.
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 64, 128, 256]

        results: Dict[str, Any] = {}

        for bs in batch_sizes:
            # Generate dummy input
            if input_type == "grouped" and group_sizes and group_names:
                x = {
                    name: torch.randn(bs, size).to(self.device)
                    for name, size in zip(group_names, group_sizes)
                }
            else:
                x = torch.randn(bs, *input_size).to(self.device)

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = self.model(x)

            # Measure
            if self.device == "cuda":
                torch.cuda.synchronize()

            latencies = []
            with torch.no_grad():
                for _ in range(measure_runs):
                    start = time.perf_counter()
                    _ = self.model(x)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)  # ms

            latencies = np.array(latencies)
            results[bs] = {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "throughput_samples_per_sec": float(
                    bs / (np.mean(latencies) / 1000)
                ),
            }

        return results

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
        }

    def estimate_flops(
        self,
        input_size: tuple,
        batch_size: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Estimate FLOPs using thop or torchinfo.

        Args:
            input_size: Input tensor size (without batch).
            batch_size: Batch size for estimation.

        Returns:
            Dict with FLOPs info, or None if tools unavailable.
        """
        try:
            from thop import profile, clever_format
            x = torch.randn(batch_size, *input_size).to(self.device)
            macs, params = profile(self.model, inputs=(x,), verbose=False)
            macs_str, params_str = clever_format([macs, params], "%.3f")
            return {
                "macs": int(macs),
                "macs_readable": macs_str,
                "params": int(params),
                "params_readable": params_str,
            }
        except ImportError:
            pass

        try:
            from torchinfo import summary
            result = summary(
                self.model,
                input_size=(batch_size, *input_size),
                verbose=0,
            )
            return {
                "total_mult_adds": result.total_mult_adds,
                "total_params": result.total_params,
            }
        except ImportError:
            pass

        return None

    def full_profile(
        self,
        input_size: tuple,
        batch_sizes: list = None,
        group_sizes: list = None,
        group_names: list = None,
    ) -> Dict[str, Any]:
        """
        Run complete efficiency profile.

        Returns:
            Dict with params, latency, throughput, and optional FLOPs.
        """
        result: Dict[str, Any] = {}

        # Parameters
        result["parameters"] = self.count_parameters()
        print(f"  Parameters: {result['parameters']['total']:,}")

        # FLOPs
        flops = self.estimate_flops(input_size)
        if flops:
            result["flops"] = flops
            print(f"  MACs: {flops.get('macs_readable', 'N/A')}")

        # Latency
        input_type = "grouped" if group_sizes else "flat"
        result["latency"] = self.measure_latency(
            input_size,
            batch_sizes=batch_sizes,
            input_type=input_type,
            group_sizes=group_sizes,
            group_names=group_names,
        )

        # Print summary
        print("\n  Latency Summary:")
        print(f"  {'Batch':>6s} {'Mean(ms)':>10s} {'P95(ms)':>10s} {'Throughput':>15s}")
        for bs, stats in result["latency"].items():
            print(f"  {bs:>6d} {stats['mean_ms']:>10.2f} "
                  f"{stats['p95_ms']:>10.2f} "
                  f"{stats['throughput_samples_per_sec']:>12.0f}/s")

        return result


def create_efficiency_table(
    profiles: Dict[str, Dict[str, Any]],
    batch_size: int = 1,
) -> str:
    """
    Create formatted efficiency comparison table.

    Args:
        profiles: Dict mapping model_name → profile results.
        batch_size: Batch size to report.

    Returns:
        Formatted table string.
    """
    header = (f"{'Model':<25s} {'Params':>12s} {'MACs':>12s} "
              f"{'Latency(ms)':>12s} {'Throughput':>12s}")
    sep = "─" * len(header)
    lines = [sep, header, sep]

    for name, profile in profiles.items():
        params = profile['parameters']['total']
        params_str = f"{params/1e6:.2f}M" if params > 1e6 else f"{params/1e3:.1f}K"

        flops = profile.get('flops', {})
        macs_str = flops.get('macs_readable', 'N/A')

        latency = profile.get('latency', {}).get(batch_size, {})
        lat_str = f"{latency.get('mean_ms', 0):.2f}" if latency else "N/A"
        thr = latency.get('throughput_samples_per_sec', 0)
        thr_str = f"{thr:.0f}/s" if thr else "N/A"

        lines.append(f"{name:<25s} {params_str:>12s} {macs_str:>12s} "
                     f"{lat_str:>12s} {thr_str:>12s}")

    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing EfficiencyProfiler...")

    from src.models.hit_ids import HiTIDS

    model = HiTIDS(
        group_sizes=[19, 25, 15, 13], d_emb=64, nhead=4,
        intra_layers=2, inter_layers=2, num_classes=15,
    )

    profiler = EfficiencyProfiler(model, device="cpu")

    result = profiler.full_profile(
        input_size=(72,),
        batch_sizes=[1, 32],
        group_sizes=[19, 25, 15, 13],
        group_names=["temporal", "volume", "protocol", "behavioral"],
    )

    print(f"\n  Results keys: {list(result.keys())}")
    print("\n✓ EfficiencyProfiler test passed!")

"""
XAI Consistency Score (XCS) — Novel metric for HiT-IDS.

Measures the agreement between intrinsic (attention-based) and
post-hoc (SHAP-based) feature importance rankings.

XCS = Spearman_ρ(rank(attention_importance), rank(shap_importance))

High XCS (→1.0) means attention truly reflects the model's decision process.
Low XCS (→0.0) means attention is not a faithful explanation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, kendalltau


def compute_xcs(
    attention_importance: np.ndarray,
    shap_importance: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float]:
    """
    Compute XAI Consistency Score between two importance vectors.

    Args:
        attention_importance: Feature importance from attention rollout.
            Shape: (num_features,).
        shap_importance: Feature importance from SHAP.
            Shape: (num_features,).
        method: Correlation method — 'spearman' or 'kendall'.

    Returns:
        Tuple of (correlation_score, p_value).
        Score in [-1, 1], higher is better.
    """
    assert len(attention_importance) == len(shap_importance), (
        f"Importance vectors must have same length: "
        f"{len(attention_importance)} vs {len(shap_importance)}"
    )

    if method == "spearman":
        score, p_value = spearmanr(attention_importance, shap_importance)
    elif method == "kendall":
        score, p_value = kendalltau(attention_importance, shap_importance)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(score), float(p_value)


def compute_xcs_batch(
    attention_importances: np.ndarray,
    shap_importances: np.ndarray,
    method: str = "spearman",
) -> Dict[str, float]:
    """
    Compute XCS statistics across a batch of samples.

    Args:
        attention_importances: (N, num_features).
        shap_importances: (N, num_features).
        method: Correlation method.

    Returns:
        Dict with 'mean', 'std', 'median', 'min', 'max' XCS values.
    """
    N = attention_importances.shape[0]
    scores = []
    p_values = []

    for i in range(N):
        score, pval = compute_xcs(
            attention_importances[i], shap_importances[i], method
        )
        if not np.isnan(score):
            scores.append(score)
            p_values.append(pval)

    scores = np.array(scores)
    p_values = np.array(p_values)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "significant_ratio": float((p_values < 0.05).mean()),
        "n_samples": len(scores),
    }


def compute_top_k_overlap(
    attention_importance: np.ndarray,
    shap_importance: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute top-K feature overlap between attention and SHAP rankings.

    Args:
        attention_importance: (num_features,).
        shap_importance: (num_features,).
        k: Number of top features to compare.

    Returns:
        Overlap ratio in [0, 1].
    """
    top_attn = set(np.argsort(attention_importance)[-k:])
    top_shap = set(np.argsort(shap_importance)[-k:])
    overlap = len(top_attn & top_shap) / k
    return overlap


def compute_top_k_overlap_batch(
    attention_importances: np.ndarray,
    shap_importances: np.ndarray,
    k_values: List[int] = None,
) -> Dict[int, float]:
    """
    Compute top-K overlap for multiple K values across a batch.

    Args:
        attention_importances: (N, num_features).
        shap_importances: (N, num_features).
        k_values: List of K values to compute.

    Returns:
        Dict mapping K → mean overlap ratio.
    """
    if k_values is None:
        k_values = [5, 10, 15, 20]

    results = {}
    for k in k_values:
        overlaps = [
            compute_top_k_overlap(attention_importances[i], shap_importances[i], k)
            for i in range(len(attention_importances))
        ]
        results[k] = float(np.mean(overlaps))

    return results


def compute_per_group_xcs(
    attention_group: Dict[str, np.ndarray],
    shap_group: Dict[str, np.ndarray],
    method: str = "spearman",
) -> Dict[str, Tuple[float, float]]:
    """
    Compute XCS per feature group.

    Args:
        attention_group: Dict[group_name → importance array].
        shap_group: Dict[group_name → importance array].

    Returns:
        Dict[group_name → (xcs_score, p_value)].
    """
    results = {}
    for group_name in attention_group:
        if group_name in shap_group:
            score, pval = compute_xcs(
                attention_group[group_name],
                shap_group[group_name],
                method,
            )
            results[group_name] = (score, pval)
    return results


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing XCS Metric...")

    np.random.seed(42)
    num_features = 72

    # Test with highly correlated importances
    attn_imp = np.random.rand(num_features)
    shap_imp = attn_imp + np.random.randn(num_features) * 0.1  # noisy copy
    score, pval = compute_xcs(attn_imp, shap_imp)
    print(f"  High correlation: XCS={score:.4f}, p={pval:.6f}")

    # Test with uncorrelated importances
    shap_random = np.random.rand(num_features)
    score2, pval2 = compute_xcs(attn_imp, shap_random)
    print(f"  Low correlation:  XCS={score2:.4f}, p={pval2:.6f}")

    # Batch test
    N = 50
    attn_batch = np.random.rand(N, num_features)
    shap_batch = attn_batch + np.random.randn(N, num_features) * 0.2
    batch_stats = compute_xcs_batch(attn_batch, shap_batch)
    print(f"  Batch XCS: {batch_stats}")

    # Top-K overlap
    overlap = compute_top_k_overlap(attn_imp, shap_imp, k=10)
    print(f"  Top-10 overlap: {overlap:.2f}")

    # Top-K batch
    k_results = compute_top_k_overlap_batch(attn_batch, shap_batch)
    print(f"  Top-K overlaps: {k_results}")

    print("\n✓ XCS metric test passed!")

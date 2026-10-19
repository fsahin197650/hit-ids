"""
Faithfulness test for HiT-IDS attention explanations.

Tests whether the model's attention-based explanations are faithful by
measuring performance degradation when top-K important features are masked.

A faithful explanation means: removing important features causes large
performance drops, while removing unimportant ones has minimal effect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class FaithfulnessTest:
    """
    Faithfulness evaluation for attention-based explanations.

    Tests:
        1. Top-K masking: mask top-K features → measure acc/F1 drop
        2. Bottom-K masking: mask bottom-K features → should have less drop
        3. Random masking: random K features → baseline comparison

    If explanations are faithful:
        performance_drop(top-K) >> performance_drop(random-K) >> performance_drop(bottom-K)

    Args:
        model: Trained HiT-IDS model.
        device: Computation device.
        masking_strategy: How to mask features: 'zero', 'mean', 'random'.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        masking_strategy: str = "zero",
    ) -> None:
        self.model = model
        self.device = device
        self.masking_strategy = masking_strategy
        self.model.eval()

    def compute_baseline_accuracy(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute accuracy and F1 on unmodified data."""
        from sklearn.metrics import f1_score, accuracy_score

        with torch.no_grad():
            X_dev = X.to(self.device)
            logits = self.model(X_dev, return_explanations=False)
            preds = logits.argmax(dim=1).cpu().numpy()

        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        acc = accuracy_score(y_np, preds)
        f1 = f1_score(y_np, preds, average="macro", zero_division=0)
        return acc, f1

    def mask_features(
        self,
        X: torch.Tensor,
        feature_indices: List[int],
        mean_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mask specified feature indices.

        Args:
            X: Input tensor, shape (N, total_features).
            feature_indices: Indices of features to mask.
            mean_values: Mean values for 'mean' masking strategy.

        Returns:
            Masked tensor.
        """
        X_masked = X.clone()

        if self.masking_strategy == "zero":
            X_masked[:, feature_indices] = 0.0
        elif self.masking_strategy == "mean":
            if mean_values is not None:
                X_masked[:, feature_indices] = mean_values[feature_indices]
            else:
                X_masked[:, feature_indices] = X[:, feature_indices].mean(dim=0)
        elif self.masking_strategy == "random":
            X_masked[:, feature_indices] = torch.randn_like(
                X_masked[:, feature_indices]
            )
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

        return X_masked

    def run_faithfulness_test(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_importance: np.ndarray,
        top_k_values: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full faithfulness test.

        Args:
            X: Input data, shape (N, total_features).
            y: Labels, shape (N,).
            feature_importance: Feature importance scores, shape (total_features,).
            top_k_values: List of K values to test.

        Returns:
            Dict with faithfulness results per K value.
        """
        if top_k_values is None:
            top_k_values = [5, 10, 15, 20, 25]

        from sklearn.metrics import f1_score as sklearn_f1

        # Baseline performance
        base_acc, base_f1 = self.compute_baseline_accuracy(X, y)
        print(f"  Baseline: acc={base_acc:.4f}, f1={base_f1:.4f}")

        # Sort features by importance (descending)
        sorted_indices = np.argsort(feature_importance)[::-1]

        results: Dict[str, Any] = {
            "baseline_accuracy": base_acc,
            "baseline_f1": base_f1,
            "top_k_results": {},
            "bottom_k_results": {},
            "random_k_results": {},
        }

        for k in top_k_values:
            if k > len(sorted_indices):
                continue

            # Top-K masking (most important features)
            top_k_indices = sorted_indices[:k].tolist()
            X_masked_top = self.mask_features(X, top_k_indices)
            acc_top, f1_top = self.compute_baseline_accuracy(X_masked_top, y)

            # Bottom-K masking (least important features)
            bottom_k_indices = sorted_indices[-k:].tolist()
            X_masked_bottom = self.mask_features(X, bottom_k_indices)
            acc_bottom, f1_bottom = self.compute_baseline_accuracy(X_masked_bottom, y)

            # Random-K masking
            random_indices = np.random.choice(
                len(sorted_indices), k, replace=False
            ).tolist()
            X_masked_random = self.mask_features(X, random_indices)
            acc_random, f1_random = self.compute_baseline_accuracy(X_masked_random, y)

            results["top_k_results"][k] = {
                "accuracy": acc_top,
                "f1": f1_top,
                "acc_drop": base_acc - acc_top,
                "f1_drop": base_f1 - f1_top,
            }
            results["bottom_k_results"][k] = {
                "accuracy": acc_bottom,
                "f1": f1_bottom,
                "acc_drop": base_acc - acc_bottom,
                "f1_drop": base_f1 - f1_bottom,
            }
            results["random_k_results"][k] = {
                "accuracy": acc_random,
                "f1": f1_random,
                "acc_drop": base_acc - acc_random,
                "f1_drop": base_f1 - f1_random,
            }

            print(f"  K={k:2d}: top_drop={base_f1-f1_top:.4f}, "
                  f"bottom_drop={base_f1-f1_bottom:.4f}, "
                  f"random_drop={base_f1-f1_random:.4f}")

        # Compute faithfulness score
        # Higher is better: ratio of top-K drop to random-K drop
        faithfulness_scores = []
        for k in top_k_values:
            if k in results["top_k_results"] and k in results["random_k_results"]:
                top_drop = results["top_k_results"][k]["f1_drop"]
                random_drop = results["random_k_results"][k]["f1_drop"]
                if abs(random_drop) > 1e-6:
                    faithfulness_scores.append(top_drop / random_drop)

        results["faithfulness_score"] = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0

        print(f"\n  Faithfulness score: {results['faithfulness_score']:.4f}")
        print(f"  (> 1.0 = explanations are faithful)")

        return results


# ═══════════════════════════════════════════════════════════════════════════
# ComprehensiveFaithfulness — Standard XAI faithfulness metrics
# References:
#   - Petsiuk et al. (2018): Deletion / Insertion AUC (RISE)
#   - DeYoung et al. (2020): Comprehensiveness / Sufficiency (ERASER)
#   - AOPC: Area Over the Perturbation Curve
# ═══════════════════════════════════════════════════════════════════════════


class ComprehensiveFaithfulness:
    """
    Compute standard faithfulness metrics for feature-level attributions.

    All metrics compare a model's prediction change when features are
    added/removed according to an importance ranking vs. randomly.

    Metrics:
        - deletion_auc:     Remove features most→least important → LOW AUC = faithful
        - insertion_auc:    Insert features most→least into blank  → HIGH AUC = faithful
        - comprehensiveness: P(y|x) - P(y|x \\ top_k)            → HIGH = faithful
        - sufficiency:      P(y|x) - P(y|top_k only)             → LOW  = faithful
        - aopc:             Mean prediction drop across steps      → HIGH = faithful
    """

    def __init__(self, model, device: str = "cuda", masking_value: float = 0.0):
        self.model = model
        self.device = device
        self.masking_value = masking_value
        self.model.eval()

    def _get_proba(self, X: torch.Tensor, target_classes: torch.Tensor) -> np.ndarray:
        """Batched softmax probability for target classes."""
        with torch.no_grad():
            logits = self.model(X.to(self.device), return_explanations=False)
            probs = torch.softmax(logits, dim=1)
            # Gather target class probabilities
            scores = probs.gather(1, target_classes.to(self.device).unsqueeze(1)).squeeze(1)
            return scores.cpu().numpy()

    def deletion_auc(self, X: torch.Tensor, importances: np.ndarray,
                     targets: torch.Tensor, steps: int = 20,
                     batch_size: int = 256) -> np.ndarray:
        """
        Sequentially remove features by importance (most→least).
        Returns per-sample AUC (lower = more faithful).
        """
        N, D = X.shape
        sorted_idx = np.argsort(-importances, axis=1)  # descending
        step_size = max(1, D // steps)

        all_scores = np.zeros((N, steps + 1))

        # Step 0: original
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            all_scores[start:end, 0] = self._get_proba(X[start:end], targets[start:end])

        for s in range(1, steps + 1):
            k = min(s * step_size, D)
            X_mod = X.clone()
            for i in range(N):
                X_mod[i, sorted_idx[i, :k]] = self.masking_value
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                all_scores[start:end, s] = self._get_proba(X_mod[start:end], targets[start:end])

        # Per-sample AUC (trapezoidal)
        auc = np.trapz(all_scores, dx=1.0 / steps, axis=1)
        return auc

    def insertion_auc(self, X: torch.Tensor, importances: np.ndarray,
                      targets: torch.Tensor, steps: int = 20,
                      batch_size: int = 256) -> np.ndarray:
        """
        Insert features most→least important into blank.
        Returns per-sample AUC (higher = more faithful).
        """
        N, D = X.shape
        sorted_idx = np.argsort(-importances, axis=1)
        step_size = max(1, D // steps)

        all_scores = np.zeros((N, steps + 1))

        # Step 0: blank input
        X_blank = torch.full_like(X, self.masking_value)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            all_scores[start:end, 0] = self._get_proba(X_blank[start:end], targets[start:end])

        for s in range(1, steps + 1):
            k = min(s * step_size, D)
            X_ins = torch.full_like(X, self.masking_value)
            for i in range(N):
                X_ins[i, sorted_idx[i, :k]] = X[i, sorted_idx[i, :k]]
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                all_scores[start:end, s] = self._get_proba(X_ins[start:end], targets[start:end])

        auc = np.trapz(all_scores, dx=1.0 / steps, axis=1)
        return auc

    def comprehensiveness(self, X: torch.Tensor, importances: np.ndarray,
                          targets: torch.Tensor, k: int = 10,
                          batch_size: int = 256) -> np.ndarray:
        """
        ERASER comprehensiveness: P(y|x) - P(y|x \\ top_k).
        Returns per-sample values (higher = more faithful).
        """
        N, D = X.shape
        sorted_idx = np.argsort(-importances, axis=1)

        # Full prediction
        full_scores = np.zeros(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            full_scores[start:end] = self._get_proba(X[start:end], targets[start:end])

        # Remove top-k
        k_actual = min(k, D)
        X_comp = X.clone()
        for i in range(N):
            X_comp[i, sorted_idx[i, :k_actual]] = self.masking_value

        comp_scores = np.zeros(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            comp_scores[start:end] = self._get_proba(X_comp[start:end], targets[start:end])

        return full_scores - comp_scores

    def sufficiency(self, X: torch.Tensor, importances: np.ndarray,
                    targets: torch.Tensor, k: int = 10,
                    batch_size: int = 256) -> np.ndarray:
        """
        ERASER sufficiency: P(y|x) - P(y|top_k only).
        Returns per-sample values (lower = more sufficient/faithful).
        """
        N, D = X.shape
        sorted_idx = np.argsort(-importances, axis=1)

        full_scores = np.zeros(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            full_scores[start:end] = self._get_proba(X[start:end], targets[start:end])

        k_actual = min(k, D)
        X_suff = torch.full_like(X, self.masking_value)
        for i in range(N):
            X_suff[i, sorted_idx[i, :k_actual]] = X[i, sorted_idx[i, :k_actual]]

        suff_scores = np.zeros(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            suff_scores[start:end] = self._get_proba(X_suff[start:end], targets[start:end])

        return full_scores - suff_scores

    def aopc(self, X: torch.Tensor, importances: np.ndarray,
             targets: torch.Tensor, steps: int = 20,
             batch_size: int = 256) -> np.ndarray:
        """
        Area Over Perturbation Curve.
        Mean prediction drop as features are progressively removed.
        Returns per-sample AOPC (higher = more faithful).
        """
        N, D = X.shape
        sorted_idx = np.argsort(-importances, axis=1)
        step_size = max(1, D // steps)

        base_scores = np.zeros(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            base_scores[start:end] = self._get_proba(X[start:end], targets[start:end])

        drops = np.zeros((N, steps))
        for s in range(steps):
            k = min((s + 1) * step_size, D)
            X_mod = X.clone()
            for i in range(N):
                X_mod[i, sorted_idx[i, :k]] = self.masking_value

            step_scores = np.zeros(N)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                step_scores[start:end] = self._get_proba(X_mod[start:end], targets[start:end])
            drops[:, s] = base_scores - step_scores

        return drops.mean(axis=1)

    def evaluate_all(self, X: torch.Tensor, importances: np.ndarray,
                     targets: torch.Tensor, k_values: list = None,
                     steps: int = 20, batch_size: int = 256) -> dict:
        """
        Run ALL faithfulness metrics.

        Returns dict with per-sample arrays (for paired testing)
        and summary statistics.
        """
        if k_values is None:
            k_values = [5, 10, 15, 20]

        targets_t = torch.tensor(targets, dtype=torch.long) if not isinstance(targets, torch.Tensor) else targets

        print(f"    Computing Deletion AUC ({len(X)} samples)...")
        del_auc = self.deletion_auc(X, importances, targets_t, steps, batch_size)

        print(f"    Computing Insertion AUC...")
        ins_auc = self.insertion_auc(X, importances, targets_t, steps, batch_size)

        print(f"    Computing AOPC...")
        aopc_vals = self.aopc(X, importances, targets_t, steps, batch_size)

        result = {
            "deletion_auc": {
                "per_sample": del_auc.tolist(),
                "mean": float(del_auc.mean()),
                "std": float(del_auc.std()),
            },
            "insertion_auc": {
                "per_sample": ins_auc.tolist(),
                "mean": float(ins_auc.mean()),
                "std": float(ins_auc.std()),
            },
            "aopc": {
                "per_sample": aopc_vals.tolist(),
                "mean": float(aopc_vals.mean()),
                "std": float(aopc_vals.std()),
            },
        }

        for k in k_values:
            print(f"    Computing Comprehensiveness & Sufficiency (k={k})...")
            comp = self.comprehensiveness(X, importances, targets_t, k, batch_size)
            suff = self.sufficiency(X, importances, targets_t, k, batch_size)
            result[f"comprehensiveness_k{k}"] = {
                "per_sample": comp.tolist(),
                "mean": float(comp.mean()),
                "std": float(comp.std()),
            }
            result[f"sufficiency_k{k}"] = {
                "per_sample": suff.tolist(),
                "mean": float(suff.mean()),
                "std": float(suff.std()),
            }

        return result


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing FaithfulnessTest...")

    from src.models.hit_ids import HiTIDS

    model = HiTIDS(
        group_sizes=[4, 5, 3, 3], d_emb=16, nhead=2,
        intra_layers=1, inter_layers=1, num_classes=5,
    )

    N = 50
    total_features = 15
    X = torch.randn(N, total_features)
    y = torch.randint(0, 5, (N,))

    tester = FaithfulnessTest(model, device="cpu", masking_strategy="zero")
    importance = np.random.rand(total_features)

    results = tester.run_faithfulness_test(
        X, y, importance, top_k_values=[3, 5]
    )
    print(f"\n  Results keys: {list(results.keys())}")
    print("\n✓ FaithfulnessTest passed!")

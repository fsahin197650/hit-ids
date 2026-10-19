"""
XAI Visualization module for HiT-IDS.

Generates publication-quality figures for the paper:
    1. Per-class group importance heatmap (15 classes × 4 groups)
    2. Top-10 feature bar chart (attention vs SHAP side-by-side)
    3. Attack signature radar chart
    4. XCS distribution histogram
    5. Faithfulness test curves
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


# Set publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


class XAIVisualizer:
    """
    Generates all XAI visualizations for the HiT-IDS paper.

    Args:
        save_dir: Directory to save figures.
        label_names: Class label names.
        group_names: Feature group names.
    """

    def __init__(
        self,
        save_dir: str = "results/figures",
        label_names: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None,
    ) -> None:
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.label_names = label_names or [f"Class_{i}" for i in range(15)]
        self.group_names = group_names or [
            "Temporal", "Volume", "Protocol", "Behavioral"
        ]

    def plot_group_importance_heatmap(
        self,
        group_importance: np.ndarray,
        save_name: str = "group_importance_heatmap.pdf",
    ) -> str:
        """
        Plot per-class group importance heatmap.

        Args:
            group_importance: Shape (num_classes, num_groups).
            save_name: Filename to save.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 10))

        num_classes = group_importance.shape[0]
        labels = self.label_names[:num_classes]

        sns.heatmap(
            group_importance,
            xticklabels=self.group_names,
            yticklabels=labels,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Importance Score"},
        )

        ax.set_title("Per-Class Feature Group Importance", fontweight="bold")
        ax.set_xlabel("Feature Group")
        ax.set_ylabel("Attack Class")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        path = os.path.join(self.save_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_top_features_comparison(
        self,
        attention_importance: np.ndarray,
        shap_importance: np.ndarray,
        feature_names: List[str],
        top_k: int = 10,
        save_name: str = "top_features_comparison.pdf",
    ) -> str:
        """
        Plot top-K features bar chart: attention vs SHAP side-by-side.

        Args:
            attention_importance: (num_features,).
            shap_importance: (num_features,).
            feature_names: Names of all features.
            top_k: Number of top features to show.
            save_name: Filename.

        Returns:
            Path to saved figure.
        """
        # Get union of top-K from both methods
        top_attn_idx = np.argsort(attention_importance)[-top_k:][::-1]
        top_shap_idx = np.argsort(shap_importance)[-top_k:][::-1]
        union_idx = list(dict.fromkeys(list(top_attn_idx) + list(top_shap_idx)))[:top_k]

        names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in union_idx]
        attn_vals = attention_importance[union_idx]
        shap_vals = shap_importance[union_idx]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.barh(x + width/2, attn_vals, width, label="Attention Rollout",
                        color="#2196F3", alpha=0.85)
        bars2 = ax.barh(x - width/2, shap_vals, width, label="SHAP",
                        color="#FF5722", alpha=0.85)

        ax.set_yticks(x)
        ax.set_yticklabels(names)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top-{top_k} Feature Importance: Attention vs SHAP",
                     fontweight="bold")
        ax.legend(loc="lower right")
        ax.invert_yaxis()

        path = os.path.join(self.save_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_attack_signature_radar(
        self,
        group_importance: np.ndarray,
        class_indices: Optional[List[int]] = None,
        save_name: str = "attack_signatures_radar.pdf",
    ) -> str:
        """
        Plot radar chart showing attack signatures (group activation patterns).

        Args:
            group_importance: (num_classes, num_groups).
            class_indices: Which classes to plot (default: all).
            save_name: Filename.

        Returns:
            Path to saved figure.
        """
        num_groups = group_importance.shape[1]
        angles = np.linspace(0, 2 * np.pi, num_groups, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        if class_indices is None:
            # Select interesting classes (skip Benign, pick attack types)
            n_classes = min(group_importance.shape[0], 8)
            class_indices = list(range(1, n_classes + 1))

        colors = plt.cm.Set2(np.linspace(0, 1, len(class_indices)))

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for idx, cls in enumerate(class_indices):
            if cls >= group_importance.shape[0]:
                continue
            values = group_importance[cls].tolist()
            values += values[:1]
            label = self.label_names[cls] if cls < len(self.label_names) else f"Class_{cls}"
            ax.plot(angles, values, "o-", linewidth=2, label=label, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.group_names)
        ax.set_title("Attack Signatures: Feature Group Activation", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        path = os.path.join(self.save_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_xcs_distribution(
        self,
        xcs_scores: np.ndarray,
        save_name: str = "xcs_distribution.pdf",
    ) -> str:
        """
        Plot XCS distribution histogram across test samples.

        Args:
            xcs_scores: Array of XCS values per sample.
            save_name: Filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(xcs_scores, bins=50, color="#4CAF50", alpha=0.75,
                edgecolor="black", linewidth=0.5)

        mean_xcs = np.mean(xcs_scores)
        median_xcs = np.median(xcs_scores)
        ax.axvline(mean_xcs, color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_xcs:.3f}")
        ax.axvline(median_xcs, color="blue", linestyle=":", linewidth=2,
                   label=f"Median: {median_xcs:.3f}")

        ax.set_xlabel("XAI Consistency Score (XCS)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of XCS Across Test Samples", fontweight="bold")
        ax.legend()

        path = os.path.join(self.save_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_faithfulness_curves(
        self,
        faithfulness_results: Dict[str, Any],
        save_name: str = "faithfulness_test.pdf",
    ) -> str:
        """
        Plot faithfulness test results — accuracy drop vs K.

        Args:
            faithfulness_results: Output from FaithfulnessTest.
            save_name: Filename.

        Returns:
            Path to saved figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        top_k_results = faithfulness_results["top_k_results"]
        bottom_k_results = faithfulness_results["bottom_k_results"]
        random_k_results = faithfulness_results["random_k_results"]

        k_values = sorted(top_k_results.keys())

        # Accuracy drop curves
        top_drops = [top_k_results[k]["acc_drop"] for k in k_values]
        bottom_drops = [bottom_k_results[k]["acc_drop"] for k in k_values]
        random_drops = [random_k_results[k]["acc_drop"] for k in k_values]

        ax1.plot(k_values, top_drops, "o-", label="Top-K (Most Important)",
                 color="#F44336", linewidth=2)
        ax1.plot(k_values, random_drops, "s--", label="Random-K",
                 color="#9E9E9E", linewidth=2)
        ax1.plot(k_values, bottom_drops, "^:", label="Bottom-K (Least Important)",
                 color="#4CAF50", linewidth=2)

        ax1.set_xlabel("K (Number of Masked Features)")
        ax1.set_ylabel("Accuracy Drop")
        ax1.set_title("Faithfulness: Accuracy Drop", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # F1 drop curves
        top_f1 = [top_k_results[k]["f1_drop"] for k in k_values]
        bottom_f1 = [bottom_k_results[k]["f1_drop"] for k in k_values]
        random_f1 = [random_k_results[k]["f1_drop"] for k in k_values]

        ax2.plot(k_values, top_f1, "o-", label="Top-K (Most Important)",
                 color="#F44336", linewidth=2)
        ax2.plot(k_values, random_f1, "s--", label="Random-K",
                 color="#9E9E9E", linewidth=2)
        ax2.plot(k_values, bottom_f1, "^:", label="Bottom-K (Least Important)",
                 color="#4CAF50", linewidth=2)

        ax2.set_xlabel("K (Number of Masked Features)")
        ax2.set_ylabel("Macro-F1 Drop")
        ax2.set_title("Faithfulness: F1 Drop", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_name: str = "training_curves.pdf",
    ) -> str:
        """
        Plot training loss/F1 curves across both phases.

        Args:
            history: Training history dict with keys like 'train_loss', 'val_f1'.
            save_name: Filename.

        Returns:
            Path to saved figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(history.get("train_loss", [])) + 1)

        # Loss
        axes[0].plot(epochs, history.get("train_loss", []), label="Train",
                     color="#2196F3", linewidth=1.5)
        axes[0].plot(epochs, history.get("val_loss", []), label="Validation",
                     color="#F44336", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training/Validation Loss", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1
        axes[1].plot(epochs, history.get("train_f1", []), label="Train",
                     color="#2196F3", linewidth=1.5)
        axes[1].plot(epochs, history.get("val_f1", []), label="Validation",
                     color="#F44336", linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro F1")
        axes[1].set_title("Macro F1 Score", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning Rate
        axes[2].plot(epochs, history.get("learning_rate", []),
                     color="#4CAF50", linewidth=1.5)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule", fontweight="bold")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        normalize: bool = True,
        save_name: str = "confusion_matrix.pdf",
    ) -> str:
        """
        Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix, shape (num_classes, num_classes).
            normalize: Whether to normalize rows.
            save_name: Filename.

        Returns:
            Path to saved figure.
        """
        if normalize:
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        else:
            cm_norm = cm

        num_classes = cm.shape[0]
        labels = self.label_names[:num_classes]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f" if normalize else "d",
            xticklabels=labels, yticklabels=labels,
            cmap="Blues", ax=ax, linewidths=0.5,
            vmin=0, vmax=1 if normalize else None,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
        ax.set_title(title, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        path = os.path.join(self.save_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing XAIVisualizer...")

    viz = XAIVisualizer(save_dir="/tmp/hit_ids_test_figures")
    np.random.seed(42)

    # Dummy data
    group_importance = np.random.dirichlet([1] * 4, size=15)
    viz.plot_group_importance_heatmap(group_importance)

    attn_imp = np.random.rand(72)
    shap_imp = attn_imp + np.random.randn(72) * 0.1
    feature_names = [f"Feature_{i}" for i in range(72)]
    viz.plot_top_features_comparison(attn_imp, shap_imp, feature_names)

    viz.plot_attack_signature_radar(group_importance, class_indices=[1, 2, 3, 4, 5])

    xcs_scores = np.random.normal(0.7, 0.15, 500)
    viz.plot_xcs_distribution(xcs_scores)

    viz.plot_confusion_matrix(np.random.randint(0, 100, (5, 5)))

    print("\n✓ XAIVisualizer test passed!")

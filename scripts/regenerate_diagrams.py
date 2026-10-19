#!/usr/bin/env python3
"""
Regenerate architectural diagrams (Fig 1-3) at 600 DPI using matplotlib.
These replace the low-res 640x640 AI-generated images.

Usage:
    python scripts/regenerate_diagrams.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────
C = {
    "temporal":   "#DBEAFE",  "temporal_e":   "#3B82F6",
    "volume":     "#FEF3C7",  "volume_e":     "#F59E0B",
    "protocol":   "#D1FAE5",  "protocol_e":   "#10B981",
    "behavioral": "#FCE7F3",  "behavioral_e": "#EC4899",
    "box":        "#F3F4F6",  "box_e":        "#6B7280",
    "head":       "#EDE9FE",  "head_e":       "#7C3AED",
    "inter":      "#FFF7ED",  "inter_e":      "#EA580C",
    "cls":        "#FEE2E2",  "cls_e":        "#DC2626",
    "benign":     "#D1FAE5",  "benign_e":     "#059669",
    "attack":     "#FEE2E2",  "attack_e":     "#DC2626",
    "intrinsic":  "#DBEAFE",  "intrinsic_e":  "#2563EB",
    "posthoc":    "#FFF1F2",  "posthoc_e":    "#E11D48",
    "xcs":        "#F5F3FF",  "xcs_e":        "#7C3AED",
}


def _box(ax, x, y, w, h, text, fc, ec, fs=9, fw="normal", lw=1.2, alpha=1.0,
          va="center", ha="center", text_color="black", rounded=True):
    """Draw a rounded box with centered text."""
    style = "round,pad=0.02" if rounded else "square,pad=0"
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=style, facecolor=fc, edgecolor=ec,
                          linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=text_color, zorder=3, linespacing=1.3)
    return rect


def _arrow(ax, x1, y1, x2, y2, color="#6B7280", lw=1.2, style="->",
           connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, color=color, lw=lw,
                          connectionstyle=connectionstyle, zorder=1,
                          mutation_scale=12)
    ax.add_patch(arr)


# ═══════════════════════════════════════════════════════════════════════
# FIG 1: HiT-IDS Architecture
# ═══════════════════════════════════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 7)
    ax.axis("off")

    # ── Stage labels at top ──
    stages = [
        (0.4, "Stage 1:\nInput"),
        (2.3, "Stage 2:\nFeature\nGrouping"),
        (4.3, "Stage 3:\nPer-Feature\nTokenization"),
        (6.3, "Stage 4:\nIntra-Group\nTransformer"),
        (8.5, "Stage 5:\nInter-Group\nTransformer"),
        (10.2, "Stage 6:\nClassification\nHead"),
    ]
    for sx, label in stages:
        ax.text(sx, 6.65, label, ha="center", va="top", fontsize=8,
                fontweight="bold", color="#374151")

    # ── Stage 1: Input ──
    _box(ax, 0.4, 3.5, 0.9, 1.8, "61\nNetwork\nFlow\nFeatures",
         C["box"], C["box_e"], fs=9, fw="bold")

    # ── Stage 2: Feature Groups ──
    groups = [
        ("Temporal\n(23 features)", C["temporal"], C["temporal_e"], 5.3),
        ("Volume\n(24 features)", C["volume"], C["volume_e"], 4.1),
        ("Protocol\n(10 features)", C["protocol"], C["protocol_e"], 2.9),
        ("Behavioral\n(4 features)", C["behavioral"], C["behavioral_e"], 1.7),
    ]
    for label, fc, ec, gy in groups:
        _box(ax, 2.3, gy, 1.3, 0.85, label, fc, ec, fs=8, fw="bold")
        _arrow(ax, 0.85, 3.5, 1.65, gy)

    # ── Stage 3: Tokenization ──
    for i, (_, _, _, gy) in enumerate(groups):
        _box(ax, 4.3, gy, 1.1, 0.65, "Linear\nd=128", C["box"], C["box_e"], fs=8)
        _arrow(ax, 2.95, gy, 3.75, gy)

    # ── CLS tokens ──
    for _, _, _, gy in groups:
        _box(ax, 5.15, gy + 0.0, 0.4, 0.35, "[CLS]", C["cls"], C["cls_e"], fs=7, fw="bold")
        _arrow(ax, 4.85, gy, 4.95, gy)

    # ── Stage 4: Intra-Group Transformers ──
    colors_intra = [C["temporal"], C["volume"], C["protocol"], C["behavioral"]]
    edges_intra = [C["temporal_e"], C["volume_e"], C["protocol_e"], C["behavioral_e"]]
    for i, (_, _, _, gy) in enumerate(groups):
        _box(ax, 6.5, gy, 1.4, 0.75,
             "Intra-Group\nTransformer",
             colors_intra[i], edges_intra[i], fs=8, fw="bold")
        _arrow(ax, 5.35, gy, 5.8, gy)
        ax.text(6.5, gy - 0.52, "L=2, h=8", ha="center", fontsize=6.5,
                color="#6B7280", style="italic")

    # ── Stage 5: Inter-Group Transformer ──
    _box(ax, 8.5, 3.5, 1.5, 2.0,
         "Inter-Group\nTransformer\n\nL=2, h=8",
         C["inter"], C["inter_e"], fs=9, fw="bold")

    for _, _, _, gy in groups:
        _arrow(ax, 7.2, gy, 7.75, 3.5,
               color="#EA580C", lw=1.0)

    # CLS output
    _box(ax, 8.5, 5.1, 0.4, 0.35, "[CLS]", C["cls"], C["cls_e"], fs=7, fw="bold")
    _arrow(ax, 8.5, 4.5, 8.5, 4.93, color="#DC2626")

    # ── Stage 6: Classification Head ──
    _box(ax, 10.2, 4.5, 1.0, 0.75,
         "Binary:\nBenign vs\nAttack",
         C["benign"], C["benign_e"], fs=8, fw="bold")
    _arrow(ax, 8.7, 5.1, 9.7, 4.7, color="#059669")

    _box(ax, 10.2, 2.5, 1.0, 0.75,
         "Multi-class:\n12 Attack\nTypes",
         C["attack"], C["attack_e"], fs=8, fw="bold")
    _arrow(ax, 8.7, 5.1, 9.7, 2.7, color="#DC2626")

    # Title
    ax.set_title("HiT-IDS architecture", fontsize=13,
                 fontweight="bold", pad=5)

    fig.savefig(f"{FIG_DIR}/fig1_architecture.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.1)
    fig.savefig(f"{FIG_DIR}/fig1_architecture.png", format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("  ✓ fig1_architecture.pdf + .png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 2: Cascade Detection Pipeline
# ═══════════════════════════════════════════════════════════════════════
def fig2_cascade():
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(-1, 9)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    # Input
    _box(ax, 4, 8, 2.5, 0.7, "Network Flow (61 features)", C["box"], C["box_e"], fs=10, fw="bold")

    # Stage 1 label
    ax.text(-0.3, 6.5, "Stage 1\nBinary\nDetection", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F9FAFB", ec="#D1D5DB"))

    # Binary classifier
    _box(ax, 4, 6.5, 2.8, 0.8,
         "HiT-IDS Binary Classifier\nBenign vs. Attack",
         C["intrinsic"], C["intrinsic_e"], fs=10, fw="bold")
    _arrow(ax, 4, 7.65, 4, 6.9, lw=1.5)

    # Binary F1 annotation
    ax.text(6.8, 6.5, "Binary F1 = 0.993\n(5-fold, calibrated)", ha="left", va="center",
            fontsize=9, color="#2563EB", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#EFF6FF", ec="#93C5FD"))

    # Split
    _box(ax, 1.8, 5, 1.6, 0.7, "BENIGN", C["benign"], C["benign_e"], fs=10, fw="bold")
    _box(ax, 6.2, 5, 1.6, 0.7, "MALICIOUS", C["attack"], C["attack_e"], fs=10, fw="bold")

    _arrow(ax, 3.0, 6.1, 1.8, 5.35, color="#059669", lw=1.5)
    _arrow(ax, 5.0, 6.1, 6.2, 5.35, color="#DC2626", lw=1.5)

    ax.text(2.0, 5.75, "~83%", ha="center", fontsize=8, color="#059669", fontweight="bold")
    ax.text(6.0, 5.75, "~17%", ha="center", fontsize=8, color="#DC2626", fontweight="bold")

    # Stage 2 label
    ax.text(-0.3, 3.3, "Stage 2\nAttack\nClassification", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F9FAFB", ec="#D1D5DB"))

    # Multi-class classifier
    _box(ax, 4, 3.6, 2.8, 0.8,
         "HiT-IDS Multi-class Classifier\n12 Attack Types",
         C["head"], C["head_e"], fs=10, fw="bold")
    _arrow(ax, 6.2, 4.65, 4.5, 4.0, color="#DC2626", lw=1.5)

    ax.text(6.8, 3.6, "Macro-F1 = 0.895 (13 cl.)\n0.948 (11 cl.), 5-fold", ha="left", va="center",
            fontsize=9, color="#7C3AED", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#F5F3FF", ec="#C4B5FD"))

    # Attack types
    attacks_row1 = ["DDoS", "DoS Hulk", "PortScan", "DoS\nGoldenEye"]
    attacks_row2 = ["Bot", "Web\nAttack", "FTP-\nPatator", "SSH-\nPatator"]
    attacks_row3 = ["DoS\nSlowhttptest", "DoS\nSlowloris", "Heartbleed", "Infiltration"]

    for i, name in enumerate(attacks_row1):
        x = 1.5 + i * 2.0
        _box(ax, x, 2.2, 1.3, 0.55, name, "#E0E7FF", "#6366F1", fs=7.5)
        _arrow(ax, 4, 3.2, x, 2.47, color="#6366F1", lw=0.8)

    for i, name in enumerate(attacks_row2):
        x = 1.5 + i * 2.0
        _box(ax, x, 1.3, 1.3, 0.55, name, "#E0E7FF", "#6366F1", fs=7.5)
        _arrow(ax, 4, 3.2, x, 1.57, color="#6366F1", lw=0.8)

    for i, name in enumerate(attacks_row3):
        x = 1.5 + i * 2.0
        _box(ax, x, 0.4, 1.3, 0.55, name, "#E0E7FF", "#6366F1", fs=7.5)
        _arrow(ax, 4, 3.2, x, 0.67, color="#6366F1", lw=0.8)

    ax.set_title("Two-stage cascade", fontsize=13,
                 fontweight="bold", pad=10)

    fig.savefig(f"{FIG_DIR}/fig2_cascade_detection.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.1)
    fig.savefig(f"{FIG_DIR}/fig2_cascade_detection.png", format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("  ✓ fig2_cascade_detection.pdf + .png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 3: Dual XAI Framework
# ═══════════════════════════════════════════════════════════════════════
def fig3_xai():
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")

    # ── HiT-IDS Model (top center) ──
    _box(ax, 5, 7, 2.5, 0.7, "HiT-IDS Model", C["box"], C["box_e"], fs=11, fw="bold")

    # ── Left: Intrinsic pathway ──
    # Background panel
    rect_l = FancyBboxPatch((0.3, 0.9), 3.9, 5.3,
                            boxstyle="round,pad=0.1", facecolor=C["intrinsic"],
                            edgecolor=C["intrinsic_e"], linewidth=1.5, alpha=0.3, zorder=0)
    ax.add_patch(rect_l)
    ax.text(2.25, 6.05, "Intrinsic (Attention-based)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1E40AF")

    _box(ax, 2.25, 5.2, 2.8, 0.65,
         "Attention Weights\nIntra-group + Inter-group",
         "#BFDBFE", C["intrinsic_e"], fs=9, fw="bold")
    _arrow(ax, 4.0, 6.65, 2.25, 5.52, color=C["intrinsic_e"], lw=1.5)

    _box(ax, 2.25, 4.0, 2.8, 0.65,
         "Attention Rollout\nΣ (A × I) across layers",
         "#BFDBFE", C["intrinsic_e"], fs=9)
    _arrow(ax, 2.25, 4.87, 2.25, 4.32, color=C["intrinsic_e"], lw=1.2)

    _box(ax, 2.25, 2.8, 2.8, 0.65,
         "Feature Importance\nRanking (Attention)",
         "#93C5FD", C["intrinsic_e"], fs=9, fw="bold")
    _arrow(ax, 2.25, 3.67, 2.25, 3.12, color=C["intrinsic_e"], lw=1.2)

    # Cost annotation
    ax.text(2.25, 1.95, "3.4 ms / flow\n(0.035 ms batched)", ha="center", fontsize=9,
            color="#1E40AF", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#EFF6FF", ec="#93C5FD"))

    # ── Right: Post-hoc pathway ──
    rect_r = FancyBboxPatch((5.8, 0.9), 3.9, 5.3,
                            boxstyle="round,pad=0.1", facecolor=C["posthoc"],
                            edgecolor=C["posthoc_e"], linewidth=1.5, alpha=0.3, zorder=0)
    ax.add_patch(rect_r)
    ax.text(7.75, 6.05, "Post-hoc (SHAP-based)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#9F1239")

    _box(ax, 7.75, 5.2, 2.8, 0.65,
         "GradientExplainer\n(SHAP)",
         "#FECDD3", C["posthoc_e"], fs=9, fw="bold")
    _arrow(ax, 6.0, 6.65, 7.75, 5.52, color=C["posthoc_e"], lw=1.5)

    _box(ax, 7.75, 4.0, 2.8, 0.65,
         "SHAP Values\nPer-feature marginal contribution",
         "#FECDD3", C["posthoc_e"], fs=9)
    _arrow(ax, 7.75, 4.87, 7.75, 4.32, color=C["posthoc_e"], lw=1.2)

    _box(ax, 7.75, 2.8, 2.8, 0.65,
         "Feature Importance\nRanking (SHAP)",
         "#FDA4AF", C["posthoc_e"], fs=9, fw="bold")
    _arrow(ax, 7.75, 3.67, 7.75, 3.12, color=C["posthoc_e"], lw=1.2)

    # Cost annotation
    ax.text(7.75, 1.95, "518 ms / flow", ha="center", fontsize=9,
            color="#9F1239", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFF1F2", ec="#FDA4AF"))

    # ── Bottom: XCS convergence ──
    _box(ax, 5, 1.0, 3.5, 0.8,
         "Consistency check\nSpearman ρ = 0.43  |  Top-10 overlap = 44%",
         C["xcs"], C["xcs_e"], fs=9, fw="bold")
    _arrow(ax, 2.25, 2.47, 3.8, 1.4, color=C["xcs_e"], lw=1.5)
    _arrow(ax, 7.75, 2.47, 6.2, 1.4, color=C["xcs_e"], lw=1.5)

    # Speedup annotation
    ax.text(5, 0.15, "Rollout: 150× cheaper than GradientSHAP on the same model, 2.5× cheaper than XGBoost TreeSHAP (one flow)", ha="center",
            fontsize=10, color="#7C3AED", fontweight="bold", fontstyle="italic")

    ax.set_title("Explanation extraction and consistency check", fontsize=13,
                 fontweight="bold", pad=10)

    fig.savefig(f"{FIG_DIR}/fig3_xai_framework.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.1)
    fig.savefig(f"{FIG_DIR}/fig3_xai_framework.png", format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("  ✓ fig3_xai_framework.pdf + .png")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  REGENERATING ARCHITECTURAL DIAGRAMS (600 DPI)")
    print("=" * 60)

    fig1_architecture()
    fig2_cascade()
    fig3_xai()

    print()
    print("  DONE — All 3 diagrams regenerated")
    print("=" * 60)

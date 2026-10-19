#!/usr/bin/env python3
"""
Convert paper_draft.md to DOCX with embedded figures and fixed formulas.

Steps:
1. Read paper_draft.md
2. Replace figure references with actual image embeds
3. Fix LaTeX formulas for pandoc
4. Write pandoc-ready .md
5. Convert to .docx via pandoc
"""

import re
import subprocess
import sys
import os

SRC = "paper_draft.md"
OUT_MD = "paper_draft_pandoc.md"
OUT_DOCX = "paper_draft.docx"
FIG_DIR = "results/figures"

# Map figure references to actual image files
FIGURE_MAP = {
    "(Fig. 1)": f"(Fig. 1)\n\n![Figure 1: HiT-IDS Architecture]({FIG_DIR}/fig1_architecture.png)",
    "(Fig. 2)": f"(Fig. 2)\n\n![Figure 2: Group Importance Heatmap]({FIG_DIR}/group_importance_heatmap.png)",
}


def main():
    with open(SRC, "r") as f:
        text = f.read()

    # ── 1. Embed figures at their reference points ───────────────────
    for ref, replacement in FIGURE_MAP.items():
        text = text.replace(ref, replacement, 1)

    # Replace "Figure X" and "Figure Y" inline references with images
    text = text.replace(
        "**Figure X.** Explainability cost comparison (see `results/figures/xai_cost_comparison.pdf`).",
        f"![Figure 3: Explainability Cost Comparison]({FIG_DIR}/xai_cost_comparison.png)\n\n*Figure 3. Explainability computational cost per sample across models. Green = intrinsic (no extra cost), Red = post-hoc SHAP.*"
    )
    text = text.replace(
        "**Figure Y.** Per-sample explainability agreement matrix (see `results/figures/per_sample_agreement.pdf`).",
        f"![Figure 4: Per-Sample Agreement Matrix]({FIG_DIR}/per_sample_agreement.png)\n\n*Figure 4. Per-sample Spearman ρ agreement between models\\' XAI explanations.*"
    )

    # ── 2. Add additional figures at end of relevant sections ────────
    # Add confusion matrix after per-class analysis (Section 5.3)
    text = text.replace(
        "### 5.4 Explainability Results",
        f"![Figure 5: Confusion Matrix]({FIG_DIR}/confusion_matrix.png)\n\n*Figure 5. Confusion matrix for the attack classifier (13 classes).*\n\n### 5.4 Explainability Results"
    )

    # Add top features comparison figure after Table 7
    text = text.replace(
        "Seven of the ten features appear in both lists.",
        f"![Figure 6: Top Features Comparison]({FIG_DIR}/top_features_comparison.png)\n\n*Figure 6. Top-10 features from Attention Rollout and SHAP side by side.*\n\nSeven of the ten features appear in both lists."
    )

    # Add XCS distribution figure after XCS table
    text = text.replace(
        "The mean XCS of 0.479 shows a **moderate positive correlation**",
        f"![Figure 7: XCS Distribution]({FIG_DIR}/xcs_distribution.png)\n\n*Figure 7. Distribution of per-sample XCS (Spearman ρ) across 200 attack samples.*\n\nThe mean XCS of 0.479 shows a **moderate positive correlation**"
    )

    # Add attack signatures radar after Section 6.3
    text = text.replace(
        "These patterns are not random. They match documented attack methods",
        f"![Figure 8: Attack Signatures Radar]({FIG_DIR}/attack_signatures_radar.png)\n\n*Figure 8. Radar plot of group importance per attack type.*\n\nThese patterns are not random. They match documented attack methods"
    )

    # Add baseline SHAP comparison and feature agreement figures
    text = text.replace(
        "Significance: \\*\\*\\* p < 0.001",
        f"![Figure 9: Feature Importance Heatmap]({FIG_DIR}/baseline_shap_comparison.png)\n\n*Figure 9. Feature importance comparison across all models (top-15 features).*\n\nSignificance: \\*\\*\\* p < 0.001"
    )

    # ── 3. Fix LaTeX formulas for pandoc ────────────────────────────
    # pandoc handles $...$ inline math fine, but we need to ensure
    # special chars aren't escaped within math blocks
    
    # Fix escaped underscores in math mode (common pandoc issue)
    # No changes needed - pandoc handles $...$ natively

    # ── 4. Clean up markdown for pandoc ─────────────────────────────
    # Remove horizontal rules that pandoc might misinterpret
    text = text.replace("\n---\n", "\n\n---\n\n")

    # Ensure bold text with ** works
    # (pandoc handles this fine)

    # ── 5. Add metadata header ──────────────────────────────────────
    header = """---
title: "HiT-IDS: A Hierarchical Transformer with Intrinsic Explainability for Network Intrusion Detection"
author: ""
date: ""
geometry: margin=2.5cm
fontsize: 11pt
---

"""
    text = header + text

    # Remove the first H1 title since it's now in metadata
    text = text.replace("# HiT-IDS: A Hierarchical Transformer with Intrinsic Explainability for Network Intrusion Detection\n", "", 1)

    # ── 6. Write and convert ────────────────────────────────────────
    with open(OUT_MD, "w") as f:
        f.write(text)
    print(f"  Written: {OUT_MD}")

    # Convert with pandoc
    cmd = [
        "pandoc", OUT_MD,
        "-o", OUT_DOCX,
        "--from", "markdown+tex_math_dollars+pipe_tables+yaml_metadata_block",
        "--reference-doc=/dev/null",  # use default template
        "--toc",                       # table of contents
        "--toc-depth=3",
        "--number-sections",
        "--wrap=auto",
    ]
    
    # Try without reference doc first
    cmd = [
        "pandoc", OUT_MD,
        "-o", OUT_DOCX,
        "--from", "markdown+tex_math_dollars+pipe_tables+yaml_metadata_block",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [ERROR] pandoc failed:")
        print(result.stderr)
        # Try simpler command
        cmd_simple = ["pandoc", OUT_MD, "-o", OUT_DOCX, "--from", "markdown"]
        print(f"  Retrying: {' '.join(cmd_simple)}")
        result = subprocess.run(cmd_simple, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] Simple pandoc also failed:")
            print(result.stderr)
            sys.exit(1)

    size_kb = os.path.getsize(OUT_DOCX) / 1024
    print(f"\n  ✅ Created: {OUT_DOCX} ({size_kb:.0f} KB)")
    print(f"  Figures embedded from: {FIG_DIR}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build the Computers & Security submission DOCX from manuscript/draft_numbered.md.

- adds the title block (title, author, affiliation, e-mail)
- inserts each figure (with caption) after the paragraph that first mentions "Fig. N"
- inserts appendix figures at the Appendix C paragraph
- converts with pandoc using manuscript/reference.docx (Times 12 pt, 1.5 spacing, A4, 2.5 cm)

Usage: python scripts/build_manuscript_docx.py [--out manuscript/HiT-IDS_CompSec_v1.docx]
"""

import argparse
import os
import re
import subprocess

FIG = "results/figures"
FIGURES = {
    1: ([f"{FIG}/fig1_architecture.png"],
        "HiT-IDS architecture. The 72 flow features are tokenised individually, partitioned into four semantic groups "
        "(19/25/15/13 features), encoded by four intra-group Transformers with their own [CLS] tokens, fused by an "
        "inter-group Transformer, and classified by a two-stage cascade (attack vs. benign, then attack type)."),
    2: ([f"{FIG}/fig3_xai_framework.png"],
        "Explanation extraction. Left: attention rollout over the intra- and inter-group encoders yields a group-level "
        "and a feature-level attribution from the forward pass. Right: GradientSHAP is used only as an independent "
        "post-hoc reference for the consistency check of Section 5.7. Latencies are the single-flow medians of Table 12."),
    3: ([f"{FIG}/clean_rv_thr/per_class_f1_models.png"],
        "Per-class F1 on the real test folds of CIC-IDS2017 (mean ± standard deviation over five folds) for HiT-IDS and "
        "the four main baselines. Heartbleed and Infiltration have 2–3 and 7–8 test flows per fold."),
    4: ([f"{FIG}/clean_rv_thr/cm_hit_ids.png"],
        "Confusion matrix of the HiT-IDS cascade summed over the five test folds (2,516,147 real flows; colour on a log "
        "scale, cells annotated with counts). Rows are true classes."),
    5: ([f"{FIG}/clean_rv_thr/roc_per_class_hit_ids.png", f"{FIG}/clean_rv_thr/pr_per_class_hit_ids.png"],
        "One-vs-rest ROC (left, log false-positive axis) and precision–recall (right) curves of HiT-IDS per class, pooled "
        "over the five test folds, with the per-class AUC and average precision in the legends."),
    6: ([f"{FIG}/clean_rv_thr/threshold_sweep.png"],
        "Stage-1 threshold sweep for the deployment model. Solid lines: real validation split (used to choose t* = 0.635); "
        "dashed: real test split (shown only for reference). Left axis: 13-class macro-F1 of the full cascade; right axis "
        "(log): false-positive rate."),
    7: ([f"{FIG}/clean_rv_thr/faithfulness_metrics.png"],
        "Faithfulness of attention rollout, input gradient and a random ranking on CIC-IDS2017 (top; attack-type model, "
        "496 flows) and UNSW-NB15 (bottom; binary model, 500 flows). Bars are means with 95% confidence intervals over "
        "flows. Lower is better for deletion AUC and sufficiency; higher is better for the rest."),
    8: ([f"{FIG}/clean_rv_thr/xcs_distribution.png"],
        "Distribution of the per-flow Spearman correlation between each attribution and GradientSHAP for the predicted "
        "class, 500 correctly classified attack flows of the deployment model."),
    9: ([f"{FIG}/clean_rv/group_importance_folds.png"],
        "Group-level attribution per attack class, mean ± standard deviation over the five independently trained fold "
        "models (up to 2,000 correctly classified flows per class and fold)."),
    10: ([f"{FIG}/clean_rv_thr/latency_bars.png"],
         "Explanation latency per flow under one protocol (100 attack flows, 100-flow SHAP background, median of five "
         "repeats, RTX 5090), for one flow at a time and for a batch of 100. Log scale."),
    11: ([f"{FIG}/clean_rv_thr/fig_attention_divergence.png"],
         "Group-level attention divergence (misclassified minus correctly classified flows of the same true class) for "
         "the five largest error groups of the deployment model."),
}
APPENDIX = [
    (f"{FIG}/reliability_diagram.png", "Fig. C1. Reliability diagrams of the Stage-1 detector and the tree ensembles on the CIC-IDS2017 test split (10 bins)."),
    (f"{FIG}/robustness_feature_noise.png", "Fig. C2. Attack-vs-benign F1 under additive Gaussian feature noise (σ in units of each feature's standard deviation)."),
    (f"{FIG}/threshold_benefit_scatter.png", "Fig. C3. Cross-dataset transfer: benefit of the adaptive threshold (ΔF1) against the adapted threshold value."),
]

TITLE_BLOCK = """---
title: "HiT-IDS: Validated hierarchical attention explanations for flow-based intrusion detection at low latency"
author: "Fatih Şahin"
---

Istanbul Topkapı University, Faculty of Engineering and Natural Sciences, Istanbul, Türkiye. E-mail: fatihsahin@topkapi.edu.tr. ORCID: 0000-0002-8036-3156.

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="manuscript/draft_numbered.md")
    ap.add_argument("--out", default="manuscript/HiT-IDS_CompSec_v1.docx")
    args = ap.parse_args()
    text = open(args.src).read()
    text = text.split("\n", 1)[1]  # drop the H1 title line (pandoc title block replaces it)

    paras = text.split("\n\n")
    placed = set()
    out = []
    for p in paras:
        out.append(p)
        if p.startswith("|") or p.startswith("#"):
            continue
        for n, (files, cap) in FIGURES.items():
            if n in placed:
                continue
            if re.search(rf"\bFig\. {n}\b", p):
                missing = [f for f in files if not os.path.exists(f)]
                if missing:
                    out.append(f"*[figure file missing: {', '.join(missing)}]*")
                else:
                    for f in files:
                        out.append(f"![]({f}){{width=100%}}" if len(files) == 1 else f"![]({f}){{width=49%}}")
                out.append(f"**Fig. {n}.** {cap}")
                placed.add(n)
    text = "\n\n".join(out)

    # Appendix C figures
    app = []
    for f, cap in APPENDIX:
        if os.path.exists(f):
            app.append(f"![]({f}){{width=85%}}\n\n**{cap}**")
    marker = "## Declarations"
    text = text.replace(marker, "\n\n".join(app) + "\n\n" + marker, 1)

    tmp = "manuscript/_build.md"
    open(tmp, "w").write(TITLE_BLOCK + text)
    cmd = ["pandoc", tmp, "-f", "markdown+pipe_tables+implicit_figures", "-t", "docx",
           "--reference-doc=manuscript/reference.docx", "-o", args.out]
    subprocess.run(cmd, check=True)
    not_placed = sorted(set(FIGURES) - placed)
    print(f"  wrote {args.out}; figures placed: {sorted(placed)}; NOT placed: {not_placed}")


if __name__ == "__main__":
    main()

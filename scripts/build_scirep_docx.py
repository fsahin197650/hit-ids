#!/usr/bin/env python3
"""Build the Scientific Reports DOCX from manuscript/scirep_numbered.md (figures after first mention)."""
import os, re, subprocess
F = "results/figures"
FIG = {1: [f"{F}/fig1_architecture.png"], 2: [f"{F}/clean_rv_thr/per_class_f1_models.png"], 3: [f"{F}/clean_rv_thr/cm_hit_ids.png"],
       4: [f"{F}/scirep/faithfulness_4methods.png"], 5: [f"{F}/scirep/stability_cic_nsm.png"], 6: [f"{F}/clean_rv_thr/latency_bars.png"],
       7: [f"{F}/scirep/transfer_per_class.png"], 8: [f"{F}/clean_rv_thr/threshold_sweep.png"]}
SUPP = [("S1", [f"{F}/clean_rv_thr/roc_per_class_hit_ids.png", f"{F}/clean_rv_thr/pr_per_class_hit_ids.png"]),
        ("S2", [f"{F}/clean_rv/group_importance_folds.png"]), ("S3", [f"{F}/clean_rv_thr/fig_attention_divergence.png"]),
        ("S4", [f"{F}/clean_rv_thr/xcs_distribution.png"]), ("S5", [f"{F}/reliability_diagram.png"]), ("S6", [f"{F}/robustness_feature_noise.png"])]
TITLE = """---
title: "HiT-IDS: hierarchical attention explanations for flow-based intrusion detection, validated for faithfulness, cost and stability"
author: "Fatih Şahin"
---

Istanbul Topkapı University, Faculty of Engineering and Natural Sciences, Istanbul, Türkiye. E-mail: fatihsahin@topkapi.edu.tr

"""
text = open("manuscript/scirep_numbered.md").read().split("\n", 1)[1]
legends = {int(m.group(1)): m.group(2) for m in re.finditer(r"\*\*Figure (\d+)\.\*\* (.+)", text)}
paras, out, placed = text.split("\n\n"), [], set()
for p in paras:
    out.append(p)
    if p.startswith(("|", "#", "**Figure")):
        continue
    for n, files in FIG.items():
        if n not in placed and re.search(rf"\bFig\. {n}\b", p):
            for f in files:
                out.append(f"![]({f}){{width=100%}}" if os.path.exists(f) else f"*[missing {f}]*")
            out.append(f"**Figure {n}.** {legends.get(n, '')}")
            placed.add(n)
text = "\n\n".join(out)
supp = ["## Supplementary figures", ""]
for tag, files in SUPP:
    for f in files:
        if os.path.exists(f):
            supp.append(f"![]({f}){{width=80%}}")
    supp.append(f"**Supplementary Figure {tag}.**\n")
text = text.replace("## References", "\n".join(supp) + "\n\n## References", 1)
open("manuscript/_build_scirep.md", "w").write(TITLE + text)
subprocess.run(["pandoc", "manuscript/_build_scirep.md", "-f", "markdown+pipe_tables+implicit_figures", "-t", "docx",
                "--reference-doc=manuscript/reference.docx", "-o", "manuscript/HiT-IDS_SciRep_v1.docx"], check=True)
print("  wrote manuscript/HiT-IDS_SciRep_v1.docx; placed figures", sorted(placed))

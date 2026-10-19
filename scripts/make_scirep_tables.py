#!/usr/bin/env python3
"""
All manuscript tables for the Scientific Reports version, generated from the
np2 result files (61 features, no Phase 2, every model threshold-calibrated).

Writes manuscript/tables_scirep.md and manuscript/numbers_scirep.json
"""

from __future__ import annotations

import json
import os

import numpy as np

L = lambda p: json.load(open(p)) if os.path.exists(p) else None
V62 = "../transformer_based_IDS_XAI_v62"
DISP = {"HiT-IDS (Cascade)": "HiT-IDS", "random_forest": "Random Forest", "xgboost": "XGBoost",
        "bilstm": "BiLSTM", "vanilla_transformer": "Flat Transformer", "Deep-IDS": "Deep-IDS (LSTM)"}
ORDER = ["xgboost", "random_forest", "bilstm", "vanilla_transformer", "HiT-IDS (Cascade)"]
LOWER = {"deletion_auc", "sufficiency_k5", "sufficiency_k10", "sufficiency_k15", "sufficiency_k20"}
MET = [("deletion_auc", "Deletion AUC ↓"), ("insertion_auc", "Insertion AUC ↑"), ("aopc", "AOPC ↑"),
       ("comprehensiveness_k5", "Comprehensiveness@5 ↑"), ("comprehensiveness_k10", "Comprehensiveness@10 ↑"),
       ("comprehensiveness_k15", "Comprehensiveness@15 ↑"), ("comprehensiveness_k20", "Comprehensiveness@20 ↑"),
       ("sufficiency_k5", "Sufficiency@5 ↓"), ("sufficiency_k10", "Sufficiency@10 ↓"),
       ("sufficiency_k15", "Sufficiency@15 ↓"), ("sufficiency_k20", "Sufficiency@20 ↓")]


def tbl(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def macro_big(rows, classes, exclude):
    big = [c for c in classes if c not in exclude]
    return np.array([np.mean([r["per_class"][c]["f1"] for c in big]) for r in rows])


def wins(R, x, y):
    return sum((R[x][m]["mean"] < R[y][m]["mean"]) if m in LOWER else (R[x][m]["mean"] > R[y][m]["mean"]) for m, _ in MET)


def main():
    os.makedirs("manuscript", exist_ok=True)
    md, N = [], {}
    cv = L("results/tables/cv_evaluation_clean_rv_thr.json")
    cnt = L("data/processed/clean_full_class_counts.json")
    nsm = L("results/tables/nsm_session_cv.json")
    unsw = L("results/unsw_nb15/tables/unsw_binary_cv_results.json")
    deep = L("revision_results/results_deepids_clean_rv.json")
    dstat = L("revision_results/stats_deep-ids_lstm_clean_rv.json")
    dep = L("results/tables/deployment_eval.json")
    fa = L("results/tables/faithfulness_results.json")
    fa2 = L("results/tables/faithfulness_results_phase2.json")
    fn = L("results/tables/nsm_faithfulness.json")
    sh = L("results/tables/stats_hardened.json")
    st = L("results/tables/attribution_stability_clean_rv.json")
    stn = L("results/tables/nsm_attribution_stability.json")
    xcs = L("results/tables/xcs_both_clean.json")
    lat = L("results/tables/explanation_latency_clean.json")
    tr = L("results/tables/transfer_cic_to_nsm.json")
    abl = L("results/tables/ablation_results_clean_rv_phase2design.json")
    ea = L("results/tables/error_analysis_clean.json")
    ut = L("results/tables/explanation_utility.json")
    af = L("results/tables/ablation_faithfulness.json")
    cal = L("results/tables/calibration.json")
    h2h = L("revision_results/faithfulness_h2h.json")
    fgi = L("results/tables/fold_group_importance_clean_rv.json")
    classes = cv["class_names"]
    rare = ("Heartbleed", "Infiltration")

    # ── T1 datasets ─────────────────────────────────────────────────────
    md.append("## Table 1. Datasets after cleaning\n")
    c13 = cnt["counts_13_class_merged"]
    rows = [["CIC-IDS2017", f"{cnt['total_rows']:,}", "13", "61", "5-fold stratified; scaler, SMOTE, validation inside the fold",
             f"BENIGN {c13['BENIGN']:,}; Heartbleed {c13['Heartbleed']}, Infiltration {c13['Infiltration']}"]]
    nf = nsm["folds"]
    n_nsm = sum(f["n_test_real"] for f in nf)
    rows.append(["NSM-IDS-2026 v1.2", f"{n_nsm:,}", "10", str(sum(nsm['protocol']['groups'].values())),
                 "4 session-grouped folds (3 sessions each); one training session held out for validation",
                 "split fragments excluded, exact duplicates removed"])
    rows.append(["UNSW-NB15", "257,673", "2 (normal/attack)", "68", "5-fold stratified, fold-internal SMOTE", "official train+test concatenated"])
    md.append(tbl(["Dataset", "Flows", "Classes", "Features", "Evaluation", "Notes"], rows))
    N["cic_rows"] = cnt["total_rows"]; N["nsm_rows"] = n_nsm
    for k, v in c13.items():
        N[f"cic_count_{k}"] = v

    # ── T2 CIC main ─────────────────────────────────────────────────────
    md.append("\n## Table 2. CIC-IDS2017, 5-fold cross-validation, every model with a validation-calibrated detection threshold\n")
    S = cv["summary"]
    rows = []
    for m in ORDER:
        s = S[m]; pf = cv["per_fold_results"][m]
        m11 = macro_big(pf, classes, rare)
        am = s.get("mean_macro_f1_argmax", np.mean([r.get("test_macro_f1_at_0.5", np.nan) for r in pf]))
        rows.append([DISP[m], f"{s['mean_macro_f1']:.3f} ± {s['std_macro_f1']:.3f}", f"{m11.mean():.3f} ± {m11.std():.3f}",
                     f"{am:.3f}", f"{s['binary']['recall']['mean']:.4f}", f"{100*s['binary']['fpr']['mean']:.2f}%",
                     f"{s['mean_macro_roc_auc_ovr']:.4f}"])
        N[f"cic_macro13_{m}"] = s["mean_macro_f1"]; N[f"cic_macro13_std_{m}"] = s["std_macro_f1"]
        N[f"cic_macro11_{m}"] = float(m11.mean()); N[f"cic_macro11_std_{m}"] = float(m11.std())
        N[f"cic_argmax_{m}"] = float(am); N[f"cic_fpr_{m}"] = s["binary"]["fpr"]["mean"]; N[f"cic_dr_{m}"] = s["binary"]["recall"]["mean"]
    if deep:
        rows.append([DISP["Deep-IDS"], f"{deep['mean_macro_f1']:.3f} ± {deep['std_macro_f1']:.3f}", "—", "—", "—", "—", "—"])
        N["cic_macro13_deepids"] = deep["mean_macro_f1"]
    md.append(tbl(["Model", "Macro-F1, 13 classes", "Macro-F1, 11 classes (≥389 test flows/fold)", "Macro-F1 at argmax",
                   "Detection rate", "False-positive rate", "Macro ROC-AUC"], rows))
    md.append("\nThe two rarest classes (Heartbleed 2–3 and Infiltration 7–8 test flows per fold) dominate the fold-to-fold variance of the 13-class macro-F1; the 11-class column excludes them.\n")

    # ── T3 paired tests ──────────────────────────────────────────────────
    md.append("## Table 3. Paired comparison with HiT-IDS over the five folds (13-class macro-F1; two-sided paired t-test, t-based 95% CI)\n")
    rows = []
    hit = np.array(S["HiT-IDS (Cascade)"]["per_fold_f1"])
    for m in ["xgboost", "random_forest", "bilstm", "vanilla_transformer"]:
        t = cv["statistical_tests"][m]
        rows.append([DISP[m], f"{t['delta_mean']:+.3f}", f"[{t['delta_ci'][0]:+.3f}, {t['delta_ci'][1]:+.3f}]", f"{t['cohens_d']:.2f}", f"{t['p_value']:.3f}"])
        N[f"cic_delta_{m}"] = t["delta_mean"]; N[f"cic_p_{m}"] = t["p_value"]
    if dstat:
        rows.append([DISP["Deep-IDS"], f"{dstat['delta_mean']:+.3f}", f"[{dstat['ci_95'][0]:+.3f}, {dstat['ci_95'][1]:+.3f}]" if "ci_95" in dstat else "—",
                     f"{dstat['cohens_d']:.2f}", f"{dstat['paired_t_test']['p_value_two_sided']:.3f}" if "paired_t_test" in dstat else "—"])
    md.append(tbl(["Baseline", "ΔF1 (HiT-IDS − baseline)", "95% CI", "Cohen's d", "p"], rows))

    # ── T4 per-class ─────────────────────────────────────────────────────
    md.append("\n## Table 4. Per-class F1 on CIC-IDS2017 (mean over folds; HiT-IDS also with precision and recall)\n")
    rows = []
    for c in classes:
        h = S["HiT-IDS (Cascade)"]["per_class"][c]
        rows.append([c, f"{sum(h['support_per_fold']):,}", f"{h['precision_mean']:.3f}", f"{h['recall_mean']:.3f}",
                     f"{h['f1_mean']:.3f} ± {h['f1_std']:.3f}"] +
                    [f"{S[m]['per_class'][c]['f1_mean']:.3f}" for m in ["xgboost", "random_forest", "bilstm", "vanilla_transformer"]])
        N[f"cic_hit_f1_{c}"] = h["f1_mean"]
    md.append(tbl(["Class", "Test flows (5 folds)", "HiT-IDS P", "HiT-IDS R", "HiT-IDS F1", "XGBoost", "RF", "BiLSTM", "Flat Tr."], rows))

    # ── T5 NSM ───────────────────────────────────────────────────────────
    md.append("\n## Table 5. NSM-IDS-2026 v1.2, session-grouped 4-fold cross-validation (calibrated thresholds)\n")
    Sn = nsm["summary"]; rows = []
    for m in ORDER:
        s = Sn[m]; t = nsm["statistical_tests"].get(m)
        rows.append([DISP[m], f"{s['mean_macro_f1']:.3f} ± {s['std_macro_f1']:.3f}", f"{s['mean_macro_f1_argmax']:.3f}",
                     f"{s['binary']['recall']['mean']:.4f}", f"{100*s['binary']['fpr']['mean']:.2f}%",
                     (f"{t['delta_mean']:+.3f} (p = {t['p_value']:.2f})" if t else "—")])
        N[f"nsm_macro_{m}"] = s["mean_macro_f1"]; N[f"nsm_macro_std_{m}"] = s["std_macro_f1"]; N[f"nsm_fpr_{m}"] = s["binary"]["fpr"]["mean"]
    md.append(tbl(["Model", "Macro-F1, 10 classes", "at argmax", "Detection rate", "False-positive rate", "ΔF1 vs HiT-IDS"], rows))
    md.append("\nPer-class HiT-IDS F1 (mean over folds): " + ", ".join(f"{c} {v['f1_mean']:.3f}" for c, v in Sn["HiT-IDS (Cascade)"]["per_class"].items()) + ".\n")
    for c, v in Sn["HiT-IDS (Cascade)"]["per_class"].items():
        N[f"nsm_hit_f1_{c}"] = v["f1_mean"]

    # ── T6 UNSW ──────────────────────────────────────────────────────────
    md.append("## Table 6. UNSW-NB15, binary, 5-fold cross-validation\n")
    rows = [[DISP.get(m, m), f"{unsw['summary'][m]['mean_macro_f1']:.3f} ± {unsw['summary'][m]['std_macro_f1']:.3f}", f"{unsw['summary'][m]['mean_accuracy']:.3f}"]
            for m in ["random_forest", "xgboost", "bilstm", "HiT-IDS", "vanilla_transformer"] if m in unsw["summary"]]
    md.append(tbl(["Model", "Macro-F1", "Accuracy"], rows))
    for m in unsw["summary"]:
        N[f"unsw_macro_{m}"] = unsw["summary"][m]["mean_macro_f1"]

    # ── T7 faithfulness ──────────────────────────────────────────────────
    R, Rn = fa["results"], fn["results"]
    md.append(f"\n## Table 7. Faithfulness of four attributions on the same model: CIC-IDS2017 deployment model ({fa['n_samples']} flows) and NSM-IDS-2026 fold-1 model ({fn['n_samples']} flows)\n")
    rows = []
    for m, lab in MET:
        rows.append([lab] + [f"{R[k][m]['mean']:.3f}" for k in ("attention", "gradient", "shap", "random")] +
                    [f"{Rn[k][m]['mean']:.3f}" for k in ("attention", "gradient", "shap", "random")])
    md.append(tbl(["Metric", "CIC rollout", "CIC gradient", "CIC GradientSHAP", "CIC random", "NSM rollout", "NSM gradient", "NSM GradientSHAP", "NSM random"], rows))
    for ds, RR in (("cic", R), ("nsm", Rn)):
        N[f"faith_{ds}_attn_gt_random"] = wins(RR, "attention", "random"); N[f"faith_{ds}_attn_gt_grad"] = wins(RR, "attention", "gradient")
        N[f"faith_{ds}_attn_gt_shap"] = wins(RR, "attention", "shap"); N[f"faith_{ds}_shap_gt_attn"] = wins(RR, "shap", "attention")
    md.append(f"\nRollout better than random on {N['faith_cic_attn_gt_random']}/11 (CIC) and {N['faith_nsm_attn_gt_random']}/11 (NSM) metrics; "
              f"better than input gradient on {N['faith_cic_attn_gt_grad']}/11 and {N['faith_nsm_attn_gt_grad']}/11; "
              f"better than GradientSHAP on {N['faith_cic_attn_gt_shap']}/11 and {N['faith_nsm_attn_gt_shap']}/11. "
              f"Holm-corrected attention-vs-random tests: {sh['families']['P2_faithfulness_CIC2017']['n_sig_holm']}/11 significant on CIC-IDS2017, "
              f"{sh['families']['E2_faithfulness_NB15']['n_sig_holm']}/11 on UNSW-NB15.\n")
    # NSM tests
    for k, rows_ in fn["tests"].items():
        N[f"nsm_{k}_sig_wins"] = sum(r["attention_better"] and r["p_two_sided"] < 0.05 for r in rows_)
    # utility + hier vs flat + phase2
    u5 = ut["results"]["k=5"]
    N["utility_k5_attn_drop"] = u5["attn_drop_global"]; N["utility_k5_random_drop"] = u5["random_drop_mean"]; N["utility_k5_p"] = u5["p_value"]
    N["hier_vs_flat_sig"] = af["n_significant_hier_vs_flat"]
    if fa2:
        R2 = fa2["results"]["attention"]
        N["phase2_faith_main_better"] = sum((R["attention"][m]["mean"] < R2[m]["mean"]) if m in LOWER else (R["attention"][m]["mean"] > R2[m]["mean"]) for m, _ in MET)
    if h2h:
        N["h2h_summary"] = h2h.get("summary", {}) if isinstance(h2h.get("summary"), dict) else str(h2h.get("summary"))[:200]

    # ── T8 stability ─────────────────────────────────────────────────────
    md.append("## Table 8. Stability of attributions across independently trained fold models (CIC-IDS2017: 5 folds; NSM-IDS-2026: 4 session folds)\n")
    rows = []
    for m in st["methods"]:
        a, b = st["methods"][m], stn["methods"].get(m)
        rows.append([m, f"{a['interfold_rho_global']['mean']:.2f}", f"{a['per_class_rho_mean']:.2f}", f"{100*a['dominant_agreement_mean']:.0f}%",
                     f"{b['interfold_rho_global']['mean']:.2f}" if b else "—", f"{b['per_class_rho_mean']:.2f}" if b else "—",
                     f"{100*b['dominant_agreement_mean']:.0f}%" if b else "—"])
        key = m.lower().replace(" ", "_").replace("-", "_")
        N[f"stab_cic_global_{key}"] = a["interfold_rho_global"]["mean"]; N[f"stab_cic_dom_{key}"] = a["dominant_agreement_mean"]
        if b:
            N[f"stab_nsm_global_{key}"] = b["interfold_rho_global"]["mean"]; N[f"stab_nsm_dom_{key}"] = b["dominant_agreement_mean"]
    md.append(tbl(["Attribution", "CIC: inter-fold ρ (global)", "CIC: per-class ρ", "CIC: dominant-group agreement",
                   "NSM: inter-fold ρ (global)", "NSM: per-class ρ", "NSM: dominant-group agreement"], rows))

    # ── T9 XCS ───────────────────────────────────────────────────────────
    md.append("\n## Table 9. Agreement of rollout with GradientSHAP (500 flows, CIC deployment model)\n")
    P = xcs["pairs"]
    rows = [[lab, f"{P[k]['mean_rho']:.2f} ± {P[k]['std_rho']:.2f}", f"{P[k]['top5_overlap']:.2f}", f"{P[k]['top10_overlap']:.2f}"]
            for k, lab in (("rollout_vs_shap_predclass", "attention rollout"), ("attn_x_grad_vs_shap_predclass", "attention × |gradient|"),
                           ("gradient_only_vs_shap_predclass", "|gradient| alone"))]
    md.append(tbl(["Attribution", "Spearman ρ vs SHAP", "top-5 overlap", "top-10 overlap"], rows))
    N["xcs_rollout_rho"] = P["rollout_vs_shap_predclass"]["mean_rho"]; N["xcs_rollout_top10"] = P["rollout_vs_shap_predclass"]["top10_overlap"]

    # ── T10 latency ──────────────────────────────────────────────────────
    md.append("\n## Table 10. Explanation latency per flow, one protocol (100 flows, median of 5 repeats, RTX 5090)\n")
    rows = []
    for k, v in lat["results_ms_per_sample"].items():
        s, b = v.get("single_ms_per_sample", {}).get("median"), v.get("batched_ms_per_sample", {}).get("median")
        rows.append([k, v["type"], f"{s:.2f}" if s else "—", f"{b:.3f}" if b else "—"])
        key = k.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        if s:
            N[f"lat_single_{key}"] = s; N[f"lat_batched_{key}"] = b
    md.append(tbl(["Method", "Type", "One flow (ms)", "Batched (ms/flow)"], rows))

    # ── T11 transfer ─────────────────────────────────────────────────────
    md.append(f"\n## Table 11. Zero-shot transfer CIC-IDS2017 → NSM-IDS-2026 on {len(tr['portable_features'])} portable features (attack vs benign on all {tr['n_nsm_flows']:,} NSM flows)\n")
    rows = []
    for nm in ("source_scaled", "target_scaled"):
        for m in ("HiT-IDS", "random_forest", "xgboost"):
            b = tr["normalisation"][nm][m]["at_cic_threshold"]
            rows.append([nm.replace("_", "-"), DISP.get(m, m), f"{b['f1']:.3f}", f"{b['detection_rate']:.3f}", f"{100*b['fpr']:.2f}%", f"{b['roc_auc']:.3f}"])
            N[f"transfer_{nm}_{m}_f1"] = b["f1"]; N[f"transfer_{nm}_{m}_auc"] = b["roc_auc"]; N[f"transfer_{nm}_{m}_dr"] = b["detection_rate"]
    md.append(tbl(["Normalisation", "Model (trained on CIC-IDS2017)", "Binary F1", "Detection rate", "FPR", "ROC-AUC"], rows))
    pc = tr["normalisation"]["source_scaled"]["HiT-IDS"]["per_nsm_class"]
    md.append("\nHiT-IDS per-class detection rate on NSM (source-scaled): " + ", ".join(
        f"{c} {v.get('detection_rate', v.get('false_positive_rate')):.2f}" for c, v in pc.items()) + ". In-domain CIC validation binary F1 of the portable model: "
        f"{tr['cic_in_domain_validation']['binary']['f1']:.3f}.\n")
    for c, v in pc.items():
        N[f"transfer_dr_{c}"] = v.get("detection_rate", v.get("false_positive_rate"))

    # ── T12 ablation ─────────────────────────────────────────────────────
    md.append("## Table 12. Ablation on CIC-IDS2017 (5-fold, argmax rule; reference = design including the entropy fine-tuning stage)\n")
    rows = [["—", "Reference (with entropy fine-tuning)", f"{abl['reference']['mean_macro_f1']:.3f}", "—"]]
    for k in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        v = abl[k]; rows.append([k, v["name"], f"{v['mean_macro_f1']:.3f} ± {v['std_macro_f1']:.3f}", f"{v['delta_f1']:+.3f}"])
        N[f"abl_{k}_delta"] = v["delta_f1"]
    md.append(tbl(["ID", "Variant", "Macro-F1", "ΔF1"], rows))

    # ── T13 deployment ───────────────────────────────────────────────────
    md.append("\n## Table 13. Deployment model (70/15/15 split) at the argmax rule and at the validation-calibrated threshold\n")
    rows = []
    for k, lab in (("argmax_0.5", "argmax"), ("calibrated", f"calibrated (t* = {dep['stage1_threshold']:.3f})")):
        r = dep["test"][k]; m11 = np.mean([r["per_class"][c]["f1"] for c in classes if c not in rare])
        rows.append([lab, f"{r['macro_f1']:.3f}", f"{m11:.3f}", f"{r['binary']['recall']:.4f}", f"{100*r['binary']['fpr']:.3f}%",
                     f"{r['per_class']['Bot']['f1']:.3f}", f"{r['per_class']['Infiltration']['recall']:.2f} ({r['per_class']['Infiltration']['support']} flows)"])
        N[f"deploy_{k}_macro13"] = r["macro_f1"]; N[f"deploy_{k}_macro11"] = float(m11); N[f"deploy_{k}_fpr"] = r["binary"]["fpr"]
    md.append(tbl(["Rule", "Macro-F1 (13)", "Macro-F1 (11)", "Detection rate", "FPR", "Bot F1", "Infiltration recall"], rows))
    N["deploy_threshold"] = dep["stage1_threshold"]; N["deploy_n_test"] = dep["n_test"]
    N["ece_hit"] = cal["CIC-IDS2017_HiT-IDS"]["ece"]; N["ece_rf"] = cal["CIC-IDS2017_RF"]["ece"]; N["ece_xgb"] = cal["CIC-IDS2017_XGBoost"]["ece"]

    # ── T14 errors ───────────────────────────────────────────────────────
    md.append("\n## Table 14. Largest error groups of the deployment model (calibrated) and their group-attribution divergence\n")
    rows = [[v["desc"], f"{v['n']:,}", f"{v['confidence_mean']:.2f}"] + [f"{v['group_attention_delta'][g]:+.2f}" for g in v["group_attention_delta"]]
            for v in ea["error_groups"].values()]
    md.append(tbl(["Error group", "n", "mean conf.", "Δ temporal", "Δ volume", "Δ protocol", "Δ behavioral"], rows))
    N["errors_total"] = ea["totals"]["misclassified"]; N["errors_n"] = ea["totals"]["n"]
    N["group_stability_rho"] = fgi["interfold_spearman_global72"]["mean"]

    open("manuscript/tables_scirep.md", "w").write("\n".join(md) + "\n")
    json.dump(N, open("manuscript/numbers_scirep.json", "w"), indent=2, default=float)
    print("  wrote manuscript/tables_scirep.md and numbers_scirep.json")


if __name__ == "__main__":
    main()

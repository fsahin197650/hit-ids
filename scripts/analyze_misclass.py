#!/usr/bin/env python3
"""
Analyze misclass_artifacts.npz for Section 5.9 error analysis.
Computes per-error-group attention divergence statistics.
"""
import numpy as np
import json
import sys

ART = sys.argv[1] if len(sys.argv) > 1 else "results/tables/misclass_artifacts_clean.npz"
data = np.load(ART, allow_pickle=True)
report = {"artifacts": ART, "error_groups": {}, "stage1": {}, "confidence": {}}
y_true = data["y_true"]
y_pred = data["y_pred"]
attn_group = data["attn_group"]  # (N, 4)
attn_feat = data["attn_feat"]    # (N, 72)
cm = data["confusion_matrix"]
group_names = list(data["group_names"])
class_names = list(data["class_names"])

misclass_mask = y_true != y_pred
print(f"Total: {len(y_true)}, Misclassified: {misclass_mask.sum()}")

# ── Define error groups ──────────────────────────────────────────────────
# Error groups = the 6 largest off-diagonal confusion pairs (data-driven, not hand-picked)
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
error_groups = {}
for flat in np.argsort(cm_off.ravel())[::-1][:6]:
    i, j = divmod(int(flat), cm.shape[0])
    if cm_off[i, j] == 0:
        break
    kind = ("false positive" if i == 0 else "false negative at Stage 1" if j == 0 else "sibling confusion")
    error_groups[f"G{len(error_groups)+1}-{class_names[i][:6]}->{class_names[j][:6]}"] = {
        "true": i, "pred": j, "desc": f"{class_names[i]} → {class_names[j]} ({kind})"}

print("\n" + "="*80)
print("ERROR GROUP ANALYSIS")
print("="*80)

for group_id, info in error_groups.items():
    mask = (y_true == info["true"]) & (y_pred == info["pred"])
    n = mask.sum()
    if n == 0:
        continue
    
    # Correctly classified samples of the TRUE class
    correct_mask = (y_true == info["true"]) & (y_pred == info["true"])
    n_correct = correct_mask.sum()
    
    print(f"\n--- {group_id}: {info['desc']} (n={n}) ---")
    
    # Group-level attention for misclassified
    mis_group = attn_group[mask].mean(axis=0)
    cor_group = attn_group[correct_mask].mean(axis=0) if n_correct > 0 else np.zeros(4)
    
    print(f"  Group attention (misclassified):  {dict(zip(group_names, [f'{v:.4f}' for v in mis_group]))}")
    print(f"  Group attention (correct):        {dict(zip(group_names, [f'{v:.4f}' for v in cor_group]))}")
    print(f"  Group attention divergence (mis - cor):")
    for i, gn in enumerate(group_names):
        delta = mis_group[i] - cor_group[i]
        arrow = "↑" if delta > 0 else "↓"
        print(f"    {gn:12s}: {delta:+.4f} {arrow}")
    
    # Top-3 feature divergences
    mis_feat = attn_feat[mask].mean(axis=0)
    cor_feat = attn_feat[correct_mask].mean(axis=0) if n_correct > 0 else np.zeros(72)
    feat_delta = np.abs(mis_feat - cor_feat)
    top3_idx = np.argsort(feat_delta)[::-1][:5]
    print(f"  Top-5 divergent features:")
    for idx in top3_idx:
        print(f"    Feature {idx:2d}: mis={mis_feat[idx]:.4f}, cor={cor_feat[idx]:.4f}, Δ={feat_delta[idx]:.4f}")
    report["error_groups"][group_id] = {
        "desc": info["desc"], "n": int(n), "n_correct_of_true_class": int(n_correct),
        "group_attention_misclassified": dict(zip(group_names, mis_group.round(4).tolist())),
        "group_attention_correct": dict(zip(group_names, cor_group.round(4).tolist())),
        "group_attention_delta": dict(zip(group_names, (mis_group - cor_group).round(4).tolist())),
        "top5_divergent_feature_idx": [int(i) for i in top3_idx]}

# ── Stage 1 analysis ─────────────────────────────────────────────────────
stage1 = data["stage1_pred"]
print("\n" + "="*80)
print("STAGE 1 ERROR ANALYSIS")
print("="*80)

# Stage 1 false negatives: attacks classified as BENIGN
for cls_id, cls_name in enumerate(class_names):
    if cls_id == 0:
        continue
    attack_mask = y_true == cls_id
    fn_mask = attack_mask & (stage1 == 0)  # Stage 1 said BENIGN
    n_attack = attack_mask.sum()
    n_fn = fn_mask.sum()
    if n_fn > 0:
        print(f"  {cls_name:20s}: {n_fn:5d}/{n_attack:6d} missed at Stage 1 ({100*n_fn/n_attack:.2f}%)")
    report["stage1"][cls_name] = {"missed": int(n_fn), "total": int(n_attack)}

# Stage 1 false positives: BENIGN classified as ATTACK
benign_mask = y_true == 0
fp_stage1 = benign_mask & (stage1 == 1)
print(f"\n  BENIGN → ATTACK (Stage 1 FP): {fp_stage1.sum()}/{benign_mask.sum()} ({100*fp_stage1.sum()/benign_mask.sum():.3f}%)")
report["stage1"]["BENIGN_false_positive"] = {"count": int(fp_stage1.sum()), "total": int(benign_mask.sum())}

# ── Confidence analysis ──────────────────────────────────────────────────
conf = data["confidence"]
print("\n" + "="*80)
print("CONFIDENCE ANALYSIS")
print("="*80)
report["confidence"] = {"correct_mean": float(conf[~misclass_mask].mean()), "correct_std": float(conf[~misclass_mask].std()),
                        "misclassified_mean": float(conf[misclass_mask].mean()), "misclassified_std": float(conf[misclass_mask].std())}
print(f"  Correct samples:       mean={conf[~misclass_mask].mean():.4f}, std={conf[~misclass_mask].std():.4f}")
print(f"  Misclassified samples: mean={conf[misclass_mask].mean():.4f}, std={conf[misclass_mask].std():.4f}")

# Per error group confidence
for group_id, info in error_groups.items():
    mask = (y_true == info["true"]) & (y_pred == info["pred"])
    n = mask.sum()
    if n > 0:
        print(f"  {group_id:20s}: mean_conf={conf[mask].mean():.4f}, std={conf[mask].std():.4f}, n={n}")

report["totals"] = {"n": int(len(y_true)), "misclassified": int(misclass_mask.sum())}
for gid in report["error_groups"]:
    info = error_groups[gid]; m = (y_true == info["true"]) & (y_pred == info["pred"])
    report["error_groups"][gid]["confidence_mean"] = float(conf[m].mean())
out = ART.replace("misclass_artifacts", "error_analysis").replace(".npz", ".json")
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n✓ Analysis complete! wrote {out}")

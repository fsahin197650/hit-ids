#!/usr/bin/env python3
"""A1 (Flat Transformer) diagnostic — run to find why F1=0.049."""
import torch, yaml, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate_cv import merge_labels_np, set_seed
from scripts.run_ablation import FlatTransformerIDS, create_ablation_model, train_flat_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data (same as ablation script)
splits_dir = config['paths']['splits']
train_data = torch.load(f'{splits_dir}/train.pt', weights_only=False)
val_data = torch.load(f'{splits_dir}/val.pt', weights_only=False)
test_data = torch.load(f'{splits_dir}/test.pt', weights_only=False)
X_all = torch.cat([train_data['X'], val_data['X'], test_data['X']]).numpy()
y_all = merge_labels_np(torch.cat([train_data['y'], val_data['y'], test_data['y']]).numpy())

# Use fold 1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(iter(skf.split(X_all, y_all)))
X_test = X_all[test_idx].astype(np.float32)
y_test = y_all[test_idx].astype(np.int64)

# Train/val split
set_seed(42)
n = len(train_idx)
idx = np.random.RandomState(42).permutation(n)
val_size = max(1, n // 10)
X_tr = X_all[train_idx[idx[val_size:]]].astype(np.float32)
y_tr = y_all[train_idx[idx[val_size:]]].astype(np.int64)
X_vl = X_all[train_idx[idx[:val_size]]].astype(np.float32)
y_vl = y_all[train_idx[idx[:val_size]]].astype(np.int64)

print(f"Train: {len(y_tr)}, Val: {len(y_vl)}, Test: {len(y_test)}")
print(f"Train label dist: {np.bincount(y_tr, minlength=13)}")

# --- Step 1: Train binary model (5 epochs only) ---
print("\n=== BINARY MODEL (5 epochs) ===")
y_tr_bin = (y_tr > 0).astype(np.int64)
y_vl_bin = (y_vl > 0).astype(np.int64)
print(f"Binary train: benign={np.sum(y_tr_bin==0)}, attack={np.sum(y_tr_bin==1)}")

bin_model, _ = create_ablation_model('A1', config, 2, device)
best_f1 = train_flat_model(bin_model, X_tr, y_tr_bin, X_vl, y_vl_bin,
                           config, device, max_epochs=5, patience=5, num_classes=2)
print(f"Binary val F1: {best_f1:.4f}")

# Check binary predictions on test
bin_model.eval()
bin_preds = []
with torch.no_grad():
    for i in range(0, len(X_test), 512):
        xb = torch.tensor(X_test[i:i+512], dtype=torch.float32).to(device)
        bp = bin_model(xb).argmax(1).cpu()
        bin_preds.append(bp)
bin_preds = torch.cat(bin_preds).numpy()
y_test_bin = (y_test > 0).astype(np.int64)
print(f"Binary test preds: 0(benign)={np.sum(bin_preds==0)}, 1(attack)={np.sum(bin_preds==1)}")
print(f"Binary test actual: 0(benign)={np.sum(y_test_bin==0)}, 1(attack)={np.sum(y_test_bin==1)}")
print(f"Binary test F1: {f1_score(y_test_bin, bin_preds, average='macro'):.4f}")

# --- Step 2: Check cascade eval logic ---
print("\n=== CASCADE EVAL CHECK ===")
final = torch.zeros(len(y_test), dtype=torch.long)
total_attacks = 0
batch_size = 512
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        end = min(i + batch_size, len(X_test))
        xb = torch.tensor(X_test[i:end], dtype=torch.float32).to(device)
        bp = bin_model(xb).argmax(1)
        mask = bp == 1
        n_atk = mask.sum().item()
        total_attacks += n_atk
        if mask.any():
            # Use dummy attack predictions (class 1) to verify indexing
            ap = torch.ones(n_atk, dtype=torch.long)  # all class 1
            global_idx = torch.arange(i, end)[mask.cpu()]
            final[global_idx] = ap

print(f"Total samples flagged as attack: {total_attacks}/{len(y_test)}")
print(f"Final pred distribution: {np.bincount(final.numpy(), minlength=13)}")
print(f"Cascade F1 (dummy attack=1): {f1_score(y_test, final.numpy(), average='macro', zero_division=0):.4f}")

print("\n=== DONE ===")

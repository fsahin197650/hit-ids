"""
PyTorch Dataset for HiT-IDS.

Provides grouped feature tensors for the hierarchical transformer.
Each sample returns 4 feature group tensors + label.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HiTIDSDataset(Dataset):
    """
    PyTorch Dataset that returns grouped features for HiT-IDS.

    Each __getitem__ call returns:
        groups: dict mapping group_name → tensor of shape (num_features_in_group,)
        label: integer class label

    Usage:
        >>> dataset = HiTIDSDataset.from_split("data/splits", "train")
        >>> groups, label = dataset[0]
        >>> groups['temporal'].shape  # (19,)
        >>> groups['volume'].shape    # (25,)
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        group_data: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            X: Full feature tensor, shape (N, total_features).
            y: Label tensor, shape (N,).
            group_data: Dict mapping group name → tensor (N, group_size).
            metadata: Optional metadata dict with group info.
        """
        self.X = X
        self.y = y
        self.group_data = group_data
        self.metadata = metadata or {}

        # Validate shapes
        assert len(self.X) == len(self.y), (
            f"X and y length mismatch: {len(self.X)} vs {len(self.y)}"
        )
        for name, tensor in self.group_data.items():
            assert len(tensor) == len(self.X), (
                f"Group '{name}' length mismatch: {len(tensor)} vs {len(self.X)}"
            )

    @classmethod
    def from_split(
        cls,
        splits_dir: str,
        split: str = "train",
    ) -> "HiTIDSDataset":
        """
        Load a dataset split from disk.

        Args:
            splits_dir: Path to the splits directory.
            split: One of 'train', 'val', 'test'.

        Returns:
            HiTIDSDataset instance.
        """
        splits_path = Path(splits_dir)
        split_file = splits_path / f"{split}.pt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                f"Run preprocessing first."
            )

        data = torch.load(split_file, map_location="cpu", weights_only=False)

        # Load metadata if available
        metadata = {}
        metadata_file = splits_path / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

        return cls(
            X=data["X"],
            y=data["y"],
            group_data=data.get("groups", {}),
            metadata=metadata,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (group_dict, label) where group_dict maps
            group_name → feature tensor.
        """
        groups = {
            name: tensor[idx] for name, tensor in self.group_data.items()
        }
        label = self.y[idx]
        return groups, label

    @property
    def num_classes(self) -> int:
        """Number of unique classes."""
        return len(torch.unique(self.y))

    @property
    def group_names(self) -> List[str]:
        """Names of feature groups."""
        return list(self.group_data.keys())

    @property
    def group_sizes(self) -> List[int]:
        """Number of features per group."""
        return [tensor.shape[1] for tensor in self.group_data.values()]

    @property
    def class_counts(self) -> Dict[int, int]:
        """Count of samples per class."""
        unique, counts = torch.unique(self.y, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def get_class_weights(self, method: str = "inverse") -> torch.Tensor:
        """
        Compute class weights for imbalanced learning.

        Args:
            method: 'inverse' (1/count) or 'effective' (effective number).

        Returns:
            Tensor of shape (num_classes,) with per-class weights.
        """
        counts = self.class_counts
        num_classes = max(counts.keys()) + 1
        total = sum(counts.values())

        weights = torch.zeros(num_classes, dtype=torch.float32)

        if method == "inverse":
            for cls_id, count in counts.items():
                weights[cls_id] = total / (num_classes * count)
        elif method == "effective":
            beta = 0.9999
            for cls_id, count in counts.items():
                effective = (1.0 - beta ** count) / (1.0 - beta)
                weights[cls_id] = 1.0 / effective
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        Returns:
            Tensor of shape (N,) with per-sample weights.
        """
        class_weights = self.get_class_weights()
        return class_weights[self.y]


def collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Stacks group tensors and labels into batches.

    Args:
        batch: List of (group_dict, label) tuples.

    Returns:
        Tuple of (batched_groups, batched_labels).
    """
    group_names = list(batch[0][0].keys())

    batched_groups: Dict[str, torch.Tensor] = {}
    for name in group_names:
        batched_groups[name] = torch.stack([item[0][name] for item in batch])

    labels = torch.stack([item[1] for item in batch])

    return batched_groups, labels


def create_dataloaders(
    splits_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, val, and test splits.

    Args:
        splits_dir: Path to splits directory.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        use_weighted_sampler: Use weighted sampling for training.

    Returns:
        Dict mapping split name → DataLoader.
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = HiTIDSDataset.from_split(splits_dir, split)

        if split == "train" and use_weighted_sampler:
            sample_weights = dataset.get_sample_weights()
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True,
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )

    return loaders


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Create dummy data for testing
    print("Testing HiTIDSDataset with dummy data...")

    N = 500
    group_data = {
        "temporal": torch.randn(N, 19),
        "volume": torch.randn(N, 25),
        "protocol": torch.randn(N, 15),
        "behavioral": torch.randn(N, 13),
    }
    X = torch.cat(list(group_data.values()), dim=1)
    y = torch.randint(0, 15, (N,))

    dataset = HiTIDSDataset(X=X, y=y, group_data=group_data)
    print(f"Dataset size: {len(dataset)}")
    print(f"Group names: {dataset.group_names}")
    print(f"Group sizes: {dataset.group_sizes}")
    print(f"Num classes: {dataset.num_classes}")
    print(f"Class counts: {dataset.class_counts}")

    # Test __getitem__
    groups, label = dataset[0]
    for name, tensor in groups.items():
        print(f"  {name}: shape={tensor.shape}")
    print(f"  label: {label}")

    # Test DataLoader with collate_fn
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    batch_groups, batch_labels = next(iter(loader))
    print(f"\nBatch test:")
    for name, tensor in batch_groups.items():
        print(f"  {name}: shape={tensor.shape}")
    print(f"  labels: shape={batch_labels.shape}")

    print("\n✓ All tests passed!")

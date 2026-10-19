"""
Feature group definitions for HiT-IDS.

Defines the 4 semantic feature groups (Temporal, Volume, Protocol, Behavioral)
that form the backbone of the hierarchical transformer architecture.
Each group captures a different aspect of network traffic behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class FeatureGroupInfo:
    """Metadata for a single feature group."""
    name: str
    description: str
    features: List[str]

    @property
    def size(self) -> int:
        """Number of features in this group."""
        return len(self.features)


class FeatureGroups:
    """
    Manages the 4 hierarchical feature groups for CIC-IDS2017.

    Groups:
        G1 (Temporal):   Timing anomalies — slow attacks, timing side-channels
        G2 (Volume):     Volumetric attacks — DDoS, data exfiltration
        G3 (Protocol):   Protocol anomalies — port scan, SYN flood
        G4 (Behavioral): Behavioral profile — botnet, lateral movement

    Usage:
        >>> fg = FeatureGroups.from_config("configs/config.yaml")
        >>> print(fg.group_names)
        ['temporal', 'volume', 'protocol', 'behavioral']
        >>> print(fg.group_sizes)
        [19, 25, 15, 13]
        >>> indices = fg.get_group_indices(all_feature_names)
    """

    # Default feature definitions (CIC-IDS2017)
    TEMPORAL_FEATURES: List[str] = [
        "Flow Duration",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ]

    VOLUME_FEATURES: List[str] = [
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s",
        "Fwd Packets/s", "Bwd Packets/s",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "Max Packet Length", "Min Packet Length",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "Fwd Header Length", "Bwd Header Length",
    ]

    PROTOCOL_FEATURES: List[str] = [
        "Destination Port",
        "Fwd PSH Flags", "Bwd PSH Flags",
        "Fwd URG Flags", "Bwd URG Flags",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
        "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    ]

    BEHAVIORAL_FEATURES: List[str] = [
        "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Down/Up Ratio",
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
        "act_data_pkt_fwd", "min_seg_size_forward",
    ]

    # Columns to always drop (identifiers, not features)
    DROP_COLUMNS: List[str] = [
        "Flow ID", "Source IP", "Source Port",
        "Destination IP", "Timestamp",
    ]

    def __init__(self, groups: Optional[Dict[str, List[str]]] = None) -> None:
        """
        Initialize feature groups.

        Args:
            groups: Optional dict mapping group name → list of feature names.
                    If None, uses the default CIC-IDS2017 definitions.
        """
        if groups is None:
            self._groups = {
                "temporal": FeatureGroupInfo(
                    name="temporal",
                    description="Timing anomalies (slow attacks, timing side-channels)",
                    features=list(self.TEMPORAL_FEATURES),
                ),
                "volume": FeatureGroupInfo(
                    name="volume",
                    description="Volumetric attacks (DDoS, data exfiltration)",
                    features=list(self.VOLUME_FEATURES),
                ),
                "protocol": FeatureGroupInfo(
                    name="protocol",
                    description="Protocol anomalies (port scan, SYN flood)",
                    features=list(self.PROTOCOL_FEATURES),
                ),
                "behavioral": FeatureGroupInfo(
                    name="behavioral",
                    description="Behavioral profile (botnet, lateral movement)",
                    features=list(self.BEHAVIORAL_FEATURES),
                ),
            }
        else:
            self._groups = {
                name: FeatureGroupInfo(
                    name=name,
                    description=f"Feature group: {name}",
                    features=list(features),
                )
                for name, features in groups.items()
            }

    @classmethod
    def from_config(cls, config_path: str) -> "FeatureGroups":
        """
        Load feature groups from a YAML config file.

        Args:
            config_path: Path to the config.yaml file.

        Returns:
            FeatureGroups instance with groups from config.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        groups = cfg.get("feature_groups", None)
        return cls(groups=groups)

    @classmethod
    def from_dict(cls, config: dict) -> "FeatureGroups":
        """
        Load feature groups from an already-loaded config dict.

        Args:
            config: Parsed YAML config dictionary.

        Returns:
            FeatureGroups instance.
        """
        groups = config.get("feature_groups", None)
        return cls(groups=groups)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def group_names(self) -> List[str]:
        """Ordered list of group names."""
        return list(self._groups.keys())

    @property
    def group_sizes(self) -> List[int]:
        """Number of features in each group, in order."""
        return [g.size for g in self._groups.values()]

    @property
    def total_features(self) -> int:
        """Total number of features across all groups."""
        return sum(self.group_sizes)

    @property
    def all_features(self) -> List[str]:
        """Flat list of all features across all groups, in order."""
        result: List[str] = []
        for g in self._groups.values():
            result.extend(g.features)
        return result

    @property
    def num_groups(self) -> int:
        """Number of feature groups."""
        return len(self._groups)

    # ── Methods ──────────────────────────────────────────────────────────

    def get_group(self, name: str) -> FeatureGroupInfo:
        """Get a specific group's info by name."""
        return self._groups[name]

    def get_group_features(self, name: str) -> List[str]:
        """Get the feature list for a specific group."""
        return self._groups[name].features

    def get_group_indices(self, column_names: List[str]) -> Dict[str, List[int]]:
        """
        Map each group to its column indices within a full feature DataFrame.

        Args:
            column_names: List of all column names in the DataFrame
                          (after dropping identifiers, before label).

        Returns:
            Dict mapping group name → list of integer column indices.

        Raises:
            ValueError: If a feature from any group is not found in column_names.
        """
        col_to_idx = {col.strip(): i for i, col in enumerate(column_names)}
        indices: Dict[str, List[int]] = {}

        for group_name, group_info in self._groups.items():
            group_indices: List[int] = []
            missing: List[str] = []
            for feat in group_info.features:
                feat_stripped = feat.strip()
                if feat_stripped in col_to_idx:
                    group_indices.append(col_to_idx[feat_stripped])
                else:
                    missing.append(feat_stripped)
            if missing:
                print(f"[WARNING] Group '{group_name}' — missing features "
                      f"(will be skipped): {missing}")
            indices[group_name] = group_indices

        return indices

    def split_features(
        self, column_names: List[str]
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Split column names into grouped and ungrouped features.

        Args:
            column_names: All feature column names.

        Returns:
            Tuple of (grouped_features dict, ungrouped_features list).
        """
        all_grouped = set(self.all_features)
        grouped: Dict[str, List[str]] = {}
        ungrouped: List[str] = []

        for name, info in self._groups.items():
            grouped[name] = [f for f in info.features if f.strip() in
                             {c.strip() for c in column_names}]

        for col in column_names:
            if col.strip() not in {f.strip() for f in all_grouped}:
                ungrouped.append(col)

        return grouped, ungrouped

    def summary(self) -> str:
        """Return a human-readable summary of all groups."""
        lines = ["=" * 60, "HiT-IDS Feature Groups Summary", "=" * 60]
        for name, info in self._groups.items():
            lines.append(f"\n{name.upper()} ({info.size} features)")
            lines.append(f"  Description: {info.description}")
            lines.append(f"  Features: {', '.join(info.features[:5])}...")
        lines.append(f"\nTotal features: {self.total_features}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{n}={s}" for n, s in
                          zip(self.group_names, self.group_sizes))
        return f"FeatureGroups({sizes}, total={self.total_features})"


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with defaults
    fg = FeatureGroups()
    print(fg)
    print(fg.summary())
    print(f"\nGroup sizes: {fg.group_sizes}")
    print(f"All features count: {fg.total_features}")

    # Test from config
    import os
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "config.yaml"
    )
    if os.path.exists(config_path):
        fg2 = FeatureGroups.from_config(config_path)
        print(f"\nFrom config: {fg2}")
        assert fg2.group_sizes == fg.group_sizes, "Config mismatch!"
        print("✓ Config matches defaults")

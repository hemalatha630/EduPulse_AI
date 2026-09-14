"""Temporal Sequence Dataset and Data Leakage Prevention Module for EduPulse AI.

Provides:
1. Target observable behaviour class encoding (6 canonical classes).
2. Track-grouped train/validation/test splitting (strictly prevents overlapping
   window data leakage between train and test sets).
3. Class distribution computation across splits.
4. Class-weight computation strictly on the training set for loss imbalance handling.
5. PyTorch SupervisedSequenceDataset and DataLoader factory.
"""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.behaviour.behaviour_labels import TARGET_BEHAVIOUR_CLASSES
from src.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
)

# Canonical 6 target observable behaviour classes
TARGET_CLASSES: List[str] = list(TARGET_BEHAVIOUR_CLASSES)
NUM_TARGET_CLASSES: int = len(TARGET_CLASSES)

CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(TARGET_CLASSES)}


class SupervisedSequenceDataset(Dataset):
    """PyTorch Dataset yielding (sequence_tensor, target_label, metadata).

    Shape of sequence_tensor: (sequence_length, feature_dimension).
    target_label: int64 scalar tensor in [0, NUM_TARGET_CLASSES - 1].
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        metadata_df: Optional[pd.DataFrame] = None,
    ):
        if len(sequences) != len(labels):
            raise ValueError(
                f"Sequences length ({len(sequences)}) must match labels length ({len(labels)})."
            )
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()
        self.metadata_df = metadata_df.reset_index(drop=True) if metadata_df is not None else None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self.sequences[idx]
        target = self.labels[idx]
        meta = {}
        if self.metadata_df is not None and idx < len(self.metadata_df):
            meta = self.metadata_df.iloc[idx].to_dict()
        return sample, target, meta


@dataclass
class TemporalDatasetSplit:
    """Container holding dataset splits, track groupings, and class distributions."""

    train_dataset: SupervisedSequenceDataset
    val_dataset: SupervisedSequenceDataset
    test_dataset: SupervisedSequenceDataset
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    train_tracks: List[int]
    val_tracks: List[int]
    test_tracks: List[int]
    train_distribution: Dict[str, int]
    val_distribution: Dict[str, int]
    test_distribution: Dict[str, int]
    class_weights: torch.Tensor
    total_sequences: int
    excluded_uncertain_count: int


def encode_behaviour_labels(
    metadata_df: pd.DataFrame,
    target_col: str = "dominant_behaviour",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Map string behaviour labels in metadata to target class integers.

    Args:
        metadata_df: DataFrame containing sequence metadata.
        target_col: Column name containing behaviour strings.

    Returns:
        Tuple of:
        - valid_mask: Boolean array indicating sequences with valid target classes.
        - integer_labels: Array of integer class labels for valid sequences.
        - excluded_count: Count of ambiguous/unknown sequences excluded.
    """
    labels = []
    valid_mask = []
    excluded_count = 0

    for val in metadata_df[target_col]:
        label_str = str(val).strip()
        if label_str in CLASS_TO_IDX:
            labels.append(CLASS_TO_IDX[label_str])
            valid_mask.append(True)
        else:
            # Check for close matches or unknown
            matched = False
            for target_name, idx in CLASS_TO_IDX.items():
                if target_name.lower() == label_str.lower():
                    labels.append(idx)
                    valid_mask.append(True)
                    matched = True
                    break
            if not matched:
                valid_mask.append(False)
                excluded_count += 1

    return np.array(valid_mask, dtype=bool), np.array(labels, dtype=np.int64), excluded_count


def compute_training_class_weights(
    train_labels: np.ndarray,
    num_classes: int = NUM_TARGET_CLASSES,
) -> torch.Tensor:
    """Calculate balanced class weights computed strictly from the training set only.

    Formula: weight[c] = N_train / (N_classes_present * count[c])
    Never accesses validation or test samples to prevent distribution leakage.

    Args:
        train_labels: 1D NumPy array of training class indices.
        num_classes: Total number of classes (default: 6).

    Returns:
        torch.FloatTensor of shape (num_classes,).
    """
    if len(train_labels) == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    counts = np.bincount(train_labels, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)

    present_classes = np.where(counts > 0)[0]
    total_samples = len(train_labels)

    for c in present_classes:
        weights[c] = total_samples / (len(present_classes) * counts[c])

    # For classes not present in training set, set neutral weight 1.0
    absent_classes = np.where(counts == 0)[0]
    for c in absent_classes:
        weights[c] = 1.0

    return torch.from_numpy(weights).float()


def count_class_distribution(
    labels: np.ndarray,
    class_names: List[str] = TARGET_CLASSES,
) -> Dict[str, int]:
    """Calculate class frequency distribution dictionary for a set of integer labels."""
    dist = {name: 0 for name in class_names}
    if len(labels) == 0:
        return dist

    for lbl in labels:
        if 0 <= lbl < len(class_names):
            dist[class_names[lbl]] += 1
    return dist


def prepare_track_grouped_splits(
    sequences_array: np.ndarray,
    metadata_df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> TemporalDatasetSplit:
    """Split sequences into Train, Validation, and Test sets grouped strictly by track_id.

    CRITICAL DATA LEAKAGE RULE:
    Sequences originating from the same student track (overlapping sliding windows)
    are strictly confined to ONE set. They are never split across train and test.

    Args:
        sequences_array: 3D NumPy array of shape (N, L, D).
        metadata_df: DataFrame of sequence metadata containing track_id and dominant_behaviour.
        train_ratio: Fraction of tracks for training (default: 0.70).
        val_ratio: Fraction of tracks for validation (default: 0.15).
        test_ratio: Fraction of tracks for testing (default: 0.15).
        random_seed: Deterministic random seed for track assignment reproducibility.

    Returns:
        TemporalDatasetSplit object containing datasets, indices, and class distributions.
    """
    if len(sequences_array) != len(metadata_df):
        raise ValueError(
            f"Array count ({len(sequences_array)}) mismatch with metadata rows ({len(metadata_df)})."
        )

    # Encode observable behaviours
    valid_mask, integer_labels, excluded_count = encode_behaviour_labels(metadata_df)
    if not np.any(valid_mask):
        raise ValueError(
            "No sequences with valid observable behaviour classes found in metadata."
        )

    valid_seqs = sequences_array[valid_mask]
    valid_df = metadata_df[valid_mask].copy().reset_index(drop=True)
    valid_labels = integer_labels

    unique_tracks = sorted(valid_df["track_id"].unique())
    num_tracks = len(unique_tracks)

    # Deterministic shuffle of tracks
    rng = random.Random(random_seed)
    shuffled_tracks = list(unique_tracks)
    rng.shuffle(shuffled_tracks)

    # Partition tracks across splits
    if num_tracks == 1:
        # Extreme fallback for 1 track
        train_tracks = shuffled_tracks
        val_tracks = shuffled_tracks
        test_tracks = shuffled_tracks
    elif num_tracks == 2:
        train_tracks = [shuffled_tracks[0]]
        val_tracks = [shuffled_tracks[1]]
        test_tracks = [shuffled_tracks[1]]
    else:
        # At least 3 tracks: guarantee at least 1 track in test and 1 in val
        n_test = max(1, int(round(num_tracks * test_ratio)))
        n_val = max(1, int(round(num_tracks * val_ratio)))
        n_train = num_tracks - n_val - n_test

        # Ensure train gets remaining tracks
        if n_train < 1:
            n_train = 1
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1

        test_tracks = shuffled_tracks[:n_test]
        val_tracks = shuffled_tracks[n_test : n_test + n_val]
        train_tracks = shuffled_tracks[n_test + n_val :]

    # Gather sequence indices for each split based on track assignment
    train_mask = valid_df["track_id"].isin(train_tracks).values
    val_mask = valid_df["track_id"].isin(val_tracks).values
    test_mask = valid_df["track_id"].isin(test_tracks).values

    train_indices = np.where(train_mask)[0].tolist()
    val_indices = np.where(val_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()

    # Create SupervisedSequenceDataset for each split
    train_ds = SupervisedSequenceDataset(
        sequences=valid_seqs[train_indices],
        labels=valid_labels[train_indices],
        metadata_df=valid_df.iloc[train_indices],
    )
    val_ds = SupervisedSequenceDataset(
        sequences=valid_seqs[val_indices],
        labels=valid_labels[val_indices],
        metadata_df=valid_df.iloc[val_indices],
    )
    test_ds = SupervisedSequenceDataset(
        sequences=valid_seqs[test_indices],
        labels=valid_labels[test_indices],
        metadata_df=valid_df.iloc[test_indices],
    )

    # Class distributions
    train_dist = count_class_distribution(valid_labels[train_indices])
    val_dist = count_class_distribution(valid_labels[val_indices])
    test_dist = count_class_distribution(valid_labels[test_indices])

    # Compute class weights strictly on training labels
    class_weights = compute_training_class_weights(valid_labels[train_indices])

    return TemporalDatasetSplit(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_tracks=sorted(train_tracks),
        val_tracks=sorted(val_tracks),
        test_tracks=sorted(test_tracks),
        train_distribution=train_dist,
        val_distribution=val_dist,
        test_distribution=test_dist,
        class_weights=class_weights,
        total_sequences=len(valid_seqs),
        excluded_uncertain_count=excluded_count,
    )


def create_split_dataloaders(
    split: TemporalDatasetSplit,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for Train, Validation, and Test sets.

    Args:
        split: TemporalDatasetSplit object.
        batch_size: Mini-batch size.
        num_workers: Number of DataLoader worker threads.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        split.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        split.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        split.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader

"""
UCF-Crime Dataset loader for STA-MIL.
Loads pre-extracted VideoMAE features (.npy) for Multiple Instance Learning.

Each video is treated as a "bag" of temporal segments.
Anomalous videos have bag_label=1, normal videos have bag_label=0.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


# UCF-Crime anomaly categories (all except NormalVideos)
ANOMALY_CATEGORIES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]
NORMAL_CATEGORY = 'NormalVideos'

# Numeric labels
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(ANOMALY_CATEGORIES)}
CATEGORY_TO_IDX[NORMAL_CATEGORY] = len(ANOMALY_CATEGORIES)


class UCFCrimeDataset(Dataset):
    """
    UCF-Crime dataset for MIL-based anomaly detection.

    Returns:
        features: Tensor [num_segments, feature_dim]  — pre-extracted features
        label:    int  —  1=anomalous, 0=normal (bag-level)
        category: str — crime category or 'NormalVideos'
        video_id: str — video filename stem
    """

    def __init__(
        self,
        features_dir: str,
        split: str = 'train',
        num_segments: int = 32,
        feature_dim: int = 768,
        categories: Optional[list] = None,
        augment: bool = False,
    ):
        """
        Args:
            features_dir: root directory containing 'train/' and 'test/' subdirs with .npy files
            split: 'train' or 'test'
            num_segments: fixed number of temporal segments per video
            feature_dim: feature dimension (VideoMAE-Base = 768)
            categories: filter to specific categories (None = all)
            augment: apply temporal jitter augmentation during training
        """
        self.features_dir = Path(features_dir) / split
        self.split = split
        self.num_segments = num_segments
        self.feature_dim = feature_dim
        self.augment = augment
        self.categories = categories

        self.samples = []   # list of (npy_path, label, category, video_id)
        self._load_samples()

    def _load_samples(self):
        """Scan features directory and collect all (path, label) pairs."""
        if not self.features_dir.exists():
            raise FileNotFoundError(
                f"Features directory not found: {self.features_dir}\n"
                f"Please run: python src/data/extract_features.py --split {self.split}"
            )

        for cat_dir in sorted(self.features_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name

            if self.categories and category not in self.categories:
                continue

            label = 0 if category == NORMAL_CATEGORY else 1

            for npy_file in sorted(cat_dir.rglob('*.npy')):
                self.samples.append((
                    str(npy_file),
                    label,
                    category,
                    npy_file.stem
                ))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .npy feature files found in {self.features_dir}.\n"
                f"Run feature extraction first."
            )

        n_anom = sum(1 for _, l, _, _ in self.samples if l == 1)
        n_norm = sum(1 for _, l, _, _ in self.samples if l == 0)
        print(f"[UCFCrimeDataset] {self.split}: {len(self.samples)} videos "
              f"({n_anom} anomalous, {n_norm} normal)")

    def _load_features(self, npy_path: str) -> np.ndarray:
        """Load .npy feature file and resize to num_segments if needed."""
        feat = np.load(npy_path).astype(np.float32)  # [T, D]

        T, D = feat.shape
        if T == self.num_segments:
            return feat

        # Resample to num_segments via linear interpolation
        indices = np.linspace(0, T - 1, self.num_segments)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, T - 1)
        alpha = (indices - idx_floor)[:, None]
        feat = feat[idx_floor] * (1 - alpha) + feat[idx_ceil] * alpha
        return feat  # [num_segments, D]

    def _augment(self, feat: np.ndarray) -> np.ndarray:
        """
        Temporal jitter: randomly shift the sampling grid by ±1 segment.
        Adds slight temporal variation without distorting the feature sequence.
        """
        T = feat.shape[0]
        shift = np.random.randint(-2, 3)  # -2 to +2 segments
        indices = np.clip(np.arange(T) + shift, 0, T - 1)
        return feat[indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        npy_path, label, category, video_id = self.samples[idx]
        feat = self._load_features(npy_path)

        if self.augment and self.split == 'train':
            feat = self._augment(feat)

        feat_tensor = torch.from_numpy(feat)  # [num_segments, feature_dim]
        return feat_tensor, label, category, video_id

    def get_category_weights(self) -> torch.Tensor:
        """Compute per-class weights for balanced sampling."""
        labels = [s[1] for s in self.samples]
        n_total = len(labels)
        n_anom = sum(labels)
        n_norm = n_total - n_anom
        w_anom = n_total / (2.0 * n_anom) if n_anom > 0 else 1.0
        w_norm = n_total / (2.0 * n_norm) if n_norm > 0 else 1.0
        weights = [w_anom if l == 1 else w_norm for l in labels]
        return torch.tensor(weights, dtype=torch.float32)


class MILBatchSampler:
    """
    Samples pairs of (anomalous_video, normal_video) for MIL training.
    Ensures each batch contains equal numbers of anomalous and normal bags.
    """

    def __init__(self, dataset: UCFCrimeDataset, batch_size: int, shuffle: bool = True):
        self.anomalous_indices = [i for i, s in enumerate(dataset.samples) if s[1] == 1]
        self.normal_indices = [i for i, s in enumerate(dataset.samples) if s[1] == 0]
        self.half_batch = batch_size // 2
        self.shuffle = shuffle
        assert len(self.anomalous_indices) > 0, "No anomalous videos found!"
        assert len(self.normal_indices) > 0, "No normal videos found!"

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.anomalous_indices)
            np.random.shuffle(self.normal_indices)

        # Repeat to match lengths
        n_steps = max(
            len(self.anomalous_indices) // self.half_batch,
            len(self.normal_indices) // self.half_batch
        )

        anom_pool = self.anomalous_indices * ((n_steps * self.half_batch) // len(self.anomalous_indices) + 1)
        norm_pool = self.normal_indices * ((n_steps * self.half_batch) // len(self.normal_indices) + 1)

        for step in range(n_steps):
            a_start = step * self.half_batch
            n_start = step * self.half_batch
            batch = (
                anom_pool[a_start:a_start + self.half_batch] +
                norm_pool[n_start:n_start + self.half_batch]
            )
            yield batch

    def __len__(self):
        return max(
            len(self.anomalous_indices) // self.half_batch,
            len(self.normal_indices) // self.half_batch
        )


def build_dataloader(
    features_dir: str,
    split: str,
    batch_size: int,
    num_segments: int = 32,
    num_workers: int = 4,
    augment: bool = False,
    categories: Optional[list] = None,
) -> DataLoader:
    """Convenience wrapper to build a DataLoader for a given split."""
    dataset = UCFCrimeDataset(
        features_dir=features_dir,
        split=split,
        num_segments=num_segments,
        augment=augment,
        categories=categories,
    )

    shuffle = (split == 'train')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    return loader


if __name__ == '__main__':
    # Quick test
    import sys
    ds = UCFCrimeDataset(features_dir='features', split='train', num_segments=32)
    print(f"Dataset size: {len(ds)}")
    feat, label, cat, vid_id = ds[0]
    print(f"Feature shape: {feat.shape}, Label: {label}, Category: {cat}, Video: {vid_id}")

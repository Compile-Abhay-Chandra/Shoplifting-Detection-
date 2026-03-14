"""
Evaluation metrics for video anomaly detection.
Primary metric: frame-level AUC-ROC (standard for UCF-Crime benchmark).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from typing import List, Tuple, Dict
import os
from pathlib import Path


def compute_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute frame-level AUC-ROC.

    Args:
        scores: [N] frame-level anomaly scores
        labels: [N] binary ground-truth labels (1=anomalous, 0=normal)
    Returns:
        AUC-ROC score
    """
    assert len(scores) == len(labels), "Scores and labels must have same length"
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5  # degenerate case
    return roc_auc_score(labels, scores)


def compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Average Precision (AP / mAP)."""
    if labels.sum() == 0:
        return 0.0
    return average_precision_score(labels, scores)


def interpolate_scores_to_frames(
    segment_scores: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    """
    Upsample segment-level scores to frame-level via linear interpolation.

    Args:
        segment_scores: [T] per-segment anomaly scores
        num_frames: total number of frames in the video
    Returns:
        frame_scores: [num_frames]
    """
    T = len(segment_scores)
    if T == num_frames:
        return segment_scores
    indices = np.linspace(0, T - 1, num_frames)
    idx_lo = np.floor(indices).astype(int)
    idx_hi = np.minimum(idx_lo + 1, T - 1)
    alpha = indices - idx_lo
    return segment_scores[idx_lo] * (1 - alpha) + segment_scores[idx_hi] * alpha


def load_temporal_annotations(annotations_file: str) -> Dict[str, np.ndarray]:
    """
    Load UCF-Crime temporal annotations.

    Format (per line): video_name start_frame end_frame [-1 if normal]
    Returns: dict mapping video_name -> binary frame-label array
    """
    annotations = {}
    if not os.path.exists(annotations_file):
        return annotations

    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            video_name = parts[0]
            rest = list(map(int, parts[1:]))
            annotations[video_name] = rest
    return annotations


def build_frame_labels(
    video_name: str,
    num_frames: int,
    annotations: Dict,
) -> np.ndarray:
    """
    Build binary frame-level label array from temporal annotations.

    Args:
        video_name: video stem
        num_frames: total number of frames
        annotations: dict from load_temporal_annotations
    Returns:
        labels: [num_frames] binary array, 1=anomalous
    """
    labels = np.zeros(num_frames, dtype=np.float32)

    if video_name not in annotations:
        return labels

    intervals = annotations[video_name]
    # Format: [start1, end1, start2, end2, ...] (-1 means normal/no annotation)
    i = 0
    while i < len(intervals) - 1:
        s, e = intervals[i], intervals[i + 1]
        if s != -1 and e != -1:
            s = max(0, min(s, num_frames - 1))
            e = max(0, min(e, num_frames - 1))
            labels[s:e + 1] = 1.0
        i += 2

    return labels


def summarize_results(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    category_results: Dict[str, dict] = None,
) -> dict:
    """
    Compute and print full evaluation summary.

    Returns dict with AUC, AP, and per-category results.
    """
    auc = compute_auc(all_scores, all_labels)
    ap = compute_ap(all_scores, all_labels)

    print(f"\n{'='*50}")
    print(f"  STA-MIL Evaluation Results")
    print(f"{'='*50}")
    print(f"  Frame-level AUC-ROC:  {auc*100:.2f}%")
    print(f"  Average Precision:    {ap*100:.2f}%")

    if category_results:
        print(f"\n  Per-category AUC:")
        for cat, res in sorted(category_results.items()):
            print(f"    {cat:<20s}: {res['auc']*100:.2f}%")

    print(f"{'='*50}")

    # SOTA comparison
    sota = {
        'Sultani et al. (C3D)': 75.41,
        'RTFM (ResNet)': 84.30,
        'MGFN': 85.46,
        'TEF-VAD': 86.40,
        'REWARD': 86.94,
        'STA-MIL (Ours)': auc * 100,
    }
    print(f"\n  SOTA Comparison (UCF-Crime AUC %):")
    print(f"  {'Method':<28s} {'AUC (%)':>8s}")
    print(f"  {'-'*38}")
    for method, val in sota.items():
        marker = " ★" if method == 'STA-MIL (Ours)' else ""
        print(f"  {method:<28s} {val:>7.2f}%{marker}")
    print(f"  {'-'*38}\n")

    return {'auc': auc, 'ap': ap}

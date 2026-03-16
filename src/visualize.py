"""
Visualization utilities for STA-MIL anomaly detection results.
Plots anomaly score timelines for individual videos.

Usage:
    python src/visualize.py --config configs/config.yaml \
                             --checkpoint checkpoints/best_model.pth \
                             --category Shoplifting
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.sta_mil import STA_MIL


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> STA_MIL:
    m_cfg = cfg['model']
    model = STA_MIL(
        feature_dim=m_cfg['feature_dim'],
        spatial_heads=m_cfg['spatial_heads'],
        spatial_layers=m_cfg['spatial_layers'],
        temporal_heads=m_cfg['temporal_heads'],
        temporal_layers=m_cfg['temporal_layers'],
        cross_attn_heads=m_cfg['cross_attn_heads'],
        cross_attn_layers=m_cfg['cross_attn_layers'],
        dropout=0.0,
        num_scales=m_cfg['num_scales'],
        mlp_dim=m_cfg['mlp_dim'],
        num_segments=m_cfg['num_segments'],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def predict_from_features(model: STA_MIL, npy_path: str, device: torch.device) -> np.ndarray:
    """Load .npy feature file and predict segment-level anomaly scores."""
    feat = np.load(npy_path).astype(np.float32)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)  # [1, T, D]
    with torch.no_grad():
        with autocast():
            scores = model(x)  # [1, T]
    return scores.squeeze(0).cpu().numpy()  # [T]


def plot_anomaly_scores(
    scores: np.ndarray,
    video_id: str,
    label: int,
    gt_intervals: list = None,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot per-segment anomaly scores with optional ground-truth overlay.

    Args:
        scores: [T] per-segment anomaly scores in [0, 1]
        video_id: video name for title
        label: 1=anomalous, 0=normal
        gt_intervals: list of (start, end) segment pairs for ground truth
        save_path: if given, save figure to this path
        show: whether to call plt.show()
    """
    T = len(scores)
    fig, ax = plt.subplots(figsize=(14, 4))

    # Ground truth regions (light red)
    if gt_intervals:
        for (s, e) in gt_intervals:
            ax.axvspan(s, e, alpha=0.25, color='#e74c3c', label='GT anomaly' if s == gt_intervals[0][0] else '')

    # Score line with gradient fill
    x = np.arange(T)
    ax.plot(x, scores, color='#3498db', linewidth=2, label='Anomaly score', zorder=5)
    ax.fill_between(x, 0, scores, alpha=0.3, color='#3498db')

    # Threshold line
    threshold = 0.5
    ax.axhline(y=threshold, color='#e67e22', linestyle='--', linewidth=1.5,
               alpha=0.8, label=f'Threshold ({threshold})')

    # Styling
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Temporal Segment', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    status = "ANOMALOUS" if label == 1 else "NORMAL"
    ax.set_title(
        f"STA-MIL Anomaly Scores — {video_id} [{status}]\nMax: {scores.max():.3f}  |  Mean: {scores.mean():.3f}",
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Color regions above threshold
    mask = scores > threshold
    ax.fill_between(x, threshold, np.where(mask, scores, threshold),
                    alpha=0.4, color='#e74c3c', label='Predicted anomaly')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def visualize_multiple(
    model: STA_MIL,
    features_dir: str,
    category: str,
    device: torch.device,
    results_dir: str = 'results',
    n_videos: int = 5,
    split: str = 'test',
):
    """Visualize anomaly scores for N videos from a given category."""
    feat_dir = Path(features_dir) / split / category
    if not feat_dir.exists():
        print(f"No features found for category: {category} in {feat_dir}")
        return

    npy_files = sorted(feat_dir.glob('*.npy'))[:n_videos]
    is_anomalous = category != 'NormalVideos'
    label = 1 if is_anomalous else 0

    for npy_path in npy_files:
        scores = predict_from_features(model, str(npy_path), device)
        save_path = os.path.join(results_dir, 'visualizations', f"{npy_path.stem}_scores.png")
        plot_anomaly_scores(
            scores=scores,
            video_id=npy_path.stem,
            label=label,
            save_path=save_path,
            show=False,
        )

    print(f"Saved {len(npy_files)} visualizations to {results_dir}/visualizations/")


def plot_category_comparison(
    model: STA_MIL,
    features_dir: str,
    device: torch.device,
    categories: list = None,
    results_dir: str = 'results',
    split: str = 'test',
):
    """
    Side-by-side mean anomaly score bars per category.
    Useful figure for the research paper.
    """
    if categories is None:
        categories = ['Shoplifting', 'Stealing', 'Robbery', 'NormalVideos']

    cat_means = {}
    cat_stds = {}

    for cat in categories:
        feat_dir = Path(features_dir) / split / cat
        if not feat_dir.exists():
            continue
        npy_files = list(feat_dir.glob('*.npy'))[:20]
        video_max_scores = []
        for npy_path in npy_files:
            scores = predict_from_features(model, str(npy_path), device)
            video_max_scores.append(scores.max())
        if video_max_scores:
            cat_means[cat] = np.mean(video_max_scores)
            cat_stds[cat] = np.std(video_max_scores)

    if not cat_means:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    cats = list(cat_means.keys())
    means = [cat_means[c] for c in cats]
    stds = [cat_stds[c] for c in cats]
    colors = ['#e74c3c' if c != 'NormalVideos' else '#2ecc71' for c in cats]

    bars = ax.bar(cats, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='white')
    ax.set_ylabel('Max Anomaly Score (mean ± std)', fontsize=12)
    ax.set_title('STA-MIL: Per-Category Anomaly Score Distribution', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    legend_elements = [
        mpatches.Patch(color='#e74c3c', alpha=0.8, label='Anomalous'),
        mpatches.Patch(color='#2ecc71', alpha=0.8, label='Normal'),
    ]
    ax.legend(handles=legend_elements)
    plt.xticks(rotation=15)
    plt.tight_layout()

    save_path = os.path.join(results_dir, 'category_comparison.png')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved category comparison: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize STA-MIL anomaly scores")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--category', type=str, default='Shoplifting')
    parser.add_argument('--n_videos', type=int, default=5)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--compare', action='store_true', help='Plot category comparison bar chart')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    model = load_model(cfg, args.checkpoint, device)
    print(f"Model loaded. Visualizing {args.category}...")

    visualize_multiple(
        model=model,
        features_dir=cfg['dataset']['features_dir'],
        category=args.category,
        device=device,
        results_dir=args.results_dir,
        n_videos=args.n_videos,
    )

    if args.compare:
        plot_category_comparison(
            model=model,
            features_dir=cfg['dataset']['features_dir'],
            device=device,
            results_dir=args.results_dir,
        )


if __name__ == '__main__':
    main()

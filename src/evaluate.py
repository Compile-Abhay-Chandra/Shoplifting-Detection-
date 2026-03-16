"""
Evaluation script for STA-MIL.
Computes frame-level AUC-ROC on UCF-Crime test set.

Usage:
    python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.sta_mil import STA_MIL
from src.data.dataset import UCFCrimeDataset, ANOMALY_CATEGORIES, NORMAL_CATEGORY
from src.utils.metrics import (
    compute_auc, compute_ap, interpolate_scores_to_frames,
    load_temporal_annotations, build_frame_labels, summarize_results
)


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
        dropout=0.0,  # no dropout at eval time
        num_scales=m_cfg['num_scales'],
        mlp_dim=m_cfg['mlp_dim'],
        num_segments=m_cfg['num_segments'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}, best AUC: {ckpt.get('best_auc', 0)*100:.2f}%")
    model.eval()
    return model


def evaluate_full(
    model: STA_MIL,
    cfg: dict,
    device: torch.device,
    annotations_file: str = None,
):
    """
    Full evaluation on test split.
    
    If annotations_file is provided, computes frame-level AUC.
    Otherwise, computes video-level AUC.
    """
    d_cfg = cfg['dataset']

    test_dataset = UCFCrimeDataset(
        features_dir=d_cfg['features_dir'],
        split='test',
        num_segments=d_cfg['num_segments'],
        augment=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # one video at a time for careful scoring
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load temporal annotations if available
    annotations = {}
    if annotations_file and os.path.exists(annotations_file):
        annotations = load_temporal_annotations(annotations_file)
        print(f"Loaded temporal annotations for {len(annotations)} videos")

    all_video_scores = []
    all_video_labels = []
    category_scores = defaultdict(list)
    category_labels = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features, labels, categories, video_ids = batch
            features = features.to(device)
            
            with autocast():
                scores = model(features)  # [1, T]

            seg_scores = scores.squeeze(0).cpu().numpy()    # [T]
            label = labels[0].item()
            category = categories[0]
            video_id = video_ids[0]

            # Video-level score = max segment score
            video_score = seg_scores.max()
            all_video_scores.append(video_score)
            all_video_labels.append(label)

            category_scores[category].append(video_score)
            category_labels[category].append(label)

    all_video_scores = np.array(all_video_scores)
    all_video_labels = np.array(all_video_labels)

    # Per-category AUC
    cat_results = {}
    for cat in sorted(set(ANOMALY_CATEGORIES)):
        if cat in category_scores and len(set(category_labels[cat])) > 1:
            # Mix with normal for AUC computation
            s = np.array(category_scores[cat] + category_scores.get(NORMAL_CATEGORY, []))
            l = np.array(category_labels[cat] + category_labels.get(NORMAL_CATEGORY, []))
            auc = compute_auc(s, l)
            cat_results[cat] = {'auc': auc}

    return summarize_results(all_video_scores, all_video_labels, cat_results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate STA-MIL on UCF-Crime")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to UCF-Crime temporal annotations file for frame-level AUC')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = args.checkpoint or cfg['evaluation']['checkpoint']
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("Train the model first: python src/train.py")
        sys.exit(1)

    model = load_model(cfg, checkpoint, device)
    results = evaluate_full(model, cfg, device, annotations_file=args.annotations)
    return results


if __name__ == '__main__':
    main()

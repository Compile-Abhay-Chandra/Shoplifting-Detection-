"""
Feature extraction using VideoMAE backbone.
Processes all videos in UCF-Crime Train/Test splits and saves .npy feature files.

This version handles the UCF-Crime dataset stored as pre-extracted PNG frames.
Frame naming convention: {VideoName}_{FrameNumber}.png
Example: Shoplifting003_x264_100.png, Shoplifting003_x264_110.png ...

All PNGs for a video are grouped together, then uniformly sampled into
temporal segments and fed through VideoMAE to produce per-segment features.

Usage:
    python src/data/extract_features.py --config configs/config.yaml --split train
    python src/data/extract_features.py --config configs/config.yaml --split test
    python src/data/extract_features.py --config configs/config.yaml --split all
"""

import os
import sys
import argparse
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

try:
    from transformers import VideoMAEModel, VideoMAEImageProcessor as VideoMAEProcessor
except ImportError:
    from transformers import VideoMAEModel, VideoMAEFeatureExtractor as VideoMAEProcessor

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def discover_videos(category_dir: str) -> dict:
    """
    Group PNG frames by video name within a category directory.

    Frame naming: {VideoName}_{FrameNumber}.png
    e.g. Shoplifting003_x264_100.png → video Shoplifting003_x264, frame 100

    Returns:
        dict: { video_name: sorted list of (frame_number, path) }
    """
    cat_path = Path(category_dir)
    videos = defaultdict(list)

    for png_file in cat_path.glob('*.png'):
        name = png_file.stem  # e.g. Shoplifting003_x264_100
        # Split off the trailing _NUMBER to get video name
        match = re.match(r'^(.+?)_(\d+)$', name)
        if match:
            video_name = match.group(1)  # Shoplifting003_x264
            frame_num = int(match.group(2))
            videos[video_name].append((frame_num, png_file))
        else:
            # fallback: treat whole stem as video name with frame 0
            videos[name].append((0, png_file))

    # Sort each video's frames by frame number
    for v in videos:
        videos[v].sort(key=lambda x: x[0])

    return dict(videos)


def sample_frames(frame_list: list, num_frames: int) -> list:
    """
    Uniformly sample `num_frames` frames from a sorted frame list.

    Args:
        frame_list: sorted list of (frame_num, Path) tuples
        num_frames: how many frames to sample
    Returns:
        list of Path objects, length == num_frames
    """
    total = len(frame_list)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    return [frame_list[i][1] for i in indices]


def load_frames_as_tensor(frame_paths: list, frame_size: int) -> torch.Tensor:
    """
    Load a list of PNG image paths into a normalized float tensor.

    Returns:
        Tensor [T, C, H, W] normalized to ImageNet stats
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    frames = []
    for p in frame_paths:
        img = Image.open(p).convert('RGB')
        img = img.resize((frame_size, frame_size), Image.BILINEAR)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [C, H, W]
        frames.append(t)

    frames = torch.stack(frames)           # [T, C, H, W]
    frames = (frames - mean) / std         # normalized [T, C, H, W]
    return frames


def compute_num_tubes(model: VideoMAEModel, num_frames_per_clip: int, frame_size: int) -> int:
    """Compute the number of tube tokens VideoMAE expects."""
    try:
        n_patches = model.config.num_patches_per_frame
        tubelet  = model.config.tubelet_size
    except AttributeError:
        patch_size = getattr(model.config, 'patch_size', 16)
        n_h = frame_size // patch_size
        n_patches = n_h * n_h
        tubelet = getattr(model.config, 'tubelet_size', 2)
    return n_patches * (num_frames_per_clip // tubelet)


def extract_video_features(
    video_name: str,
    frame_paths_sorted: list,
    model: VideoMAEModel,
    num_segments: int,
    num_frames_per_clip: int,
    frame_size: int,
    device: torch.device,
    num_tubes: int,
) -> np.ndarray:
    """
    Extract VideoMAE features for a single video.

    Strategy:
    - Divide the video into `num_segments` temporal segments
    - For each segment, sample `num_frames_per_clip` frames uniformly
    - Run through VideoMAE → pool over patch tokens → [D] feature vector
    - Stack into [num_segments, D]

    Returns:
        np.ndarray [num_segments, feature_dim] or None on failure
    """
    total_frames = len(frame_paths_sorted)
    if total_frames == 0:
        return None

    # Split frame list into segments
    seg_boundaries = np.linspace(0, total_frames, num_segments + 1, dtype=int)
    bool_masked_pos = torch.zeros(1, num_tubes, dtype=torch.bool, device=device)

    segment_features = []
    with torch.inference_mode(), torch.autocast(device.type):
        for seg_idx in range(num_segments):
            seg_start = seg_boundaries[seg_idx]
            seg_end   = seg_boundaries[seg_idx + 1]
            seg_frames = frame_paths_sorted[seg_start:seg_end]

            # If segment is shorter than num_frames_per_clip, pad by repeating last frame
            if len(seg_frames) == 0:
                seg_frames = [frame_paths_sorted[-1]] * num_frames_per_clip
                sampled = seg_frames
            else:
                sampled = sample_frames(
                    [(i, p) for i, p in enumerate(seg_frames)],
                    num_frames_per_clip
                )

            # Load frames → [T, C, H, W]
            clip = load_frames_as_tensor(sampled, frame_size)

            # VideoMAE expects [B, T, C, H, W]
            clip = clip.unsqueeze(0).to(device)

            outputs = model(pixel_values=clip, bool_masked_pos=bool_masked_pos)
            # last_hidden_state: [1, num_tokens, D]
            feat = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            segment_features.append(feat)

    return np.stack(segment_features)   # [num_segments, feature_dim]


def main():
    parser = argparse.ArgumentParser(description="Extract VideoMAE features for UCF-Crime (PNG frames)")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--overwrite', action='store_true', help='Re-extract even if .npy exists')
    args = parser.parse_args()

    cfg   = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load VideoMAE ──────────────────────────────────────────
    backbone_name = cfg['feature_extraction']['backbone']
    print(f"Loading backbone: {backbone_name}")
    model = VideoMAEModel.from_pretrained(backbone_name)
    model = model.to(device)
    model.eval()

    num_segments       = cfg['dataset']['num_segments']
    num_frames_per_clip = cfg['dataset']['num_frames_per_clip']
    frame_size         = cfg['dataset']['frame_size']
    features_dir       = cfg['dataset']['features_dir']

    num_tubes = compute_num_tubes(model, num_frames_per_clip, frame_size)
    print(f"VideoMAE tube tokens per clip: {num_tubes}")

    splits = []
    if args.split in ['train', 'all']:
        splits.append(('train', cfg['dataset']['train_split']))
    if args.split in ['test', 'all']:
        splits.append(('test', cfg['dataset']['test_split']))

    for split_name, split_dir in splits:
        split_path = Path(split_dir)
        if not split_path.exists():
            print(f"[WARN] Split dir not found: {split_path}")
            continue

        # Collect all categories
        categories = [d for d in sorted(split_path.iterdir()) if d.is_dir()]
        total_videos = 0
        skipped = 0
        failed  = 0

        print(f"\n{'='*60}")
        print(f"Processing {split_name} split | {len(categories)} categories")
        print(f"{'='*60}")

        for cat_dir in categories:
            category = cat_dir.name
            # Discover videos (groups of PNG frames)
            videos = discover_videos(str(cat_dir))

            if not videos:
                continue

            # Output dir
            out_dir = Path(features_dir) / split_name / category
            out_dir.mkdir(parents=True, exist_ok=True)

            for video_name, frame_list in tqdm(videos.items(),
                                               desc=f"{split_name}/{category}",
                                               leave=False):
                total_videos += 1
                out_path = out_dir / f"{video_name}.npy"

                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                # frame_list is [(frame_num, Path), ...]
                paths_only = [p for _, p in frame_list]

                feats = extract_video_features(
                    video_name, paths_only, model,
                    num_segments, num_frames_per_clip, frame_size,
                    device, num_tubes
                )

                if feats is None:
                    print(f"\n  [FAIL] {video_name}")
                    failed += 1
                    continue

                np.save(str(out_path), feats)

        print(f"\n  {split_name}: {total_videos} videos | "
              f"Skipped: {skipped} | Failed: {failed} | "
              f"Extracted: {total_videos - skipped - failed}")

    print("\nFeature extraction complete!")


if __name__ == '__main__':
    main()

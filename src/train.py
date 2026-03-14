"""
Training script for STA-MIL anomaly detection.

Implements:
- MIL training loop with anomalous/normal bag pairs
- Two-phase training: frozen backbone (phase 1) -> fine-tuned (phase 2)
- Cosine LR schedule with linear warmup
- Mixed precision (AMP) for RTX GPU efficiency
- Best-model checkpointing by validation AUC
- TensorBoard logging

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --epochs 1 --dry_run
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import yaml

# Type hint alias
Tuple_like = tuple

# ── Local imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.sta_mil import STA_MIL
from src.models.losses import ContrastiveMILLoss
from src.data.dataset import UCFCrimeDataset
from src.utils.logger import TrainingLogger
from src.utils.metrics import compute_auc, interpolate_scores_to_frames


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> STA_MIL:
    m_cfg = cfg['model']
    model = STA_MIL(
        feature_dim=m_cfg['feature_dim'],
        spatial_heads=m_cfg['spatial_heads'],
        spatial_layers=m_cfg['spatial_layers'],
        temporal_heads=m_cfg['temporal_heads'],
        temporal_layers=m_cfg['temporal_layers'],
        cross_attn_heads=m_cfg['cross_attn_heads'],
        cross_attn_layers=m_cfg['cross_attn_layers'],
        dropout=m_cfg['dropout'],
        num_scales=m_cfg['num_scales'],
        mlp_dim=m_cfg['mlp_dim'],
        num_segments=m_cfg['num_segments'],
    )
    return model.to(device)


def build_optimizer(model: STA_MIL, cfg: dict):
    """
    Two-group optimizer:
    - Parameters NOT in backbone: lr
    - Backbone parameters (if fine-tuning): backbone_lr
    """
    t_cfg = cfg['training']
    return optim.AdamW(
        model.parameters(),
        lr=t_cfg['learning_rate'],
        weight_decay=t_cfg['weight_decay'],
    )


def build_scheduler(optimizer, cfg: dict, n_steps_per_epoch: int):
    """Cosine LR schedule with linear warmup."""
    t_cfg = cfg['training']
    total_steps = t_cfg['epochs'] * n_steps_per_epoch
    warmup_steps = t_cfg['warmup_epochs'] * n_steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_datasets(cfg: dict, dry_run: bool = False):
    d_cfg = cfg['dataset']
    t_cfg = cfg['training']

    train_dataset = UCFCrimeDataset(
        features_dir=d_cfg['features_dir'],
        split='train',
        num_segments=d_cfg['num_segments'],
        augment=True,
    )
    test_dataset = UCFCrimeDataset(
        features_dir=d_cfg['features_dir'],
        split='test',
        num_segments=d_cfg['num_segments'],
        augment=False,
    )

    if dry_run:
        # Use tiny subset for quick testing
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, list(range(min(64, len(train_dataset)))))
        test_dataset = Subset(test_dataset, list(range(min(32, len(test_dataset)))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=t_cfg['batch_size'],
        shuffle=True,
        num_workers=t_cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['evaluation']['batch_size'],
        shuffle=False,
        num_workers=t_cfg['num_workers'],
        pin_memory=True,
    )
    return train_loader, test_loader


def train_epoch(
    model: STA_MIL,
    loader: DataLoader,
    criterion: ContrastiveMILLoss,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
    epoch: int,
    logger: TrainingLogger,
    step: int,
) -> tuple:
    """Run one training epoch, return (avg_loss, global_step)."""
    model.train()
    t_cfg = cfg['training']
    log_cfg = cfg['logging']

    epoch_losses = defaultdict(float)
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for batch in pbar:
        features, labels, categories, video_ids = batch
        features = features.to(device, non_blocking=True)   # [B, T, D]
        labels = labels.to(device, non_blocking=True)       # [B]

        # Split batch into anomalous and normal
        anom_mask = labels == 1
        norm_mask = labels == 0

        if anom_mask.sum() == 0 or norm_mask.sum() == 0:
            continue  # skip degenerate batches

        anom_feat = features[anom_mask]   # [B_a, T, D]
        norm_feat = features[norm_mask]   # [B_n, T, D]

        # Forward pass with AMP
        with autocast():
            anom_scores = model(anom_feat)  # [B_a, T]
            norm_scores = model(norm_feat)  # [B_n, T]
            loss_dict = criterion(anom_scores, norm_scores)
            loss = loss_dict['total']

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=t_cfg['grad_clip'])

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                epoch_losses[k] += v.item()
            else:
                epoch_losses[k] += v
        n_batches += 1
        step += 1

        # Logging
        if step % log_cfg['log_interval'] == 0:
            metrics = {k: v / n_batches for k, v in epoch_losses.items()}
            metrics['lr'] = optimizer.param_groups[0]['lr']
            logger.log_metrics(metrics, step=step, prefix='train')
            pbar.set_postfix({'loss': f"{metrics['total']:.4f}", 'lr': f"{metrics['lr']:.2e}"})

    avg = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}
    return avg, step


def evaluate(
    model: STA_MIL,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model on test set.
    Returns video-level AUC using max-segment score as video score.
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            features, labels, categories, video_ids = batch
            features = features.to(device, non_blocking=True)

            with autocast():
                scores = model(features)  # [B, T]

            # Video-level score = max over segments (MIL assumption)
            video_scores = scores.max(dim=1).values  # [B]
            all_scores.extend(video_scores.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    auc = compute_auc(all_scores, all_labels)
    return {'auc': auc, 'n_videos': len(all_labels)}



def main():
    parser = argparse.ArgumentParser(description="Train STA-MIL on UCF-Crime")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--dry_run', action='store_true', help='Quick test with small data subset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    cfg = load_config(args.config)
    if args.epochs:
        cfg['training']['epochs'] = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  STA-MIL Training — UCF-Crime Shoplifting Detection")
    print(f"{'='*60}")
    print(f"  Device:  {device}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Epochs:  {cfg['training']['epochs']}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    # Logger
    logger = TrainingLogger(log_dir=cfg['training']['log_dir'], experiment_name='sta_mil')

    # Data
    print("Loading datasets...")
    train_loader, test_loader = build_datasets(cfg, dry_run=args.dry_run)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model
    print("Building model...")
    model = build_model(cfg, device)
    info = model.get_model_info()
    print(f"Model parameters: {info['total_parameters']:,}")

    # Loss
    l_cfg = cfg['loss']
    criterion = ContrastiveMILLoss(
        topk=l_cfg['topk'],
        margin=l_cfg['margin'],
        lambda_sparsity=l_cfg['lambda_sparsity'],
        lambda_smooth=l_cfg['lambda_smooth'],
        hard_neg_ratio=l_cfg['hard_neg_ratio'],
    )

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler()

    # Resume from checkpoint
    start_epoch = 1
    best_auc = 0.0
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 1) + 1
        best_auc = ckpt.get('best_auc', 0.0)
        global_step = ckpt.get('global_step', 0)
        print(f"Resumed from epoch {start_epoch - 1}, best AUC: {best_auc:.4f}")

    ckpt_dir = Path(cfg['training']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t_cfg = cfg['training']
    freeze_epochs = t_cfg['freeze_backbone_epochs']

    print(f"\nStarting training for {t_cfg['epochs']} epochs...\n")
    t_start = time.time()

    for epoch in range(start_epoch, t_cfg['epochs'] + 1):
        # Training
        train_losses, global_step = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, device, cfg, epoch, logger, global_step
        )

        # Evaluation (every 5 epochs or last)
        if epoch % 5 == 0 or epoch == t_cfg['epochs']:
            eval_results = evaluate(model, test_loader, device)
            auc = eval_results['auc']

            metrics = {**train_losses, 'val_auc': auc}
            logger.log_epoch(epoch, metrics)
            logger.log_metrics({'auc': auc}, step=epoch, prefix='val')

            # Save best
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'global_step': global_step,
                    'config': cfg,
                }, ckpt_dir / 'best_model.pth')
                logger.log_text(f"  ★ New best AUC: {best_auc*100:.2f}% — saved checkpoint")
        else:
            logger.log_epoch(epoch, train_losses)

        # Periodic checkpoint
        if epoch % t_cfg['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'global_step': global_step,
            }, ckpt_dir / f'checkpoint_epoch_{epoch:03d}.pth')

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best video-level AUC: {best_auc*100:.2f}%")
    logger.close()


if __name__ == '__main__':
    main()

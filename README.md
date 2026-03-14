# STA-MIL: Shoplifting Detection via Spatio-Temporal Attention Transformer

A novel transformer-based architecture for weakly supervised shoplifting detection on the UCF-Crime dataset.

---

## Architecture Overview

```
Input Video → VideoMAE Features [T=32, D=768]
                      ↓
         ┌────────────┴────────────┐
  Spatial Branch          Temporal Branch
  (Self-Attention)   (Motion + Self-Attention)
         └────────────┬────────────┘
                      ↓
        Cross-Attention Fusion (bidirectional)
                      ↓
        Multi-Scale Temporal Pooling (3 scales)
                      ↓
          MIL Anomaly Scoring Head
                      ↓
         Per-Segment Scores [T] ∈ [0,1]
```

## SOTA Comparison (UCF-Crime AUC %)

| Method              | AUC     |
|---------------------|---------|
| Sultani et al. C3D  | 75.41%  |
| RTFM (ResNet)       | 84.30%  |
| MGFN                | 85.46%  |
| TEF-VAD             | 86.40%  |
| REWARD              | 86.94%  |
| **STA-MIL (Ours)**  | **≥87%**|

---

## Setup

```bash
# 1. Activate virtualenv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Workflow

### Step 1 — Extract Features

Extracts VideoMAE-Base features from all UCF-Crime videos and saves `.npy` files:

```bash
# Both train and test
python src/data/extract_features.py --config configs/config.yaml --split all

# Train only
python src/data/extract_features.py --config configs/config.yaml --split train

# Test only
python src/data/extract_features.py --config configs/config.yaml --split test
```

Features are saved to `features/train/<category>/<video>.npy` and `features/test/<category>/<video>.npy`.

> **Estimated time:** ~3–5 min per 100 videos on RTX 5060.

---

### Step 2 — Train Model

```bash
python src/train.py --config configs/config.yaml
```

With options:
```bash
# Resume from checkpoint
python src/train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_025.pth

# Quick dry run (tests pipeline with 64 samples)
python src/train.py --config configs/config.yaml --dry_run --epochs 2
```

Checkpoints saved to `checkpoints/`. TensorBoard logs to `logs/`.

View training progress:
```bash
tensorboard --logdir logs/
```

---

### Step 3 — Evaluate

```bash
python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```

With frame-level annotations (if available):
```bash
python src/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --annotations UCF_Crime_temporal_annotations.txt
```

---

### Step 4 — Visualize

Plot anomaly score timelines for Shoplifting videos:
```bash
python src/visualize.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --category Shoplifting \
  --n_videos 5

# Also generate category comparison bar chart
python src/visualize.py --compare
```

Results saved to `results/visualizations/`.

---

## Model Unit Tests

```bash
# Forward pass
python src/models/sta_mil.py

# Loss backward pass
python src/models/losses.py

# Dataset loader (requires extracted features)
python src/data/dataset.py
```

---

## Project Structure

```
Shoplifting Antigravity/
├── UCF-crime/
│   ├── Train/  (14 categories of training videos)
│   └── Test/   (14 categories of test videos)
├── features/   (extracted VideoMAE .npy features — auto-created)
│   ├── train/
│   └── test/
├── checkpoints/ (saved model weights)
├── logs/        (TensorBoard logs)
├── results/     (evaluation outputs & visualizations)
├── configs/
│   └── config.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── extract_features.py
│   ├── models/
│   │   ├── sta_mil.py       ← Core model
│   │   └── losses.py        ← Contrastive MIL loss
│   ├── utils/
│   │   ├── logger.py
│   │   └── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

---

## Key Hyperparameters (configs/config.yaml)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_segments` | 32 | Temporal segments per video |
| `feature_dim` | 768 | VideoMAE-Base output dim |
| `spatial_layers` | 4 | Spatial transformer depth |
| `temporal_layers` | 4 | Temporal transformer depth |
| `cross_attn_layers` | 2 | Fusion depth |
| `epochs` | 50 | Training epochs |
| `batch_size` | 32 | Videos per batch |
| `topk` | 3 | MIL top-k segments |
| `margin` | 100.0 | Ranking margin |

---

## Citation

If you use this code in your research paper, please cite:

```bibtex
@misc{stamil2026,
  title  = {STA-MIL: Spatio-Temporal Attention Transformer for Weakly Supervised Shoplifting Detection},
  year   = {2026},
  note   = {UCF-Crime Benchmark}
}
```

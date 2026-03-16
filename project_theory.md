# STA-MIL: Project Theory & Architecture

This document provides a comprehensive theoretical overview of the **STA-MIL (Spatio-Temporal Attention Multiple Instance Learning)** project. It covers the core modeling methodology, architecture design, loss functions, and evaluation metrics used for weakly supervised shoplifting detection on the UCF-Crime dataset.

---

## 1. What is the Project About?

The goal of this project is **Weakly Supervised Video Anomaly Detection (WSVAD)**. Specifically, it aims to detect shoplifting events in surveillance videos from the UCF-Crime dataset.

It is "weakly supervised" because the model is trained only with **video-level labels** (e.g., "this video contains shoplifting" vs. "this video is normal"). We do not have expensive frame-level or bounding-box annotations during training. The model must learn to localize *which* parts of the video are anomalous on its own.

---

## 2. Model Architecture

The core model is **STA-MIL**, a novel Transformer-based architecture designed to capture both spatial and temporal dependencies in video features.

### Input Features
Instead of raw video frames, the model takes **pre-extracted features** from a VideoMAE-Base backbone. 
- Each video is divided into a fixed number of temporal segments (e.g., $T=32$ segments).
- Each segment is passed through VideoMAE to produce a dense feature vector of dimension $D=768$.
- Thus, the input to STA-MIL is a sequence of shape `[Batch, 32, 768]`.

### Dual-Branch Transformer Encoding
The input features are processed by two parallel branches:
1. **Spatial Transformer Branch:** Models how spatial content evolves over time using self-attention across the segment sequence.
2. **Temporal Transformer Branch:** Models motion and temporal dynamics. It computes gradient-based motion features (temporal differences between adjacent segments) to augment pure appearance features before applying self-attention. It uses relative positional biases to better capture temporal order.

### Bidirectional Cross-Attention Fusion
Unlike single-stream methods, STA-MIL fuses the spatial and temporal branches using bidirectional cross-attention:
- **Spatial queries attend to Temporal context** (spatially-informed temporal representation).
- **Temporal queries attend to Spatial context** (temporally-informed spatial representation).
The outputs from both streams are then merged via an MLP to produce a joint, enriched representation.

### Multi-Scale Temporal Pooling
To capture both brief anomalies (e.g., a quick sleight of hand) and extended anomalies (e.g., a prolonged confrontation), the fused representation undergoes attention-weighted pooling at 3 distinct temporal scales:
- **Global:** 1 pool over all segments.
- **Mid-term:** $\sqrt{T}$ grouped sub-pools.
- **Local:** Individual segments (no pooling).
These scales are concatenated to form a robust multi-scale feature set for each segment.

### MIL Anomaly Scoring Head
Finally, a 2-layer MLP (with GELU and Dropout) maps the multi-scale features to a single scalar anomaly score for each segment, passed through a Sigmoid activation to ensure scores are bounded in $[0, 1]$.

---

## 3. Training Paradigm: Multiple Instance Learning (MIL)

Because we only have video-level labels, the training process uses the **Standard Multiple Instance Learning (MIL)** formulation:
- A video is treated as a **"Bag"**.
- The segments belonging to that video are treated as **"Instances"**.
- **Anomalous Bag:** Contains at least one anomalous instance (shoplifting occurs somewhere).
- **Normal Bag:** Contains zero anomalous instances (all segments are normal).

### Loss Functions
The model uses a **Contrastive MIL Loss** composed of three parts to enforce this logic:

1. **Ranking Loss with Hard Negative Mining (Primary)** 
   The model selects the Top-$K$ (e.g., $K=3$) highest-scoring segments from an anomalous bag, and the Top-$K$ highest-scoring segments from a normal bag (these are the "hard negatives"). It applies a hinge ranking loss to enforce that the maximum anomaly score in the anomalous video is at least *margin* higher than the maximum anomaly score in the normal video:
   $$ \mathcal{L}_{rank} = \max(0, \text{margin} - \max(S_{anom}) + \max(S_{norm})) $$

2. **Sparsity Loss ($\mathcal{L}_{sparsity}$)**
   An L1 penalty applied to the predicted scores. Since anomalies are assumed to be rare, temporally localized events (sparse in time), this encourages the model to predict low scores for the vast majority of segments.

3. **Smoothness Loss ($\mathcal{L}_{smoothness}$)**
   An L2 penalty on the differences between anomaly scores of adjacent segments. Anomalies usually happen over contiguous frames, so this prevents jagged, rapidly oscillating predictions.

---

## 4. Evaluation Metrics

### 1. Area Under the Receiver Operating Characteristic Curve (AUC-ROC)
The standard metric for Video Anomaly Detection. Since this is an imbalanced classification problem (most frames/videos are normal), accuracy is not a reliable metric.

Depending on the available labels, AUC is calculated in two ways:
- **Video-Level AUC:** If frame-level annotations aren't available, we take the maximum segment score from a video as its "video-level anomaly score" and compute AUC against the video-level ground truth (0 or 1).
- **Frame-Level AUC:** If exact start/end timestamps of the anomaly are known (via `UCF_Crime_temporal_annotations.txt`), the segment scores are interpolated to match the total number of frames in the video. The AUC is then calculated by comparing the predicted score of every single frame against its ground-truth label (1 if it falls within the anomalous timestamp window, 0 otherwise). 

STA-MIL is designed to achieve $\ge 87\%$ AUC on UCF-Crime, matching or exceeding prior state-of-the-art methods like TEF-VAD, RTFM, and MGFN.

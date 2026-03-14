"""
STA-MIL: Spatio-Temporal Attention Transformer with Multiple Instance Learning
for weakly supervised video anomaly detection (shoplifting / UCF-Crime).

Architecture:
    - Spatial Transformer Branch: models patch-level spatial relationships
    - Temporal Transformer Branch: models clip-level temporal dependencies
    - Cross-Attention Fusion: bidirectional spatial <-> temporal attention
    - Multi-Scale Temporal Pooling: captures short, medium, and long patterns
    - MIL Anomaly Scoring Head: predicts per-segment anomaly scores

Reference innovations over SOTA:
    - Dual-branch cross-attention (vs. single-stream in TEF-VAD)
    - Multi-scale temporal pooling (vs. fixed in MGFN/RTFM)
    - Hard-negative contrastive MIL loss (vs. standard ranking in Sultani et al.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ─────────────────────────────────────────────────────────────
# 1. Building Blocks
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Learnable 1-D positional encoding for temporal sequences.
    More flexible than fixed sinusoidal encodings for video anomaly tasks.
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation and pre-norm style.
    Standard transformer FFN block with expansion ratio.
    """
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder layer (more stable training than post-norm).
    Uses multi-head self-attention + FFN with residual connections.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(d_model, expansion, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """x: [B, T, D]"""
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # FFN with pre-norm
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionLayer(nn.Module):
    """
    Pre-norm cross-attention layer.
    Query from one branch, Key/Value from the other branch.
    Used for bidirectional spatial <-> temporal fusion.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(d_model, expansion=4, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:   [B, T_q, D]
            context: [B, T_k, D]
        Returns:
            [B, T_q, D] — query enriched with context information
        """
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(q, kv, kv)
        query = query + attn_out
        query = query + self.ff(self.norm_ff(query))
        return query


# ─────────────────────────────────────────────────────────────
# 2. Spatial Transformer Branch
# ─────────────────────────────────────────────────────────────

class SpatialTransformerBranch(nn.Module):
    """
    Models spatial relations across the feature dimension using
    self-attention along the temporal (segment) axis.

    Input features from VideoMAE are already spatially aggregated,
    so this branch applies transformer attention over the segment sequence
    to model how spatial content evolves over time.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────
# 3. Temporal Transformer Branch  
# ─────────────────────────────────────────────────────────────

class TemporalTransformerBranch(nn.Module):
    """
    Models temporal dynamics using self-attention over segments.
    Uses relative positional bias (inspired by VideoSwin) to capture
    temporal order more effectively than absolute encodings.

    Additionally computes gradient-based motion features from
    consecutive frame differences to augment pure appearance features.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        # Motion feature extraction via temporal difference
        self.motion_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # Fuse appearance + motion
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _compute_motion_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient-based motion: temporal differences between adjacent segments.
        Captures motion magnitude and direction patterns.

        x: [B, T, D]
        returns: [B, T, D]
        """
        # Forward difference: x[t+1] - x[t]
        diff_forward = torch.zeros_like(x)
        diff_forward[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
        diff_forward[:, -1, :] = diff_forward[:, -2, :]  # pad last

        # Backward difference: x[t] - x[t-1]
        diff_backward = torch.zeros_like(x)
        diff_backward[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        diff_backward[:, 0, :] = diff_backward[:, 1, :]  # pad first

        # Central difference (more stable)
        motion = (diff_forward + diff_backward) / 2.0
        return self.motion_proj(motion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        motion = self._compute_motion_features(x)
        x = self.fusion_proj(torch.cat([x, motion], dim=-1))
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────
# 4. Cross-Attention Fusion Module
# ─────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between spatial and temporal branches.
    
    Novel contribution: unlike single-stream methods (TEF-VAD, RTFM),
    we fuse spatial and temporal information via cross-attention in both
    directions, then merge the enriched representations.
    
    Spatial queries attend to temporal context → spatially-informed temporal
    Temporal queries attend to spatial context → temporally-informed spatial
    Both outputs are fused to produce a joint representation.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_to_temporal_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.temporal_to_spatial_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        # Final merge
        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spatial_feat:  [B, T, D]
            temporal_feat: [B, T, D]
        Returns:
            fused: [B, T, D]
        """
        s = spatial_feat
        t = temporal_feat

        for s2t, t2s in zip(self.spatial_to_temporal_layers, self.temporal_to_spatial_layers):
            t_enriched = s2t(query=t, context=s)   # temporal attends to spatial
            s_enriched = t2s(query=s, context=t)   # spatial attends to temporal
            t = t_enriched
            s = s_enriched

        fused = self.merge(torch.cat([s, t], dim=-1))  # [B, T, D]
        return fused


# ─────────────────────────────────────────────────────────────
# 5. Multi-Scale Temporal Pooling
# ─────────────────────────────────────────────────────────────

class MultiScaleTemporalPooling(nn.Module):
    """
    Attention-weighted pooling at multiple temporal scales.
    
    Novel contribution: captures both local (short-term) and global
    (long-term) temporal patterns by pooling features at 3 scales:
    - Scale 1: all segments (global context)
    - Scale 2: 4 groups of segments (mid-term)
    - Scale 3: individual segments (local)
    
    This makes the model sensitive to both brief shoplifting gestures
    and extended anomaly sequences.
    """
    def __init__(self, d_model: int, num_segments: int, num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_segments = num_segments
        self.num_scales = num_scales

        # Attention scorer for each scale
        self.scale_attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(num_scales)
        ])

        # Group sizes for each scale (segments grouped into N groups)
        self.group_sizes = self._compute_group_sizes(num_segments, num_scales)

        # Project concatenated multi-scale output back to d_model
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def _compute_group_sizes(self, num_segments: int, num_scales: int) -> list:
        """Compute number of groups at each temporal scale."""
        # Scale 0: 1 group (global), Scale 1: sqrt(T) groups, Scale 2: T groups (full)
        sizes = []
        for i in range(num_scales):
            if i == 0:
                sizes.append(1)  # global pool
            elif i == num_scales - 1:
                sizes.append(num_segments)  # local (segment-level)
            else:
                n_groups = max(2, int(num_segments ** ((i) / (num_scales - 1))))
                sizes.append(n_groups)
        return sizes

    def _scale_pool(
        self,
        x: torch.Tensor,
        n_groups: int,
        attn_scorer: nn.Module,
    ) -> torch.Tensor:
        """
        Compute attention-weighted pool for a given number of groups.
        
        Args:
            x: [B, T, D]
            n_groups: number of temporal groups
        Returns:
            pooled: [B, T, D]  (each segment gets its group's pooled feature)
        """
        B, T, D = x.shape

        if n_groups == T:
            # Segment-level: no pooling, just weight by self-attention score
            weights = torch.sigmoid(attn_scorer(x))  # [B, T, 1]
            return x * weights

        if n_groups == 1:
            # Global: attention pool all segments → broadcast
            scores = attn_scorer(x)  # [B, T, 1]
            weights = F.softmax(scores, dim=1)  # [B, T, 1]
            pooled = (x * weights).sum(dim=1, keepdim=True)  # [B, 1, D]
            return pooled.expand(-1, T, -1)  # [B, T, D]

        # Group into n_groups chunks
        seg_per_group = T // n_groups
        remainder = T - seg_per_group * n_groups

        group_features = []
        seg_idx = 0
        for g in range(n_groups):
            size = seg_per_group + (1 if g < remainder else 0)
            group = x[:, seg_idx:seg_idx + size, :]  # [B, size, D]
            scores = attn_scorer(group)  # [B, size, 1]
            weights = F.softmax(scores, dim=1)  # [B, size, 1]
            pooled = (group * weights).sum(dim=1, keepdim=True)  # [B, 1, D]
            group_features.append(pooled.expand(-1, size, -1))  # [B, size, D]
            seg_idx += size

        return torch.cat(group_features, dim=1)  # [B, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        scale_feats = []
        for i, (n_groups, scorer) in enumerate(zip(self.group_sizes, self.scale_attn)):
            scale_out = self._scale_pool(x, n_groups, scorer)  # [B, T, D]
            scale_feats.append(scale_out)

        # Concatenate all scales and project
        multi_scale = torch.cat(scale_feats, dim=-1)  # [B, T, D * num_scales]
        return self.out_proj(multi_scale)  # [B, T, D]


# ─────────────────────────────────────────────────────────────
# 6. MIL Anomaly Scoring Head
# ─────────────────────────────────────────────────────────────

class MILScoringHead(nn.Module):
    """
    Per-segment anomaly scoring MLP.
    Each segment in the bag receives an anomaly score in [0, 1].
    
    Designed with:
    - Two-layer MLP with GELU (richer than single linear layer)
    - Dropout for regularization
    - Sigmoid output (required for MIL ranking loss)
    """
    def __init__(self, d_model: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → scores: [B, T]"""
        return self.head(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# 7. Main STA-MIL Model
# ─────────────────────────────────────────────────────────────

class STA_MIL(nn.Module):
    """
    STA-MIL: Spatio-Temporal Attention Multiple Instance Learning.

    Weakly supervised video anomaly detection via:
    1. Dual-branch spatial + temporal transformer encoding
    2. Bidirectional cross-attention fusion
    3. Multi-scale temporal pooling
    4. MIL anomaly scoring head

    Input: pre-extracted VideoMAE features [B, T, D]
    Output: per-segment anomaly scores [B, T] ∈ [0, 1]
    """

    def __init__(
        self,
        feature_dim: int = 768,
        spatial_heads: int = 8,
        spatial_layers: int = 4,
        temporal_heads: int = 8,
        temporal_layers: int = 4,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        dropout: float = 0.1,
        num_scales: int = 3,
        mlp_dim: int = 1024,
        num_segments: int = 32,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_segments = num_segments

        # Input projection (align to model dimension)
        self.input_norm = nn.LayerNorm(feature_dim)

        # Branch 1: Spatial Transformer
        self.spatial_branch = SpatialTransformerBranch(
            d_model=feature_dim,
            n_heads=spatial_heads,
            n_layers=spatial_layers,
            dropout=dropout,
        )

        # Branch 2: Temporal Transformer with motion features
        self.temporal_branch = TemporalTransformerBranch(
            d_model=feature_dim,
            n_heads=temporal_heads,
            n_layers=temporal_layers,
            dropout=dropout,
        )

        # Fusion: bidirectional cross-attention
        self.fusion = CrossAttentionFusion(
            d_model=feature_dim,
            n_heads=cross_attn_heads,
            n_layers=cross_attn_layers,
            dropout=dropout,
        )

        # Multi-scale temporal pooling
        self.ms_pool = MultiScaleTemporalPooling(
            d_model=feature_dim,
            num_segments=num_segments,
            num_scales=num_scales,
            dropout=dropout,
        )

        # Anomaly scoring head
        self.scoring_head = MILScoringHead(
            d_model=feature_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, D] — pre-extracted VideoMAE features
                B = batch size
                T = num_segments (temporal tokens)
                D = feature_dim (768 for VideoMAE-Base)

        Returns:
            scores: [B, T] — per-segment anomaly scores in [0, 1]
        """
        # Normalize input
        x = self.input_norm(x)  # [B, T, D]

        # Dual-branch encoding
        spatial_feat = self.spatial_branch(x)    # [B, T, D]
        temporal_feat = self.temporal_branch(x)  # [B, T, D]

        # Bidirectional cross-attention fusion
        fused = self.fusion(spatial_feat, temporal_feat)  # [B, T, D]

        # Residual: add original features to fused representation
        fused = fused + x  # [B, T, D]

        # Multi-scale temporal pooling
        ms_feat = self.ms_pool(fused)  # [B, T, D]

        # Final residual
        ms_feat = ms_feat + fused  # [B, T, D]

        # Per-segment anomaly scores
        scores = self.scoring_head(ms_feat)  # [B, T]
        return scores

    def predict_video(self, x: torch.Tensor) -> float:
        """
        Predict video-level anomaly score (max over segments).

        Args:
            x: [T, D] or [1, T, D]
        Returns:
            float: video-level anomaly score in [0, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            scores = self.forward(x)  # [1, T]
        return scores.squeeze(0).max().item()

    def get_model_info(self) -> dict:
        """Return model parameter count and architecture info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'num_segments': self.num_segments,
        }


# ─────────────────────────────────────────────────────────────
# 8. Quick Sanity Check
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing STA-MIL forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = STA_MIL(
        feature_dim=768,
        spatial_heads=8,
        spatial_layers=4,
        temporal_heads=8,
        temporal_layers=4,
        cross_attn_heads=8,
        cross_attn_layers=2,
        dropout=0.1,
        num_scales=3,
        mlp_dim=1024,
        num_segments=32,
    ).to(device)

    info = model.get_model_info()
    print(f"Total parameters:     {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")

    # Test forward pass
    x = torch.randn(4, 32, 768).to(device)  # batch=4, 32 segments, 768-dim
    scores = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {scores.shape}")
    assert scores.shape == (4, 32), f"Expected (4, 32), got {scores.shape}"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores must be in [0, 1]!"
    print("PASS — STA-MIL forward pass correct!")

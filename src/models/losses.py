"""
Loss functions for STA-MIL weakly supervised anomaly detection.

Main loss: Contrastive MIL Ranking Loss with hard negative mining.
Based on Sultani et al. (2018) + improvements from RTFM, MGFN, and TEF-VAD.

Components:
1. Ranking loss:    top-k anomalous segments > top-k normal segments + margin
2. Sparsity loss:   L1 regularization on anomaly scores (anomalies are rare)
3. Smooth loss:     L2 temporal smoothness (adjacent segments should agree)
4. Hard neg mining: select hardest normal segments for stronger gradient signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveMILLoss(nn.Module):
    """
    Contrastive Multiple Instance Learning Loss for weakly supervised
    video anomaly detection.

    For each pair of (anomalous_bag, normal_bag):
    - Selects top-k segments from anomalous bag (should have high scores)
    - Selects top-k segments from normal bag with hard negative mining
    - Applies ranking constraint: anomalous_score > normal_score + margin
    - Adds sparsity and temporal smoothness regularization

    Reference:
        Sultani et al., "Real-world Anomaly Detection in Surveillance Videos", CVPR 2018.
        Zhang et al., "Temporal Masked Autoencoders as Industrial Anomaly Detectors", 2024.
    """

    def __init__(
        self,
        topk: int = 3,
        margin: float = 100.0,
        lambda_sparsity: float = 8e-3,
        lambda_smooth: float = 8e-4,
        hard_neg_ratio: float = 0.3,
    ):
        """
        Args:
            topk: number of top segments selected per bag for MIL
            margin: ranking margin (anomalous score - normal score >= margin)
            lambda_sparsity: weight for L1 sparsity regularization
            lambda_smooth: weight for temporal smoothness regularization
            hard_neg_ratio: fraction of normal segments to treat as hard negatives
        """
        super().__init__()
        self.topk = topk
        self.margin = margin
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smooth = lambda_smooth
        self.hard_neg_ratio = hard_neg_ratio

    def _select_topk(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Select top-k scores from each bag.

        Args:
            scores: [B, T] anomaly scores
        Returns:
            topk_scores: [B, topk]
        """
        B, T = scores.shape
        k = min(self.topk, T)
        topk_scores, _ = torch.topk(scores, k, dim=1)  # [B, k]
        return topk_scores

    def _hard_negative_mining(
        self,
        normal_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select hard negatives: normal segments with highest anomaly scores.
        These are the most challenging examples for the model.

        Args:
            normal_scores: [B, T] scores for normal videos
        Returns:
            hard_neg_scores: [B, hard_k]
        """
        B, T = normal_scores.shape
        hard_k = max(1, int(T * self.hard_neg_ratio))
        hard_scores, _ = torch.topk(normal_scores, hard_k, dim=1)  # [B, hard_k]
        return hard_scores

    def ranking_loss(
        self,
        anom_scores: torch.Tensor,
        norm_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hinge ranking loss: max(0, margin - max_anom + max_norm).

        Ensures the maximum anomaly score in anomalous videos
        is at least 'margin' higher than in normal videos.

        Args:
            anom_scores: [B_a, T] scores for anomalous videos
            norm_scores: [B_n, T] scores for normal videos
        Returns:
            scalar loss
        """
        # Top-k from anomalous bags
        anom_topk = self._select_topk(anom_scores)  # [B_a, k]
        anom_max = anom_topk.mean(dim=1)  # [B_a] — mean of top-k

        # Hard negatives from normal bags
        hard_neg = self._hard_negative_mining(norm_scores)  # [B_n, hard_k]
        norm_max = hard_neg.mean(dim=1)  # [B_n]

        # Compute loss for all pairs (broadcast)
        # anom_max: [B_a, 1], norm_max: [1, B_n]
        anom_exp = anom_max.unsqueeze(1)   # [B_a, 1]
        norm_exp = norm_max.unsqueeze(0)   # [1, B_n]

        pairs = self.margin - anom_exp + norm_exp  # [B_a, B_n]
        loss = F.relu(pairs).mean()
        return loss

    def sparsity_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        L1 sparsity: encourage most segments to have low anomaly scores.
        Anomalous events are temporally sparse — only brief periods.

        Args:
            scores: [B, T] anomaly scores
        Returns:
            scalar loss
        """
        return scores.mean()

    def smoothness_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Temporal smoothness: consecutive segments should have similar scores.
        Prevents jagged, oscillating predictions.

        Args:
            scores: [B, T] anomaly scores
        Returns:
            scalar loss
        """
        diff = scores[:, 1:] - scores[:, :-1]  # [B, T-1]
        return (diff ** 2).mean()

    def forward(
        self,
        anom_scores: torch.Tensor,
        norm_scores: torch.Tensor,
    ) -> dict:
        """
        Compute total contrastive MIL loss.

        Args:
            anom_scores: [B_a, T] — scores for anomalous video bags
            norm_scores: [B_n, T] — scores for normal video bags

        Returns:
            dict with keys: 'total', 'ranking', 'sparsity', 'smooth'
        """
        # 1. Ranking loss (primary)
        rank_loss = self.ranking_loss(anom_scores, norm_scores)

        # 2. Sparsity regularization on both (mostly on anomalous)
        sparsity = (
            self.sparsity_loss(anom_scores) +
            self.sparsity_loss(norm_scores)
        ) / 2.0

        # 3. Temporal smoothness on both
        smooth = (
            self.smoothness_loss(anom_scores) +
            self.smoothness_loss(norm_scores)
        ) / 2.0

        total = (
            rank_loss +
            self.lambda_sparsity * sparsity +
            self.lambda_smooth * smooth
        )

        return {
            'total': total,
            'ranking': rank_loss.detach(),
            'sparsity': sparsity.detach(),
            'smooth': smooth.detach(),
        }


class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross Entropy for optional supervised signal
    (used when some frame-level annotations are available).
    Reduces weight on easy negatives, focuses on hard cases.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target.float(), reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


if __name__ == '__main__':
    print("Testing ContrastiveMILLoss...")
    criterion = ContrastiveMILLoss(topk=3, margin=100.0)

    # Simulate batch
    anom_scores = torch.rand(8, 32)  # 8 anomalous videos, 32 segments
    norm_scores = torch.rand(8, 32)  # 8 normal videos, 32 segments

    losses = criterion(anom_scores, norm_scores)
    print(f"Total loss:    {losses['total'].item():.4f}")
    print(f"Ranking loss:  {losses['ranking'].item():.4f}")
    print(f"Sparsity loss: {losses['sparsity'].item():.4f}")
    print(f"Smooth loss:   {losses['smooth'].item():.4f}")

    # Test backprop
    losses['total'].backward()
    print("Backward pass: PASS")

"""Loss functions for floorplan segmentation with class imbalance handling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, TVERSKY_ALPHA, TVERSKY_BETA


class TverskyLoss(nn.Module):
    """Tversky loss for segmentation with class imbalance.

    Generalizes Dice loss with separate alpha/beta weights for FP and FN.
    Higher beta emphasizes recall (fewer missed pixels).
    """

    def __init__(
        self,
        alpha: float = TVERSKY_ALPHA,
        beta: float = TVERSKY_BETA,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Tversky loss.

        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) integer class labels
        Returns:
            Scalar loss value
        """
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets -> (B, C, H, W)
        one_hot = F.one_hot(targets, num_classes=NUM_CLASSES)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute TP, FP, FN per class (sum over batch, height, width)
        dims = (0, 2, 3)
        tp = (probs * one_hot).sum(dim=dims)
        fp = (probs * (1 - one_hot)).sum(dim=dims)
        fn = ((1 - probs) * one_hot).sum(dim=dims)

        # Tversky index per class
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Skip background (class 0), return 1 - mean of foreground classes
        return 1.0 - tversky[1:].mean()


class CombinedLoss(nn.Module):
    """Weighted combination of CrossEntropy and Tversky losses."""

    def __init__(self, ce_weight: float = 0.5, tversky_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.tversky_loss = TverskyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) integer class labels
        Returns:
            Scalar loss value
        """
        ce = self.ce_loss(logits, targets)
        tversky = self.tversky_loss(logits, targets)
        return self.ce_weight * ce + self.tversky_weight * tversky

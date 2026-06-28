"""
VGG-16 Perceptual Loss (Feature-Matching Loss)
===============================================

Computes the L1 distance between VGG-16 feature activations of the
predicted and target images.  This encourages the network to produce
outputs that are perceptually similar (structurally coherent) rather
than just pixel-accurate.

Implementation details:
  - Uses pre-relu features from VGG-16 layers:
      relu1_2  (conv1_2):  low-level edges and textures
      relu2_2  (conv2_2):  mid-level textures and patterns
      relu3_3  (conv3_3):  higher-level structural elements
      relu4_3  (conv4_3):  semantic/structural features
  - Input images are assumed to be in [0, 1] range and are normalized
    to ImageNet statistics internally.
  - VGG weights are frozen (no gradient computation through VGG).
  - The loss is the weighted sum of L1 distances at each layer.

NOTE on color hue preservation:
  The perceptual loss is ONLY used during training.  At inference time,
  no VGG network runs — only ESPCN/FlowNet.  The ImageNet normalization
  inside this loss does NOT affect inference normalization, which is
  strictly [0,1] → [0,1] with no mean/std shifting.

References:
  - Johnson et al., "Perceptual Losses for Real-Time Style Transfer
    and Super-Resolution", ECCV 2016.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


# ImageNet normalization constants
# Applied internally to VGG inputs — NOT to the SR/flow models themselves
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from VGG-16.

    Parameters
    ----------
    layer_indices : list[int]
        Indices into the VGG-16 features Sequential at which to extract
        activations.  Default indices correspond to:
          3  → relu1_2 (after conv1_2 + ReLU)
          8  → relu2_2 (after conv2_2 + ReLU)
          15 → relu3_3 (after conv3_3 + ReLU)
          22 → relu4_3 (after conv4_3 + ReLU)
    """

    def __init__(
        self,
        layer_indices: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        if layer_indices is None:
            layer_indices = [3, 8, 15, 22]

        self.layer_indices = sorted(layer_indices)
        max_layer = max(layer_indices) + 1

        # Load VGG-16 features (only up to the deepest layer we need)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])

        # Freeze all VGG weights — they are not trainable
        for param in self.features.parameters():
            param.requires_grad = False

        # Set to eval mode (disables dropout, fixes BN stats)
        self.features.eval()

        # Register ImageNet normalization constants as buffers
        # (so they move to GPU with .to(device))
        self.register_buffer("mean", _IMAGENET_MEAN.clone())
        self.register_buffer("std", _IMAGENET_STD.clone())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract feature maps at specified VGG-16 layers.

        Parameters
        ----------
        x : torch.Tensor
            Input image, [B, 3, H, W], range [0, 1].

        Returns
        -------
        list[torch.Tensor]
            Feature maps at each specified layer index.
        """
        # Normalize to ImageNet statistics
        x = (x - self.mean) / self.std

        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)

        return features

    def train(self, mode: bool = True) -> "VGGFeatureExtractor":
        """Override train to keep VGG always in eval mode."""
        # We never want VGG in training mode
        return super().train(False)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-16 feature matching.

    Computes weighted L1 distance between VGG-16 feature activations
    of predicted and target images.

    Parameters
    ----------
    layer_weights : list[float] or None
        Per-layer weights for the feature matching loss.
        Default: [0.1, 0.1, 1.0, 1.0] — emphasizes deeper layers
        for structural similarity over low-level texture matching.
    reduction : str
        'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        layer_weights: Optional[list[float]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if layer_weights is None:
            # Emphasize deeper features (more structural, less textural)
            layer_weights = [0.1, 0.1, 1.0, 1.0]

        self.layer_weights = layer_weights
        self.reduction = reduction

        # Build VGG feature extractor
        self.vgg = VGGFeatureExtractor(layer_indices=[3, 8, 15, 22])

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output, [B, 3, H, W], range [0, 1].
        target : torch.Tensor
            Ground truth, [B, 3, H, W], range [0, 1].

        Returns
        -------
        torch.Tensor
            Scalar perceptual loss.
        """
        # Extract VGG features (no gradient through VGG)
        with torch.no_grad():
            target_features = self.vgg(target)

        pred_features = self.vgg(prediction)

        # Weighted L1 distance at each layer
        total_loss = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)

        for i, (pred_f, target_f) in enumerate(zip(pred_features, target_features)):
            weight = self.layer_weights[i] if i < len(self.layer_weights) else 1.0

            if self.reduction == "mean":
                layer_loss = (pred_f - target_f.detach()).abs().mean()
            else:
                layer_loss = (pred_f - target_f.detach()).abs().sum()

            total_loss = total_loss + weight * layer_loss

        return total_loss


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    criterion = PerceptualLoss().to(device)

    pred = torch.rand(2, 3, 128, 128, device=device, requires_grad=True)
    target = torch.rand(2, 3, 128, 128, device=device)

    loss = criterion(pred, target)
    print(f"Perceptual loss (random vs random): {loss.item():.6f}")

    # Verify gradient flows to prediction (not through VGG)
    loss.backward()
    assert pred.grad is not None, "Gradient should flow to prediction"
    print(f"Prediction gradient norm: {pred.grad.norm().item():.6f}")

    # Identical images should have near-zero perceptual loss
    with torch.no_grad():
        same = torch.rand(2, 3, 128, 128, device=device)
    same_pred = same.clone().requires_grad_(True)
    same_loss = criterion(same_pred, same)
    print(f"Perceptual loss (identical):        {same_loss.item():.8f}")
    assert same_loss.item() < 1e-5, "Identical images should have ~0 perceptual loss"

    # Verify VGG is frozen
    for param in criterion.vgg.parameters():
        assert not param.requires_grad, "VGG params should be frozen"

    print("✓ Perceptual loss self-test passed.")

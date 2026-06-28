"""
Charbonnier Loss (Smooth L1 / Pseudo-Huber Loss)
=================================================

    L(x, y) = sqrt((x - y)² + ε²)

A differentiable approximation to L1 loss that is smooth at the origin.
Provides:
  - L1-like robustness to outliers (unlike L2/MSE which over-penalizes them)
  - Smooth gradient near zero (unlike L1 which has undefined gradient at 0)
  - Better convergence for super-resolution than MSE (sharper outputs)

The epsilon parameter controls the curvature near zero:
  - Smaller ε → closer to true L1 (sharper but noisier gradients)
  - Larger ε  → closer to L2 (smoother gradients, slightly blurrier results)
  - Default ε = 1e-3 is standard for super-resolution tasks.

References:
  - Charbonnier et al., "Two deterministic half-quadratic regularization
    algorithms for computed imaging", ICIP 1994.
  - Lai et al., "Fast and Accurate Image Super-Resolution with Deep
    Laplacian Pyramid Networks", TPAMI 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: sqrt((pred - target)² + ε²)

    Parameters
    ----------
    epsilon : float
        Smoothing constant.  Default 1e-3.
    reduction : str
        Reduction mode: 'mean' (default), 'sum', or 'none'.
    """

    def __init__(self, epsilon: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.epsilon_sq = epsilon * epsilon
        self.reduction = reduction

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Charbonnier loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output, any shape [B, C, H, W].
        target : torch.Tensor
            Ground truth, same shape as prediction.

        Returns
        -------
        torch.Tensor
            Scalar loss (if reduction='mean' or 'sum') or per-element loss.
        """
        diff_sq = (prediction - target).pow(2)
        loss = torch.sqrt(diff_sq + self.epsilon_sq)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    criterion = CharbonnierLoss(epsilon=1e-3)

    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    loss = criterion(pred, target)
    print(f"Charbonnier loss: {loss.item():.6f}")

    # Verify: identical inputs should give ≈ epsilon
    zero_loss = criterion(pred, pred)
    expected_zero = 1e-3  # sqrt(0 + ε²) = ε
    print(f"Zero-diff loss:  {zero_loss.item():.6f}  (expected ≈ {expected_zero})")
    assert abs(zero_loss.item() - expected_zero) < 1e-5, "Zero-diff loss incorrect"

    # Verify gradient exists at zero
    p = torch.zeros(1, requires_grad=True)
    t = torch.zeros(1)
    l = criterion(p, t)
    l.backward()
    assert p.grad is not None and p.grad.abs().item() < 1e-2, "Gradient at zero should be near-zero"

    print("✓ Charbonnier loss self-test passed.")

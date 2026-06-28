"""
FlowDataset — Synthetic Motion Triplet Dataset for FlowNet Training
====================================================================

Generates frame triplets (Frame_N-1, Frame_N, Frame_N+1) from static images
by applying synthetic random affine transformations to simulate inter-frame
camera/object motion.

Since DIV2K contains only static images (no temporal sequences), we simulate
motion by applying parameterized affine warps:
  Frame_N   = random crop from HR image
  Frame_N-1 = affine warp of Frame_N with parameters (-Δ)   (reverse motion)
  Frame_N+1 = affine warp of Frame_N with parameters (+Δ)   (forward motion)

This ensures that Frame_N is the exact temporal midpoint between N-1 and N+1,
which is the ground truth for frame interpolation at timestep=0.5.

Synthetic motion parameters (per triplet):
  - Translation: ±[0, 16] pixels horizontal & vertical
  - Rotation: ±[0, 5] degrees
  - Scale: ±[0, 0.05] relative change

The motion is applied as an affine_grid + grid_sample pipeline, which is
differentiable and produces smooth, realistic-looking inter-frame motion.

Tensor shapes (all outputs):
  frame_prev:  [3, H, W]  (Frame N-1)
  frame_mid:   [3, H, W]  (Frame N  — ground truth)
  frame_next:  [3, H, W]  (Frame N+1)

All values in [0, 1] float32.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _random_affine_matrix(
    max_translate: float = 16.0,
    max_rotate_deg: float = 5.0,
    max_scale_delta: float = 0.05,
    image_h: int = 256,
    image_w: int = 256,
) -> torch.Tensor:
    """
    Generate a random 2×3 affine transformation matrix.

    The matrix transforms pixel coordinates:
        [x', y', 1]^T = M · [x, y, 1]^T

    Parameters
    ----------
    max_translate : float
        Maximum translation in pixels (uniform random).
    max_rotate_deg : float
        Maximum rotation in degrees (uniform random).
    max_scale_delta : float
        Maximum relative scale change (uniform random around 1.0).
    image_h : int
        Image height (for normalizing translation to [-1, 1] grid coords).
    image_w : int
        Image width.

    Returns
    -------
    torch.Tensor
        Affine matrix of shape [2, 3], suitable for torch.affine_grid
        (which operates in normalized [-1, 1] coordinates).
    """
    # Random parameters (half-delta: will be applied as +Δ and -Δ)
    tx = random.uniform(-max_translate, max_translate)
    ty = random.uniform(-max_translate, max_translate)
    angle_rad = math.radians(random.uniform(-max_rotate_deg, max_rotate_deg))
    scale = 1.0 + random.uniform(-max_scale_delta, max_scale_delta)

    # Build rotation + scale matrix
    cos_a = math.cos(angle_rad) * scale
    sin_a = math.sin(angle_rad) * scale

    # Normalize translation to [-1, 1] grid coordinates
    tx_norm = tx / (image_w / 2.0)
    ty_norm = ty / (image_h / 2.0)

    # Affine matrix for grid_sample (operates in normalized coords)
    # [cos  -sin  tx]
    # [sin   cos  ty]
    affine = torch.tensor(
        [[cos_a, -sin_a, tx_norm],
         [sin_a,  cos_a, ty_norm]],
        dtype=torch.float32,
    )

    return affine


def _invert_affine(affine: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a 2×3 affine matrix.

    For a matrix [R | t] where R is 2×2 rotation+scale and t is 2×1 translation,
    the inverse is [R⁻¹ | -R⁻¹·t].

    Parameters
    ----------
    affine : torch.Tensor
        Affine matrix of shape [2, 3].

    Returns
    -------
    torch.Tensor
        Inverse affine matrix of shape [2, 3].
    """
    R = affine[:, :2]  # [2, 2]
    t = affine[:, 2:]  # [2, 1]

    R_inv = torch.inverse(R)  # [2, 2]
    t_inv = -R_inv @ t        # [2, 1]

    return torch.cat([R_inv, t_inv], dim=1)  # [2, 3]


def _apply_affine(image: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """
    Apply an affine transformation to an image using grid_sample.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor, [1, C, H, W] or [C, H, W].
    affine : torch.Tensor
        Affine matrix, [2, 3].

    Returns
    -------
    torch.Tensor
        Transformed image, same shape as input.
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True

    _, _, h, w = image.shape

    # Create sampling grid from affine matrix
    grid = F.affine_grid(
        affine.unsqueeze(0),  # [1, 2, 3]
        size=[1, image.shape[1], h, w],
        align_corners=False,
    )

    # Sample using bilinear interpolation
    result = F.grid_sample(
        image, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    if squeeze:
        result = result.squeeze(0)

    return result


class FlowDataset(Dataset):
    """
    Synthetic motion triplet dataset for optical flow / frame interpolation.

    Generates (frame_prev, frame_mid, frame_next) from static HR images
    using random affine transformations to simulate motion.

    Parameters
    ----------
    hr_dir : str or Path
        Directory containing high-resolution images.
    patch_size : int
        Crop size for training patches.  Default: 256.
    max_translate : float
        Maximum synthetic translation in pixels.  Default: 16.0.
    max_rotate_deg : float
        Maximum synthetic rotation in degrees.  Default: 5.0.
    max_scale_delta : float
        Maximum relative scale change.  Default: 0.05.
    augment : bool
        Enable random horizontal/vertical flip.  Default: True.
    max_images : int or None
        Limit number of images (for debugging).
    repeat : int
        Repeat dataset N times per epoch (different crops/transforms each time).
        Default: 4 (since each image generates unique triplet each time).
    """

    def __init__(
        self,
        hr_dir: str | Path,
        patch_size: int = 256,
        max_translate: float = 16.0,
        max_rotate_deg: float = 5.0,
        max_scale_delta: float = 0.05,
        augment: bool = True,
        max_images: Optional[int] = None,
        repeat: int = 4,
    ) -> None:
        super().__init__()

        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.max_translate = max_translate
        self.max_rotate_deg = max_rotate_deg
        self.max_scale_delta = max_scale_delta
        self.augment = augment
        self.repeat = max(1, repeat)

        # Discover images
        self.image_paths: list[Path] = sorted(
            p
            for p in self.hr_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.hr_dir}"
            )

        # Use a slightly larger internal crop to avoid border artefacts
        # from the affine warp, then center-crop to patch_size
        self._margin = int(max_translate * 1.5) + 8
        self._internal_size = patch_size + 2 * self._margin

    def __len__(self) -> int:
        return len(self.image_paths) * self.repeat

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (frame_prev, frame_mid, frame_next)
            All: [3, patch_size, patch_size], float32, range [0, 1].
        """
        img_index = index % len(self.image_paths)
        img_path = self.image_paths[img_index]

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Ensure image is large enough for internal crop
        min_size = self._internal_size
        if img_h < min_size or img_w < min_size:
            scale = max(min_size / img_h, min_size / img_w) * 1.01
            new_h = int(img_h * scale)
            new_w = int(img_w * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            img_w, img_h = image.size

        # Random crop (larger than patch_size to allow warp margin)
        top = random.randint(0, img_h - self._internal_size)
        left = random.randint(0, img_w - self._internal_size)

        crop = TF.crop(image, top, left, self._internal_size, self._internal_size)

        # Convert to tensor [3, internal_size, internal_size]
        crop_tensor = TF.to_tensor(crop)  # [0, 1]

        # ── Generate synthetic motion ────────────────────────────────────
        # Create random affine for forward motion (N → N+1)
        forward_affine = _random_affine_matrix(
            max_translate=self.max_translate,
            max_rotate_deg=self.max_rotate_deg,
            max_scale_delta=self.max_scale_delta,
            image_h=self._internal_size,
            image_w=self._internal_size,
        )

        # Inverse affine for backward motion (N → N-1)
        backward_affine = _invert_affine(forward_affine)

        # Apply transforms:
        # frame_mid   = original crop (the ground truth midpoint)
        # frame_next  = crop warped by forward affine (future frame)
        # frame_prev  = crop warped by backward affine (past frame)
        frame_next = _apply_affine(crop_tensor, forward_affine)
        frame_prev = _apply_affine(crop_tensor, backward_affine)
        frame_mid = crop_tensor  # unmodified center frame

        # ── Center crop to remove warp-margin border artefacts ───────────
        m = self._margin
        frame_prev = frame_prev[:, m:-m, m:-m].contiguous()
        frame_mid = frame_mid[:, m:-m, m:-m].contiguous()
        frame_next = frame_next[:, m:-m, m:-m].contiguous()

        # ── Augmentation ─────────────────────────────────────────────────
        if self.augment:
            # Horizontal flip (must flip all three consistently)
            if random.random() > 0.5:
                frame_prev = frame_prev.flip(2)  # flip W dimension
                frame_mid = frame_mid.flip(2)
                frame_next = frame_next.flip(2)

            # Vertical flip
            if random.random() > 0.5:
                frame_prev = frame_prev.flip(1)  # flip H dimension
                frame_mid = frame_mid.flip(1)
                frame_next = frame_next.flip(1)

        # Clamp to valid range (affine warp + border can produce tiny
        # overshoots due to floating point)
        frame_prev = frame_prev.clamp(0.0, 1.0)
        frame_mid = frame_mid.clamp(0.0, 1.0)
        frame_next = frame_next.clamp(0.0, 1.0)

        return frame_prev, frame_mid, frame_next


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    hr_dir = sys.argv[1] if len(sys.argv) > 1 else "../../trainData/train"

    if not os.path.isdir(hr_dir):
        print(f"⚠ Directory not found: {hr_dir}")
        print("  Usage: python flow_dataset.py <path_to_hr_images>")
        sys.exit(0)

    ds = FlowDataset(
        hr_dir=hr_dir,
        patch_size=256,
        max_translate=16.0,
        max_rotate_deg=5.0,
        max_scale_delta=0.05,
        augment=True,
        max_images=5,
        repeat=2,
    )

    print(f"FlowDataset | Images: {len(ds.image_paths)} | Effective len: {len(ds)}")

    prev, mid, nxt = ds[0]
    print(f"  frame_prev: {list(prev.shape)} range=[{prev.min():.3f}, {prev.max():.3f}]")
    print(f"  frame_mid:  {list(mid.shape)}  range=[{mid.min():.3f}, {mid.max():.3f}]")
    print(f"  frame_next: {list(nxt.shape)} range=[{nxt.min():.3f}, {nxt.max():.3f}]")

    assert prev.shape == (3, 256, 256), f"prev shape: {prev.shape}"
    assert mid.shape == (3, 256, 256), f"mid shape: {mid.shape}"
    assert nxt.shape == (3, 256, 256), f"next shape: {nxt.shape}"
    assert 0.0 <= prev.min() and prev.max() <= 1.0
    assert 0.0 <= mid.min() and mid.max() <= 1.0
    assert 0.0 <= nxt.min() and nxt.max() <= 1.0

    # Verify that mid and prev/next are different (motion was applied)
    diff_prev = (mid - prev).abs().mean().item()
    diff_next = (mid - nxt).abs().mean().item()
    print(f"  Mean diff (mid vs prev): {diff_prev:.4f}")
    print(f"  Mean diff (mid vs next): {diff_next:.4f}")
    assert diff_prev > 0.001, "prev should differ from mid (motion applied)"
    assert diff_next > 0.001, "next should differ from mid (motion applied)"

    print("\n✓ FlowDataset self-test passed.")

"""
UpscaleDataset — DIV2K HR/LR Paired Dataset for ESPCN Training
================================================================

Loads high-resolution images from the DIV2K dataset, extracts random crops,
and generates corresponding low-resolution inputs via bicubic downsampling.

Data pipeline:
  1. Load HR image from disk (PNG/JPEG, any size)
  2. Random crop to `patch_size × patch_size` (e.g., 256×256)
  3. Bicubic downsample by `scale_factor` → LR patch (128×128 for 2×)
  4. Random augmentation: horizontal flip + 90° rotation
  5. Normalize to [0, 1] float32
  6. Return (LR, HR) pair

Tensor shapes:
  LR:  [3, patch_size // scale_factor, patch_size // scale_factor]
  HR:  [3, patch_size, patch_size]

All pixel values in [0, 1] (float32).
No ImageNet mean/std normalization — raw pixel space only.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# Supported image extensions
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class UpscaleDataset(Dataset):
    """
    Paired HR/LR dataset for super-resolution training.

    Parameters
    ----------
    hr_dir : str or Path
        Directory containing high-resolution images.
    scale_factor : int
        Downsampling factor for generating LR inputs (2, 3, or 4).
    patch_size : int
        HR crop size in pixels.  Must be divisible by scale_factor.
        Default: 256.
    augment : bool
        Enable random horizontal flip + 90° rotation.  Default: True.
    max_images : int or None
        Limit the number of images loaded (for debugging).  None = all.
    repeat : int
        Number of times to repeat the dataset per epoch.  Useful for
        small datasets — each repeat draws a different random crop.
        Default: 1.
    """

    def __init__(
        self,
        hr_dir: str | Path,
        scale_factor: int = 2,
        patch_size: int = 256,
        augment: bool = True,
        max_images: Optional[int] = None,
        repeat: int = 1,
    ) -> None:
        super().__init__()

        self.hr_dir = Path(hr_dir)
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment
        self.repeat = max(1, repeat)

        if patch_size % scale_factor != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by "
                f"scale_factor ({scale_factor})"
            )

        # Discover all image files
        self.image_paths: list[Path] = sorted(
            p
            for p in self.hr_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.hr_dir} with extensions {_IMAGE_EXTENSIONS}"
            )

        self.lr_size = patch_size // scale_factor

    def __len__(self) -> int:
        return len(self.image_paths) * self.repeat

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (lr_patch, hr_patch) — both in [0, 1] float32, NCHW layout.
            lr_patch: [3, patch_size//S, patch_size//S]
            hr_patch: [3, patch_size, patch_size]
        """
        # Map repeated index back to actual image index
        img_index = index % len(self.image_paths)
        img_path = self.image_paths[img_index]

        # Load image as RGB PIL Image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # ── Random crop ──────────────────────────────────────────────────
        # Ensure image is large enough for the patch
        if img_h < self.patch_size or img_w < self.patch_size:
            # Resize to at least patch_size while maintaining aspect ratio
            scale = max(self.patch_size / img_h, self.patch_size / img_w) * 1.01
            new_h = int(img_h * scale)
            new_w = int(img_w * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            img_w, img_h = image.size

        # Random crop position
        top = random.randint(0, img_h - self.patch_size)
        left = random.randint(0, img_w - self.patch_size)

        # Crop HR patch
        hr_patch = TF.crop(image, top, left, self.patch_size, self.patch_size)

        # ── Generate LR via bicubic downsampling ─────────────────────────
        lr_patch = hr_patch.resize(
            (self.lr_size, self.lr_size),
            Image.BICUBIC,
        )

        # ── Random augmentation ──────────────────────────────────────────
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                hr_patch = TF.hflip(hr_patch)
                lr_patch = TF.hflip(lr_patch)

            # Random vertical flip
            if random.random() > 0.5:
                hr_patch = TF.vflip(hr_patch)
                lr_patch = TF.vflip(lr_patch)

            # Random 90° rotation (0, 90, 180, or 270 degrees)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                hr_patch = TF.rotate(hr_patch, angle, expand=False)
                lr_patch = TF.rotate(lr_patch, angle, expand=False)

        # ── Convert to tensor [0, 1] ────────────────────────────────────
        hr_tensor = TF.to_tensor(hr_patch)  # [3, H, W], float32, [0, 1]
        lr_tensor = TF.to_tensor(lr_patch)  # [3, H/S, W/S], float32, [0, 1]

        return lr_tensor, hr_tensor


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Allow specifying custom path via CLI
    hr_dir = sys.argv[1] if len(sys.argv) > 1 else "../../trainData/train"

    if not os.path.isdir(hr_dir):
        print(f"⚠ Directory not found: {hr_dir}")
        print("  Usage: python upscale_dataset.py <path_to_hr_images>")
        sys.exit(0)

    for scale in [2, 4]:
        ds = UpscaleDataset(
            hr_dir=hr_dir,
            scale_factor=scale,
            patch_size=256,
            augment=True,
            max_images=5,
            repeat=2,
        )

        print(f"\nScale {scale}× | Images: {len(ds.image_paths)} | "
              f"Effective len: {len(ds)}")

        lr, hr = ds[0]
        print(f"  LR: {list(lr.shape)} dtype={lr.dtype} "
              f"range=[{lr.min():.3f}, {lr.max():.3f}]")
        print(f"  HR: {list(hr.shape)} dtype={hr.dtype} "
              f"range=[{hr.min():.3f}, {hr.max():.3f}]")

        assert lr.shape == (3, 256 // scale, 256 // scale), f"LR shape wrong: {lr.shape}"
        assert hr.shape == (3, 256, 256), f"HR shape wrong: {hr.shape}"
        assert 0.0 <= lr.min() and lr.max() <= 1.0, "LR out of [0,1]"
        assert 0.0 <= hr.min() and hr.max() <= 1.0, "HR out of [0,1]"

    print("\n✓ UpscaleDataset self-test passed.")

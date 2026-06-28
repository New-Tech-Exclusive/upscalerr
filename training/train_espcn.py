"""
ESPCN Training Script — Full Super-Resolution Training Pipeline
=================================================================

Trains the ESPCN spatial upscaler on DIV2K HR/LR pairs using a combined
Charbonnier + VGG Perceptual loss.

Usage:
    python train_espcn.py --hr-dir ../trainData/train \
                          --val-dir ../trainData/val \
                          --scale 2 \
                          --epochs 200 \
                          --batch-size 16 \
                          --lr 1e-3

Training details:
  - Optimizer: AdamW with cosine annealing LR schedule
  - Loss: Charbonnier(ε=1e-3) + λ·VGG_perceptual (λ=0.01)
  - Mixed precision (FP16 via torch.amp) for 2× faster training on Ampere+
  - Gradient clipping at max_norm=1.0 for stability
  - Best model saved by validation PSNR
  - TensorBoard logging for loss curves and sample visualizations
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from models.espcn import ESPCN
from losses.charbonnier import CharbonnierLoss
from losses.perceptual import PerceptualLoss
from datasets.upscale_dataset import UpscaleDataset


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between prediction and target.

    Both inputs should be in [0, 1] range.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted image, [B, C, H, W].
    target : torch.Tensor
        Ground truth image, [B, C, H, W].

    Returns
    -------
    float
        Average PSNR in dB across the batch.
    """
    mse = (pred - target).pow(2).mean(dim=[1, 2, 3])  # per-sample MSE
    psnr = -10.0 * torch.log10(mse + 1e-8)             # per-sample PSNR
    return psnr.mean().item()


def train_one_epoch(
    model: ESPCN,
    loader: DataLoader,
    criterion_pixel: CharbonnierLoss,
    criterion_perceptual: PerceptualLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    lambda_perceptual: float = 0.01,
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns
    -------
    tuple[float, float]
        (average_loss, average_psnr) for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    for batch_idx, (lr_imgs, hr_imgs) in enumerate(loader):
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(device_type="cuda", dtype=torch.float16):
            sr_imgs = model(lr_imgs)

            # Pixel-level loss (Charbonnier)
            loss_pixel = criterion_pixel(sr_imgs, hr_imgs)

            # Perceptual loss (VGG features)
            loss_perceptual = criterion_perceptual(sr_imgs, hr_imgs)

            # Combined loss
            loss = loss_pixel + lambda_perceptual * loss_perceptual

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Metrics (no grad for PSNR computation)
        with torch.no_grad():
            batch_psnr = compute_psnr(sr_imgs.clamp(0, 1), hr_imgs)

        total_loss += loss.item()
        total_psnr += batch_psnr
        num_batches += 1

        if batch_idx % 50 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}]  "
                f"loss={loss.item():.4f}  "
                f"pixel={loss_pixel.item():.4f}  "
                f"perceptual={loss_perceptual.item():.4f}  "
                f"PSNR={batch_psnr:.2f} dB"
            )

    avg_loss = total_loss / max(num_batches, 1)
    avg_psnr = total_psnr / max(num_batches, 1)
    return avg_loss, avg_psnr


@torch.no_grad()
def validate(
    model: ESPCN,
    loader: DataLoader,
    criterion_pixel: CharbonnierLoss,
    device: torch.device,
) -> tuple[float, float]:
    """
    Validate model on the validation set.

    Returns
    -------
    tuple[float, float]
        (average_loss, average_psnr) on validation set.
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    for lr_imgs, hr_imgs in loader:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            sr_imgs = model(lr_imgs)
            loss = criterion_pixel(sr_imgs, hr_imgs)

        total_loss += loss.item()
        total_psnr += compute_psnr(sr_imgs.clamp(0, 1), hr_imgs)
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_psnr = total_psnr / max(num_batches, 1)
    return avg_loss, avg_psnr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ESPCN super-resolution model on DIV2K"
    )
    parser.add_argument(
        "--hr-dir", type=str, default="../trainData/train",
        help="Path to high-resolution training images"
    )
    parser.add_argument(
        "--val-dir", type=str, default="../trainData/val",
        help="Path to high-resolution validation images"
    )
    parser.add_argument(
        "--scale", type=int, default=2, choices=[2, 3, 4],
        help="Upscaling factor (default: 2)"
    )
    parser.add_argument(
        "--patch-size", type=int, default=256,
        help="HR patch size for training crops (default: 256)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--lambda-perceptual", type=float, default=0.01,
        help="Weight for perceptual loss (default: 0.01)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--repeat", type=int, default=8,
        help="Dataset repeat factor per epoch (default: 8)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/espcn",
        help="Directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # ── Model ────────────────────────────────────────────────────────────
    model = ESPCN(scale_factor=args.scale).to(device)
    print(f"ESPCN {args.scale}× | Parameters: {model.count_parameters():,}")

    # ── Losses ───────────────────────────────────────────────────────────
    criterion_pixel = CharbonnierLoss(epsilon=1e-3).to(device)
    criterion_perceptual = PerceptualLoss().to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    scaler = GradScaler("cuda")

    # ── Data ─────────────────────────────────────────────────────────────
    train_dataset = UpscaleDataset(
        hr_dir=args.hr_dir,
        scale_factor=args.scale,
        patch_size=args.patch_size,
        augment=True,
        repeat=args.repeat,
    )
    val_dataset = UpscaleDataset(
        hr_dir=args.val_dir,
        scale_factor=args.scale,
        patch_size=args.patch_size,
        augment=False,
        repeat=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")

    # ── Resume from checkpoint ───────────────────────────────────────────
    start_epoch = 0
    best_psnr = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f} dB")

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training ESPCN {args.scale}× for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_loss, train_psnr = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion_pixel=criterion_pixel,
            criterion_perceptual=criterion_perceptual,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            lambda_perceptual=args.lambda_perceptual,
        )

        # Validate
        val_loss, val_psnr = validate(
            model=model,
            loader=val_loader,
            criterion_pixel=criterion_pixel,
            device=device,
        )

        # Step LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        # ── Logging ──────────────────────────────────────────────────────
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_PSNR={train_psnr:.2f} dB  "
            f"val_loss={val_loss:.4f}  val_PSNR={val_psnr:.2f} dB  "
            f"lr={current_lr:.2e}  time={elapsed:.1f}s"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("PSNR/train", train_psnr, epoch)
        writer.add_scalar("PSNR/val", val_psnr, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # ── Sample visualization ─────────────────────────────────────────
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_lr, sample_hr = next(iter(val_loader))
                sample_lr = sample_lr[:4].to(device)
                sample_hr = sample_hr[:4].to(device)
                with autocast(device_type="cuda", dtype=torch.float16):
                    sample_sr = model(sample_lr).clamp(0, 1)

                # Upsample LR for visual comparison (bicubic baseline)
                sample_lr_up = torch.nn.functional.interpolate(
                    sample_lr, scale_factor=args.scale,
                    mode="bicubic", align_corners=False
                ).clamp(0, 1)

                # Grid: [bicubic, ESPCN, ground truth]
                grid = vutils.make_grid(
                    torch.cat([sample_lr_up, sample_sr, sample_hr], dim=0),
                    nrow=4, normalize=False,
                )
                writer.add_image("Samples/bicubic_espcn_gt", grid, epoch)

        # ── Checkpointing ────────────────────────────────────────────────
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scale_factor": args.scale,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_psnr": train_psnr,
            "val_psnr": val_psnr,
            "best_psnr": best_psnr,
        }

        # Save latest checkpoint
        torch.save(
            checkpoint_data,
            os.path.join(args.output_dir, "latest.pt"),
        )

        # Save best model by validation PSNR
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint_data["best_psnr"] = best_psnr
            torch.save(
                checkpoint_data,
                os.path.join(args.output_dir, f"best_espcn_{args.scale}x.pt"),
            )
            print(f"  ★ New best model! PSNR: {best_psnr:.2f} dB")

        # Save periodic checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(
                checkpoint_data,
                os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt"),
            )

    # ── Final summary ────────────────────────────────────────────────────
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete. Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Best model: {args.output_dir}/best_espcn_{args.scale}x.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

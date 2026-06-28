"""
FlowNet-Lite Training Script — Optical Flow Frame Interpolation
================================================================

Trains the FlowNet-Lite model for frame interpolation using synthetic
motion triplets from the DIV2K dataset.

Usage:
    python train_flownet.py --hr-dir ../trainData/train \
                            --val-dir ../trainData/val \
                            --epochs 300 \
                            --batch-size 8 \
                            --lr 5e-4

Training details:
  - Input: concat(frame_prev, frame_next) → [B, 6, H, W]
  - Target: frame_mid (ground truth middle frame)
  - Loss: Charbonnier(warped, target)
        + λ_perceptual · VGG(warped, target)
        + λ_smooth · flow_smoothness(flow)
  - Mixed precision (FP16) for training speed
  - AdamW optimizer with cosine annealing
  - Gradient clipping at max_norm=1.0
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from models.flownet import FlowNetLite
from losses.charbonnier import CharbonnierLoss
from losses.perceptual import PerceptualLoss
from datasets.flow_dataset import FlowDataset


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute average PSNR in dB across the batch."""
    mse = (pred - target).pow(2).mean(dim=[1, 2, 3])
    psnr = -10.0 * torch.log10(mse + 1e-8)
    return psnr.mean().item()


def train_one_epoch(
    model: FlowNetLite,
    loader: DataLoader,
    criterion_pixel: CharbonnierLoss,
    criterion_perceptual: PerceptualLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    lambda_perceptual: float = 0.01,
    lambda_smooth: float = 0.1,
) -> tuple[float, float, float]:
    """
    Train for one epoch.

    Returns
    -------
    tuple[float, float, float]
        (avg_loss, avg_psnr, avg_flow_magnitude)
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_flow_mag = 0.0
    num_batches = 0

    for batch_idx, (frame_prev, frame_mid, frame_next) in enumerate(loader):
        frame_prev = frame_prev.to(device, non_blocking=True)
        frame_mid = frame_mid.to(device, non_blocking=True)
        frame_next = frame_next.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            # Forward pass: estimate flow and warp
            warped, flow = model(frame_prev, frame_next, timestep=0.5)

            # ── Losses ───────────────────────────────────────────────────
            # 1. Pixel reconstruction loss
            loss_pixel = criterion_pixel(warped, frame_mid)

            # 2. Perceptual loss
            loss_perceptual = criterion_perceptual(
                warped.clamp(0, 1).float(),
                frame_mid.float(),
            )

            # 3. Flow smoothness regularization
            loss_smooth = FlowNetLite.flow_smoothness_loss(flow)

            # Combined loss
            loss = (
                loss_pixel
                + lambda_perceptual * loss_perceptual
                + lambda_smooth * loss_smooth
            )

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            batch_psnr = compute_psnr(warped.clamp(0, 1), frame_mid)
            flow_mag = flow.pow(2).sum(dim=1).sqrt().mean().item()

        total_loss += loss.item()
        total_psnr += batch_psnr
        total_flow_mag += flow_mag
        num_batches += 1

        if batch_idx % 50 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}]  "
                f"loss={loss.item():.4f}  "
                f"pixel={loss_pixel.item():.4f}  "
                f"percep={loss_perceptual.item():.4f}  "
                f"smooth={loss_smooth.item():.4f}  "
                f"PSNR={batch_psnr:.2f}  "
                f"flow_mag={flow_mag:.2f}px"
            )

    n = max(num_batches, 1)
    return total_loss / n, total_psnr / n, total_flow_mag / n


@torch.no_grad()
def validate(
    model: FlowNetLite,
    loader: DataLoader,
    criterion_pixel: CharbonnierLoss,
    device: torch.device,
) -> tuple[float, float]:
    """
    Validate on the validation set.

    Returns
    -------
    tuple[float, float]
        (avg_loss, avg_psnr)
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    for frame_prev, frame_mid, frame_next in loader:
        frame_prev = frame_prev.to(device, non_blocking=True)
        frame_mid = frame_mid.to(device, non_blocking=True)
        frame_next = frame_next.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            warped, flow = model(frame_prev, frame_next, timestep=0.5)
            loss = criterion_pixel(warped, frame_mid)

        total_loss += loss.item()
        total_psnr += compute_psnr(warped.clamp(0, 1), frame_mid)
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_psnr / n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FlowNet-Lite for frame interpolation"
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
        "--patch-size", type=int, default=256,
        help="Training patch size (default: 256)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Training batch size (default: 8, lower than ESPCN due to U-Net memory)"
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Number of training epochs (default: 300)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Initial learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--lambda-perceptual", type=float, default=0.01,
        help="Weight for perceptual loss (default: 0.01)"
    )
    parser.add_argument(
        "--lambda-smooth", type=float, default=0.1,
        help="Weight for flow smoothness loss (default: 0.1)"
    )
    parser.add_argument(
        "--max-translate", type=float, default=16.0,
        help="Max synthetic translation in pixels (default: 16.0)"
    )
    parser.add_argument(
        "--max-rotate", type=float, default=5.0,
        help="Max synthetic rotation in degrees (default: 5.0)"
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
        "--output-dir", type=str, default="checkpoints/flownet",
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
    model = FlowNetLite().to(device)
    print(f"FlowNet-Lite | Parameters: {model.count_parameters():,}")

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
    train_dataset = FlowDataset(
        hr_dir=args.hr_dir,
        patch_size=args.patch_size,
        max_translate=args.max_translate,
        max_rotate_deg=args.max_rotate,
        augment=True,
        repeat=args.repeat,
    )
    val_dataset = FlowDataset(
        hr_dir=args.val_dir,
        patch_size=args.patch_size,
        max_translate=args.max_translate,
        max_rotate_deg=args.max_rotate,
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

    # ── Resume ───────────────────────────────────────────────────────────
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
    print(f"Training FlowNet-Lite for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_loss, train_psnr, flow_mag = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion_pixel=criterion_pixel,
            criterion_perceptual=criterion_perceptual,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            lambda_perceptual=args.lambda_perceptual,
            lambda_smooth=args.lambda_smooth,
        )

        # Validate
        val_loss, val_psnr = validate(
            model=model,
            loader=val_loader,
            criterion_pixel=criterion_pixel,
            device=device,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # ── Logging ──────────────────────────────────────────────────────
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_PSNR={train_psnr:.2f}  "
            f"val_loss={val_loss:.4f}  val_PSNR={val_psnr:.2f}  "
            f"flow={flow_mag:.2f}px  lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("PSNR/train", train_psnr, epoch)
        writer.add_scalar("PSNR/val", val_psnr, epoch)
        writer.add_scalar("Flow/magnitude", flow_mag, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # ── Sample visualization ─────────────────────────────────────────
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_prev, sample_mid, sample_next = next(iter(val_loader))
                sample_prev = sample_prev[:4].to(device)
                sample_mid = sample_mid[:4].to(device)
                sample_next = sample_next[:4].to(device)

                with autocast(device_type="cuda", dtype=torch.float16):
                    sample_warped, sample_flow = model(
                        sample_prev, sample_next, timestep=0.5
                    )
                sample_warped = sample_warped.clamp(0, 1)

                # Visualize flow as RGB (normalize flow magnitude to [0, 1])
                flow_mag_vis = sample_flow.pow(2).sum(dim=1, keepdim=True).sqrt()
                flow_max = flow_mag_vis.max() + 1e-8
                flow_rgb = torch.cat([
                    (sample_flow[:, 0:1, :, :] / flow_max + 1) / 2,  # dx → red
                    (sample_flow[:, 1:2, :, :] / flow_max + 1) / 2,  # dy → green
                    flow_mag_vis / flow_max,                          # mag → blue
                ], dim=1)

                # Grid: [prev, warped, ground_truth, next, flow_vis]
                grid = vutils.make_grid(
                    torch.cat([
                        sample_prev,
                        sample_warped.float(),
                        sample_mid,
                        sample_next,
                        flow_rgb.float(),
                    ], dim=0),
                    nrow=4, normalize=False,
                )
                writer.add_image("Samples/prev_warped_gt_next_flow", grid, epoch)

        # ── Checkpointing ────────────────────────────────────────────────
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_psnr": train_psnr,
            "val_psnr": val_psnr,
            "best_psnr": best_psnr,
        }

        torch.save(
            checkpoint_data,
            os.path.join(args.output_dir, "latest.pt"),
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint_data["best_psnr"] = best_psnr
            torch.save(
                checkpoint_data,
                os.path.join(args.output_dir, "best_flownet.pt"),
            )
            print(f"  ★ New best model! PSNR: {best_psnr:.2f} dB")

        if (epoch + 1) % 50 == 0:
            torch.save(
                checkpoint_data,
                os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt"),
            )

    # ── Final summary ────────────────────────────────────────────────────
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete. Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Best model: {args.output_dir}/best_flownet.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

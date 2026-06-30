from __future__ import annotations

from typing import Sequence, Tuple

import torch


def assert_nchw_rgb_fp16(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name}: expected NCHW 4D tensor, got ndim={tensor.ndim}")
    if tensor.shape[1] != 3:
        raise ValueError(f"{name}: expected 3 RGB channels, got C={tensor.shape[1]}")
    if tensor.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"{name}: expected fp16/fp32, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name}: expected CUDA tensor")


def assert_bgra_uint8_hwc(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name}: expected HWC, got ndim={tensor.ndim}")
    if tensor.shape[2] != 4:
        raise ValueError(f"{name}: expected 4 BGRA channels, got C={tensor.shape[2]}")
    if tensor.dtype != torch.uint8:
        raise ValueError(f"{name}: expected uint8, got {tensor.dtype}")


def compute_padded_shape(h: int, w: int, alignment: int = 32) -> Tuple[int, int]:
    ph = ((h + alignment - 1) // alignment) * alignment
    pw = ((w + alignment - 1) // alignment) * alignment
    return ph, pw


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())

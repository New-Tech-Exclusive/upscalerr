from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


class CudaFrameView:
    """
    Zero-copy adoption of a DXCam grab buffer via __cuda_array_interface__.

    The returned tensor is a view into DXGI/CUDA capture memory and is only valid
    until the next grab() recycles the underlying pool slot.
    """

    def __init__(self, grab_result: Any, device: torch.device) -> None:
        self._source = grab_result
        cai = getattr(grab_result, "__cuda_array_interface__", None)
        if cai is None:
            raise RuntimeError(
                "Grab result does not expose __cuda_array_interface__. "
                "Ensure dxcam CUDA interop is active and output_color='BGRA'."
            )
        self._cai = cai
        self.tensor = torch.as_tensor(cai, device=device)
        if self.tensor.dtype != torch.uint8:
            raise RuntimeError(f"Expected uint8 capture tensor, got {self.tensor.dtype}")
        if self.tensor.ndim != 3 or self.tensor.shape[2] != 4:
            raise RuntimeError(f"Expected BGRA HWC, got shape {tuple(self.tensor.shape)}")

    @property
    def height(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def width(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def shape(self) -> Tuple[int, int, int]:
        h, w, c = self.tensor.shape
        return int(h), int(w), int(c)

    @property
    def data_ptr(self) -> int:
        return int(self.tensor.data_ptr())

    def validate_lifetime(self, grab_result: Any) -> None:
        if grab_result is not self._source:
            raise RuntimeError("Capture buffer was recycled before consumption")

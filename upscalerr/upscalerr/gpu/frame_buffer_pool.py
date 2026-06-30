from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FrameBufferPool:
    """Preallocated VRAM tensors for zero-allocation hot loop."""

    device: torch.device
    height: int
    width: int
    scale: int
    enable_framegen: bool

    def __post_init__(self) -> None:
        h, w = self.height, self.width
        oh, ow = h * self.scale, w * self.scale

        self.rgb_curr = torch.empty((1, 3, h, w), dtype=torch.float16, device=self.device)
        self.rgb_prev = torch.empty((1, 3, h, w), dtype=torch.float16, device=self.device)
        self.espcn_out = torch.empty((1, 3, oh, ow), dtype=torch.float16, device=self.device)
        self.espcn_out_prev = torch.empty((1, 3, oh, ow), dtype=torch.float16, device=self.device)
        self.display_a = torch.empty((oh, ow, 4), dtype=torch.uint8, device=self.device)
        self.display_b = torch.empty((oh, ow, 4), dtype=torch.uint8, device=self.device)

        if self.enable_framegen:
            self.flow_in = torch.empty((1, 6, h, w), dtype=torch.float16, device=self.device)
            self.flow_out = torch.empty((1, 2, h, w), dtype=torch.float16, device=self.device)
            self.flow_out_hr = torch.empty((1, 2, oh, ow), dtype=torch.float16, device=self.device)
            self.warp_mid = torch.empty((1, 3, oh, ow), dtype=torch.float16, device=self.device)
        else:
            self.flow_in = None
            self.flow_out = None
            self.flow_out_hr = None
            self.warp_mid = None

    @property
    def output_height(self) -> int:
        return self.height * self.scale

    @property
    def output_width(self) -> int:
        return self.width * self.scale

    def resize(self, height: int, width: int) -> None:
        if height == self.height and width == self.width:
            return
        self.height = height
        self.width = width
        self.__post_init__()

    def display_buffer(self, index: int) -> torch.Tensor:
        return self.display_a if (index & 1) == 0 else self.display_b

    def swap_temporal(self) -> None:
        self.rgb_prev, self.rgb_curr = self.rgb_curr, self.rgb_prev
        self.espcn_out_prev, self.espcn_out = self.espcn_out, self.espcn_out_prev

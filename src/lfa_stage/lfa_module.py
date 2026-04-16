from __future__ import annotations

import torch
from torch import nn


class LFAConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: tuple[int, int], padding: tuple[int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalFeatureAugmentation(nn.Module):
    """
    Correct LFA (parallel + residual × 2)

    Input:
        x ∈ R^{B × C × H × W}

    Pipeline:
    1. Split feature map into 4 regions
    2. For each region:
        - Apply parallel convs (1×3, 3×3, 3×1)
        - Residual add
        - Repeat again (second stage)
    3. Reconstruct full feature map
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()

        # shared conv blocks
        self.conv_1x3 = LFAConvBlock(channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv_3x3 = LFAConvBlock(channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv_3x1 = LFAConvBlock(channels, kernel_size=(3, 1), padding=(1, 0))

    def _split_regions(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, height, width = x.shape
        half_h = height // 2
        half_w = width // 2

        top_left = x[:, :, :half_h, :half_w]
        top_right = x[:, :, :half_h, half_w:]
        bottom_left = x[:, :, half_h:, :half_w]
        bottom_right = x[:, :, half_h:, half_w:]

        return top_left, top_right, bottom_left, bottom_right

    def _parallel_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply parallel convolutions and sum them
        """
        out1 = self.conv_1x3(x)
        out2 = self.conv_3x3(x)
        out3 = self.conv_3x1(x)

        return out1 + out2 + out3

    def _augment_region(self, region: torch.Tensor) -> torch.Tensor:
        """
        Two-stage LFA block:
        Stage 1: parallel + residual
        Stage 2: parallel + residual
        """

        # -------- Stage 1 --------
        parallel_out = self._parallel_conv(region)
        x1 = region + parallel_out  # residual

        # -------- Stage 2 --------
        parallel_out_2 = self._parallel_conv(x1)
        x2 = x1 + parallel_out_2  # residual

        return torch.relu(x2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # split
        top_left, top_right, bottom_left, bottom_right = self._split_regions(x)

        # process each region
        top = torch.cat(
            [
                self._augment_region(top_left),
                self._augment_region(top_right),
            ],
            dim=3,
        )

        bottom = torch.cat(
            [
                self._augment_region(bottom_left),
                self._augment_region(bottom_right),
            ],
            dim=3,
        )

        # reconstruct full feature map
        out = torch.cat([top, bottom], dim=2)
        return torch.relu(out)
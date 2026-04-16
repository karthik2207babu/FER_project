from __future__ import annotations

import torch
from torch import nn


class MSGCBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleGlobalConvolution(nn.Module):
    """
    Multi-Scale Global Convolution (MSGC)

    Input:
        x in R^{B x 128 x 28 x 28}

    Pipeline:
    1. Split channels into 4 equal groups.
    2. Process the groups with 1x1, 3x3, 5x5, and 7x7 convolutions.
    3. Concatenate all groups back together.
    4. Fuse with a 1x1 convolution and add a residual connection.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("MSGC requires the number of channels to be divisible by 4.")

        group_channels = channels // 4
        self.branch_1x1 = MSGCBranch(group_channels, kernel_size=1)
        self.branch_3x3 = MSGCBranch(group_channels, kernel_size=3)
        self.branch_5x5 = MSGCBranch(group_channels, kernel_size=5)
        self.branch_7x7 = MSGCBranch(group_channels, kernel_size=7)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_1, group_2, group_3, group_4 = torch.chunk(x, chunks=4, dim=1)

        multi_scale = torch.cat(
            [
                self.branch_1x1(group_1),
                self.branch_3x3(group_2),
                self.branch_5x5(group_3),
                self.branch_7x7(group_4),
            ],
            dim=1,
        )
        fused = self.fusion(multi_scale)
        return self.activation(fused + x)

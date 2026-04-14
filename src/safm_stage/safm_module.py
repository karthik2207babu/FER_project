from __future__ import annotations

import torch
from torch import nn


class SpatialAttentionFeatureModule(nn.Module):
    """
    SAFM (Spatial Attention Feature Module)

    Input:
        x ∈ R^{B × C × H × W}

    Output:
        same shape (B × C × H × W)

    Pipeline:
        1. Channel Avg Pool → (B,1,H,W)
        2. Channel Max Pool → (B,1,H,W)
        3. Concatenate → (B,2,H,W)
        4. Conv 7×7 → (B,1,H,W)
        5. Sigmoid → attention mask
        6. Multiply with input
    """

    def __init__(self) -> None:
        super().__init__()

        # 7×7 convolution to learn spatial attention
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """

        # -------- Step 1: Avg pooling across channel --------
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)

        # -------- Step 2: Max pooling across channel --------
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)

        # -------- Step 3: Concatenate --------
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B,2,H,W)

        # -------- Step 4: Conv 7×7 --------
        attention = self.conv(pooled)  # (B,1,H,W)

        # -------- Step 5: Sigmoid --------
        attention = self.sigmoid(attention)  # (B,1,H,W)

        # -------- Step 6: Apply attention --------
        out = x * attention  # broadcasting over channel

        return out
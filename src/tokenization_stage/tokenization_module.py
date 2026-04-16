from __future__ import annotations

import torch
from torch import nn

class RegionTokenization(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # pool
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # global token
        global_token = self.gap(x).flatten(1)
        
        # split feature map
        _, _, h, w = x.shape
        half_h = h // 2
        half_w = w // 2
        
        top_left = x[:, :, :half_h, :half_w]
        top_right = x[:, :, :half_h, half_w:]
        bottom_left = x[:, :, half_h:, :half_w]
        bottom_right = x[:, :, half_h:, half_w:]
        
        # regional tokens
        x1 = self.gap(top_left).flatten(1)
        x2 = self.gap(top_right).flatten(1)
        x3 = self.gap(bottom_left).flatten(1)
        x4 = self.gap(bottom_right).flatten(1)
        
        # stack tokens
        token_set = torch.stack([global_token, x1, x2, x3, x4], dim=1)
        
        # ensure memory layout
        token_set = token_set.contiguous()
        
        return token_set
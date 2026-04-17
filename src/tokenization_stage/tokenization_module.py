import torch
from torch import nn

class RegionTokenization(nn.Module):
    def __init__(self):
        super().__init__()
        # AdaptiveAvgPool2d(1) perfectly handles the Global Average Pooling math
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Input: F_ATT in (Batch, 128, 28, 28)
        Output: T in (Batch, 5, 128)
        """
        B, C, H, W = x.shape
        h_mid, w_mid = H // 2, W // 2
        
        # 1. Global Token: Pooling the entire 28x28 map
        # .view(B, C) flattens it from (Batch, 128, 1, 1) to (Batch, 128)
        x_g = self.gap(x).view(B, C)
        
        # 2. Split Spatial Map into 4 quadrants (14x14 each)
        r1 = x[:, :, :h_mid, :w_mid]
        r2 = x[:, :, :h_mid, w_mid:]
        r3 = x[:, :, h_mid:, :w_mid]
        r4 = x[:, :, h_mid:, w_mid:]
        
        # 3. Local Tokens: Global Average Pooling on each quadrant
        x_1 = self.gap(r1).view(B, C)
        x_2 = self.gap(r2).view(B, C)
        x_3 = self.gap(r3).view(B, C)
        x_4 = self.gap(r4).view(B, C)
        
        # 4. Token Set Creation
        # Stack the 5 tokens along dimension 1 to create the sequence
        tokens = torch.stack([x_g, x_1, x_2, x_3, x_4], dim=1)
        
        return tokens
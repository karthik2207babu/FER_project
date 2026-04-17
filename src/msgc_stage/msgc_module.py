import torch
from torch import nn

class MultiScaleGlobalConvolution(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        
        # We split the 128 channels into 4 groups of 32
        self.split_channels = channels // 4
        
        # Branch 1: 1x1 Convolution (Hyper-local details)
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 Convolution (Standard local details)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 5x5 Convolution (Regional context)
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 7x7 Convolution (Global face context)
        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Input: F_LFA in (Batch, 128, 28, 28)
        Output: F_MS in (Batch, 128, 28, 28)
        """
        # 1. Channel Split: Divide 128 channels into four 32-channel tensors
        # dim=1 is the channel dimension (Batch, Channels, Height, Width)
        c1, c2, c3, c4 = torch.split(x, self.split_channels, dim=1)
        
        # 2. Parallel Multi-Scale Convolutions
        y1 = self.branch_1x1(c1)
        y2 = self.branch_3x3(c2)
        y3 = self.branch_5x5(c3)
        y4 = self.branch_7x7(c4)
        
        # 3. Concatenation: Rebuild the 128 channels
        y = torch.cat([y1, y2, y3, y4], dim=1)
        
        # 4. Residual Fusion: Add the original input to the new multi-scale features
        f_ms = x + y
        
        return f_ms
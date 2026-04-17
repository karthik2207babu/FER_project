import torch
from torch import nn

class SequentialLFA(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        
        # 1. The Sequential Filter Chain
        # Applying these one after the other refines the features without exploding the channel count
        
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Input: x in (Batch, 128, 28, 28)
        Output: F_LFA in (Batch, 128, 28, 28)
        """
        B, C, H, W = x.shape
        
        # 1. Spatial Splitting: Divide the 28x28 map into four 14x14 regions
        h_mid, w_mid = H // 2, W // 2
        
        top_left     = x[:, :, :h_mid, :w_mid]
        top_right    = x[:, :, :h_mid, w_mid:]
        bottom_left  = x[:, :, h_mid:, :w_mid]
        bottom_right = x[:, :, h_mid:, w_mid:]
        
        regions = [top_left, top_right, bottom_left, bottom_right]
        processed_regions = []
        
        # 2. Sequential Convolutions on each 14x14 region
        for region in regions:
            # The filter chain (Sequential execution)
            feat1 = self.conv_1x3(region)
            feat2 = self.conv_3x3(feat1)
            feat3 = self.conv_3x1(feat2)
            
            # Residual Fusion
            out_region = self.final_relu(region + feat3)
            processed_regions.append(out_region)
            
        # 3. Spatial Reconstruction: Stitch the four 14x14 regions back into 28x28
        top_half = torch.cat([processed_regions[0], processed_regions[1]], dim=3)    # Concat along width
        bottom_half = torch.cat([processed_regions[2], processed_regions[3]], dim=3) # Concat along width
        f_lfa = torch.cat([top_half, bottom_half], dim=2)                            # Concat along height
        
        return f_lfa
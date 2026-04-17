import torch
from torch import nn

class SpatialAttentionFeatureModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # The convolution takes the concatenated Average and Max pool maps (2 channels)
        # and reduces them to a single spatial attention map (1 channel).
        # Kernel size 7x7 with padding 3 keeps the spatial dimensions strictly 28x28.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Input: F_MS in (Batch, 128, 28, 28)
        Output: F_ATT in (Batch, 128, 28, 28)
        """
        # 1. Channel Average Pooling
        # dim=1 is the channel dimension. keepdim=True preserves the 28x28 spatial grid.
        # Shape becomes: (Batch, 1, 28, 28)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # 2. Channel Max Pooling
        # torch.max returns a tuple of (values, indices). We only want the values [0].
        # Shape becomes: (Batch, 1, 28, 28)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        
        # 3. Concatenation
        # We concatenate along the channel dimension (dim=1) to create a 2-channel map.
        # Shape becomes: (Batch, 2, 28, 28)
        concat_pool = torch.cat([avg_pool, max_pool], dim=1)
        
        # 4. Convolution & Sigmoid Activation
        # Shape remains: (Batch, 1, 28, 28) with values strictly bounded between 0 and 1.
        attention_mask = self.sigmoid(self.conv(concat_pool))
        
        # 5. Feature Reweighting (Element-wise multiplication)
        # Broadcasting automatically multiplies the 1-channel mask across all 128 channels.
        f_att = x * attention_mask
        
        return f_att
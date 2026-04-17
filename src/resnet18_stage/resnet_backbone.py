import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class FERResNetBackbone(nn.Module):
    def __init__(self, freeze_weights: bool = False):
        """
        Extracts up to Layer 2 of ResNet18.
        freeze_weights: Set to True if you want to lock the ImageNet weights
                        and only train the custom modules.
        """
        super().__init__()
        
        # Load pre-trained ImageNet weights
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Extract exactly what is needed for F in R^(128 x 28 x 28)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        
        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        """
        Input: x in R^(3 x 224 x 224)
        Output: Feature map in R^(128 x 28 x 28)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x
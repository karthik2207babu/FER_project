import torch
from torch import nn

# Import your custom modules
from resnet18_stage.resnet_backbone import FERResNetBackbone
from lfa_stage.lfa_module import SequentialLFA
from msgc_stage.msgc_module import MultiScaleGlobalConvolution
from safm_stage.safm_module import SpatialAttentionFeatureModule
from tokenization_stage.tokenization_module import RegionTokenization
from frit_stage.frit_module import FRITTransformer
from classification_stage.classification_module import EmotionClassifier

class FERFullPipeline(nn.Module):
    def __init__(self, num_classes=7):
        super(FERFullPipeline, self).__init__()
        
        # 1. Feature Extraction (ResNet-18)
        self.backbone = FERResNetBackbone(freeze_weights=False)
        
        # 2. Local Feature Augmentation (Sequential)
        self.lfa = SequentialLFA(channels=128)
        
        # 3. Multi-Scale Global Convolution
        self.msgc = MultiScaleGlobalConvolution(channels=128)
        
        # 4. Spatial Attention Feature Module
        self.safm = SpatialAttentionFeatureModule()
        
        # 5. Region Tokenization
        self.tokenization = RegionTokenization()
        
        # 6. FRIT Transformer
        self.transformer = FRITTransformer(input_dim=128, embed_dim=64)
        
        # 7. Classification
        self.classifier = EmotionClassifier(embed_dim=64, num_classes=num_classes)

    def forward(self, x):
        """
        Input: I_a in R^(3 x 224 x 224)
        Output: y in R^(7)
        """
        # Feature Extraction -> F (128 x 28 x 28)
        f = self.backbone(x)
        
        # Local Augmentation -> F_LFA (128 x 28 x 28)
        f_lfa = self.lfa(f)
        
        # Multi-Scale context -> F_MS (128 x 28 x 28)
        f_ms = self.msgc(f_lfa)
        
        # Spatial Attention -> F_ATT (128 x 28 x 28)
        f_att = self.safm(f_ms)
        
        # Tokenization -> T (5 x 128)
        tokens = self.tokenization(f_att)
        
        # Transformer Interaction -> T' (5 x 64)
        t_prime = self.transformer(tokens)
        
        # Aggregation: Extract Global Token z = T'[0]
        # Shape: (Batch, 64)
        z = t_prime[:, 0, :]
        
        # Classification -> Logits y (7)
        logits = self.classifier(z)
        
        return logits
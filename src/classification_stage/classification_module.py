import torch
from torch import nn

class EmotionClassifier(nn.Module):
    def __init__(self, embed_dim: int = 64, num_classes: int = 7):
        """
        The final stage that translates the Transformer's global token 
        into emotion predictions.
        """
        super().__init__()
        
        # W_fc ∈ ℝ^(64 × 7)
        # This linear layer acts like a final weight matrix that determines 
        # which feature patterns correspond to which emotion.
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, z):
        """
        Input: z in (Batch, 64) - The global token from the transformer
        Output: y in (Batch, 7) - Raw logits for the 7 classes
        """
        # Calculate raw logits
        # y = z * W_fc
        y = self.fc(z)
        
        return y
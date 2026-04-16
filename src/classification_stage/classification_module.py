from __future__ import annotations

import torch
from torch import nn

class EmotionClassifier(nn.Module):
    def __init__(self, embed_dim: int = 64, num_classes: int = 7) -> None:
        super().__init__()
        # linear map
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, t_prime: torch.Tensor) -> torch.Tensor:
        # extract global (index 0)
        global_token = t_prime[:, 0, :].contiguous()
        
        # logits
        logits = self.fc(global_token)
        
        return logits
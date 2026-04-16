from __future__ import annotations

import math
import torch
from torch import nn

class FRITTransformer(nn.Module):
    def __init__(self, input_dim: int = 128, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        
        # linear projections
        self.wq = nn.Linear(input_dim, embed_dim, bias=False)
        self.wk = nn.Linear(input_dim, embed_dim, bias=False)
        self.wv = nn.Linear(input_dim, embed_dim, bias=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B, 5, 128)
        
        q = self.wq(t)  # (B, 5, 64)
        k = self.wk(t)  # (B, 5, 64)
        v = self.wv(t)  # (B, 5, 64)

        # QK^T
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.embed_dim)
        
        # attention matrix
        attention = torch.softmax(scores, dim=-1)  # (B, 5, 5)
        
        # T' = AV
        out = torch.bmm(attention, v)  # (B, 5, 64)
        out = out+t
        return out
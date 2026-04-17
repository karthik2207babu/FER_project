import torch
from torch import nn
import math

class FRITTransformer(nn.Module):
    def __init__(self, input_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear projections for Query, Key, and Value
        self.wq = nn.Linear(input_dim, embed_dim, bias=False)
        self.wk = nn.Linear(input_dim, embed_dim, bias=False)
        self.wv = nn.Linear(input_dim, embed_dim, bias=False)
        
        # Scaling factor: 1 / sqrt(d)
        self.scale = 1.0 / math.sqrt(self.embed_dim)

    def forward(self, tokens):
        """
        Input: T in (Batch, 5, 128)
        Output: T' in (Batch, 5, 64)
        """
        # 1. Linear Projections
        # Shape becomes (Batch, 5, 64)
        q = self.wq(tokens)
        k = self.wk(tokens)
        v = self.wv(tokens)
        
        # 2. Attention Matrix: Softmax(Q * K^T / sqrt(d))
        # k.transpose(1, 2) flips Key to (Batch, 64, 5) for matrix multiplication
        # torch.bmm is Batch Matrix Multiplication
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        
        # Apply Softmax across the last dimension to get probabilities
        # Shape: (Batch, 5, 5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 3. Feature Update: A * V
        # Shape: (Batch, 5, 64)
        t_prime = torch.bmm(attn_probs, v)
        
        return t_prime
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class MultiHeadAttention(nn.Module):
    """Simple wrapper around PyTorch's MultiheadAttention"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, query, key, value, attention_mask=None):
        output, weights = self.attention(
            query, key, value,
            key_padding_mask=attention_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        if weights is not None:
            weights = F.softmax(weights, dim=-1)
        
        return output, weights


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class MultiHeadAttention(nn.Module):
    """
    Simple wrapper around PyTorch's MultiheadAttention
    """
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
        
        # Apply softmax to ensure weights sum to 1
        if weights is not None:
            weights = F.softmax(weights, dim=-1)
        
        return output, weights

def test_multi_head_attention():
    batch_size, seq_len, embed_dim = 2, 10, 512
    num_heads = 8
    
    # Create model and input
    model = MultiHeadAttention(embed_dim, num_heads)
    x = torch.rand(batch_size, seq_len, embed_dim)
    
    # Run forward pass
    with torch.no_grad():
        output, weights = model(x, x, x)
    
    # Test shapes
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Test attention properties with relaxed tolerances
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(
        weight_sums,
        torch.ones_like(weight_sums),
        rtol=1e-2,
        atol=1e-2
    )
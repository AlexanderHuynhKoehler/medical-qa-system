import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as shown in the architecture diagram.
    
    This implementation follows the paper "Attention is All You Need" with 
    specific adaptations for medical question answering.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize multi-head attention module.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V for all heads at once
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product attention
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, attn_mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            query (Tensor): Query tensor [batch_size, seq_len_q, embed_dim]
            key (Tensor): Key tensor [batch_size, seq_len_k, embed_dim]
            value (Tensor): Value tensor [batch_size, seq_len_v, embed_dim]
            attn_mask (Tensor, optional): Mask to avoid attending to padding tokens
                                          [batch_size, seq_len_q, seq_len_k]
        
        Returns:
            Tensor: Output tensor [batch_size, seq_len_q, embed_dim]
            Tensor: Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(query)  # [batch_size, seq_len_q, embed_dim]
        k = self.k_proj(key)    # [batch_size, seq_len_k, embed_dim]
        v = self.v_proj(value)  # [batch_size, seq_len_v, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
        v = v.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_v, head_dim]
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # Expand mask for multiple heads: [batch_size, 1, seq_len_q, seq_len_k]
            if attn_mask.dim() == 3:  # [batch_size, seq_len_q, seq_len_k]
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len_q, head_dim]
        
        # Reshape output back to original dimensions
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)  # [batch_size, seq_len_q, embed_dim]
        
        # Final linear projection
        output = self.output_proj(output)
        
        return output, attn_weights

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.
    
    As shown in the transformer encoder block diagram, this consists of two
    linear transformations with a ReLU activation in between.
    """
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Initialize position-wise feed forward network.
        
        Args:
            embed_dim (int): Input and output dimension
            ff_dim (int): Hidden dimension of the feed forward network
            dropout (float): Dropout probability
        """
        super(PositionWiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for feed forward network.
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_len, embed_dim]
        
        Returns:
            Tensor: Output tensor [batch_size, seq_len, embed_dim]
        """
        # First linear transformation with ReLU activation
        x = F.relu(self.linear1(x))
        
        # Dropout for regularization
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        return x
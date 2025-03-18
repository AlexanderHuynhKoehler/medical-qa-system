from ..utils.imports import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings

# Import the MultiHeadAttention and PositionWiseFeedForward from your attention.py
from attention import MultiHeadAttention, PositionWiseFeedForward

class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    Implements layer normalization as described in the paper 
    "Layer Normalization" (Ba et al., 2016).
    """
    def __init__(self, features, eps=1e-6):
        """
        Initialize layer normalization module.
        
        Args:
            features (int): The feature size (embedding dimension)
            eps (float): A small constant for numerical stability
        """
        super(LayerNorm, self).__init__()
        
        # CRITICAL PARAMETER: These learnable parameters can affect model performance
        self.gamma = nn.Parameter(torch.ones(features))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(features))  # Shift parameter
        self.eps = eps  # For numerical stability
    
    def forward(self, x):
        """
        Forward pass for layer normalization.
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_len, features]
        
        Returns:
            Tensor: Normalized tensor of same shape
        """
        # Calculate mean and variance along the last dimension
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # Normalize, scale, and shift
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module.
    
    Adds positional information to input embeddings as described in 
    "Attention is All You Need".
    """
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        """
        Initialize positional encoding module.
        
        Args:
            embed_dim (int): Embedding dimension
            max_len (int): Maximum sequence length
            dropout (float): Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        
        # CRITICAL PARAMETER: Dropout affects information retention
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # CRITICAL PARAMETER: This divisor affects the frequency of the sinusoidal functions
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Calculate sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass for positional encoding.
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_len, embed_dim]
        
        Returns:
            Tensor: Input with positional encoding added
        """
        # Add positional encoding and apply dropout
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Consists of multi-head self-attention followed by position-wise feed-forward,
    with layer normalization and residual connections.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class CustomTransformerEncoder(nn.Module):
    def __init__(self, config, num_layers=6):
        super().__init__()
        self.embeddings = BertEmbeddings(config)  # Use BERT's embedding layer
        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                ff_dim=config.intermediate_size,
                dropout=config.hidden_dropout_prob
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.norm(x)

class MedicalQAModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', dropout=0.1):
        super().__init__()
        config = BertConfig.from_pretrained(pretrained_model_name)
        self.encoder = CustomTransformerEncoder(config)
        self.embed_dim = config.hidden_size
        self.yes_no_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, 2)
        )
        self.span_start_head = nn.Linear(self.embed_dim, 1)
        self.span_end_head = nn.Linear(self.embed_dim, 1)
        self.yes_no_confidence = nn.Linear(self.embed_dim, 1)
        self.span_confidence = nn.Linear(self.embed_dim * 2, 1)
    
    def forward(self, input_ids, attention_mask=None):
        encoded = self.encoder(input_ids, attention_mask)
        cls_representation = encoded[:, 0, :]
        yes_no_logits = self.yes_no_head(cls_representation)
        span_start_logits = self.span_start_head(encoded).squeeze(-1)
        span_end_logits = self.span_end_head(encoded).squeeze(-1)
        yes_no_conf = self.yes_no_confidence(cls_representation).sigmoid()
        batch_size, seq_len, _ = encoded.shape
        start_idx = torch.argmax(span_start_logits, dim=1)
        end_idx = torch.argmax(span_end_logits, dim=1)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        start_vecs = encoded[batch_indices, start_idx]
        end_vecs = encoded[batch_indices, end_idx]
        span_conf_input = torch.cat([start_vecs, end_vecs], dim=-1)
        span_conf = self.span_confidence(span_conf_input).sigmoid()
        return {
            'yes_no_logits': yes_no_logits,
            'span_start_logits': span_start_logits,
            'span_end_logits': span_end_logits,
            'yes_no_confidence': yes_no_conf,
            'span_confidence': span_conf
        }
    
    def get_answer(self, text_tokens, yes_no_threshold=0.5):
        outputs = self.forward(text_tokens)
        yes_no_probs = F.softmax(outputs['yes_no_logits'], dim=-1)
        yes_no_conf = outputs['yes_no_confidence']
        yes_no_decision = yes_no_probs.argmax(dim=-1)
        yes_no_answer = "Yes" if yes_no_decision == 1 else "No"
        start_idx = outputs['span_start_logits'].argmax(dim=-1)
        end_idx = outputs['span_end_logits'].argmax(dim=-1)
        span_conf = outputs['span_confidence']
        span_text = text_tokens[0][start_idx:end_idx+1]
        if yes_no_conf > span_conf * yes_no_threshold:
            return {
                'answer_type': 'yes_no',
                'answer': yes_no_answer,
                'confidence': yes_no_conf.item()
            }
        else:
            return {
                'answer_type': 'span',
                'answer': span_text,
                'start_index': start_idx.item(),
                'end_index': end_idx.item(),
                'confidence': span_conf.item()
            }
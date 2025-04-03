from src.utils import *
import math

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
    """Simplified Transformer Encoder Layer using built-in implementations"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        # Replace custom FFN with PyTorch's implementation
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, attention_mask=None):
        # Update parameter name to match attention.py
        attn_output, _ = self.attention(x, x, x, attention_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class CustomTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.layers = nn.ModuleList([encoder_layer for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        # Use BERT's embedding layer
        embeddings = self.bert.embeddings(input_ids)
        x = embeddings
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.norm(x)

class MedicalQAModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.1):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        config.model_name = model_name
        
        self.encoder = CustomTransformerEncoder(config)
        self.embed_dim = config.hidden_size
        
        # Simplified classification heads
        self.yes_no_head = nn.Linear(self.embed_dim, 2)
        self.span_head = nn.Linear(self.embed_dim, 2)  # Start/end positions
        self.confidence_head = nn.Linear(self.embed_dim, 2)  # Yes-no/span confidence
    
def forward(self, input_ids, attention_mask=None, start_positions=None, 
            end_positions=None, yes_no_labels=None):
    """
    Forward pass with optional loss calculation when labels are provided.
    
    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask for padding
        start_positions: Optional ground truth start positions for loss calculation
        end_positions: Optional ground truth end positions for loss calculation
        yes_no_labels: Optional ground truth yes/no labels (1=yes, 0=no, -1=not yes/no question)
    
    Returns:
        Dictionary containing model outputs and loss (if labels provided)
    """
    # Get encoder representations
    encoded = self.encoder(input_ids, attention_mask)
    cls_representation = encoded[:, 0, :]
    
    # Get yes/no logits
    yes_no_logits = self.yes_no_head(cls_representation)
    
    # Get span prediction logits
    span_logits = self.span_head(encoded)
    span_start_logits = span_logits[:, :, 0]
    span_end_logits = span_logits[:, :, 1]
    
    # Calculate confidence scores
    yes_no_confidence = self.confidence_head(cls_representation).sigmoid()
    
    # Get predicted span indices
    start_idx = torch.argmax(span_start_logits, dim=1)
    end_idx = torch.argmax(span_end_logits, dim=1)
    
    # Extract span representations for confidence calculation
    batch_size = encoded.size(0)
    start_vecs = torch.gather(encoded, 1, 
                              start_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, encoded.size(-1))).squeeze(1)
    end_vecs = torch.gather(encoded, 1, 
                            end_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, encoded.size(-1))).squeeze(1)
    span_conf_input = torch.cat([start_vecs, end_vecs], dim=-1)
    span_confidence = self.confidence_head(span_conf_input).sigmoid()
    
    # Prepare outputs dictionary
    outputs = {
        'yes_no_logits': yes_no_logits,
        'span_start_logits': span_start_logits,
        'span_end_logits': span_end_logits,
        'yes_no_confidence': yes_no_confidence,
        'span_confidence': span_confidence
    }
    
    # Calculate loss if labels are provided (training mode)
    if start_positions is not None and end_positions is not None and yes_no_labels is not None:
        # Loss for yes/no classification
        yes_no_loss = 0
        # Only calculate yes/no loss for actual yes/no questions
        yes_no_mask = (yes_no_labels != -1)
        if yes_no_mask.sum() > 0:
            # Filter yes/no questions
            filtered_yes_no_logits = yes_no_logits[yes_no_mask]
            filtered_yes_no_labels = yes_no_labels[yes_no_mask]
            yes_no_loss = F.cross_entropy(filtered_yes_no_logits, filtered_yes_no_labels)
        
        # Loss for span extraction
        # We use regular cross-entropy with ignore_index=-1
        start_loss = F.cross_entropy(span_start_logits, start_positions, ignore_index=-1)
        end_loss = F.cross_entropy(span_end_logits, end_positions, ignore_index=-1)
        span_loss = (start_loss + end_loss) / 2
        
        # Combine losses
        # You can adjust these weights if needed
        total_loss = yes_no_loss + span_loss
        
        # Add loss to outputs
        outputs['loss'] = total_loss
    
    return outputs
    
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
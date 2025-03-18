from transformers import AutoModel, AutoTokenizer
import torch

class BioBERTEmbeddings:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def encode(self, texts: list) -> torch.Tensor:
        """Encode texts to embeddings."""
        pass

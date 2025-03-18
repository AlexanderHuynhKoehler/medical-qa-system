import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from typing import Optional, Tuple

class QAHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 2 for start/end position
        
    def forward(self, sequence_output):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits

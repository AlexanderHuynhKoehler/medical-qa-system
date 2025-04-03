from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Union, Optional
from src.utils import *


class MedicalQADataset(Dataset):
    def __init__(self, questions: List[str], answers: List[str], tokenizer):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx) -> Dict:
        question = self.questions[idx]
        answer = self.answers[idx]
        
        encoding = self.tokenizer(
            question,
            answer,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

from src.utils import *
import torch


class MedicalQAPipeline:
    """
    High-level interface for medical question answering
    """
    def __init__(self, model_name='bert-base-uncased', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = MedicalQAModel(model_name=model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def answer_question(self, question: str, context: str, yes_no_threshold: float = 0.5):
        """
        Process a question and context to generate an answer
        
        Args:
            question (str): The question to answer
            context (str): The context to find the answer in
            yes_no_threshold (float): Threshold for yes/no vs span decision
        """
        # Prepare input
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            answer = self.model.get_answer(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                yes_no_threshold=yes_no_threshold
            )
        
        # Process the answer
        if answer['answer_type'] == 'span':
            # Convert token indices to text
            tokens = self.tokenizer.convert_ids_to_tokens(
                inputs['input_ids'][0][answer['start_index']:answer['end_index']+1]
            )
            answer['answer'] = self.tokenizer.convert_tokens_to_string(tokens)
        
        return answer
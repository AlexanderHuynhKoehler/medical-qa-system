import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from datetime import datetime
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset

from src.utils.metrics import compute_metrics
from src.data.preprocessing import preprocess_squad
from src.models.qa_model import model_init

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"finetune_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run fine-tuning"""
    # Configuration
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    max_length = 384  # Standard for QA tasks
    doc_stride = 128  # Overlap for handling long contexts
    subset_size = 1000  # Use a small subset for proof of concept
    output_dir = "checkpoints/squad_finetune"
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load SQuAD dataset
    logger.info("Loading SQuAD dataset")
    dataset = load_dataset("squad")
    
    # Take subset for proof of concept
    if subset_size:
        train_dataset = dataset["train"].select(range(subset_size))
        eval_dataset = dataset["validation"].select(range(subset_size // 5))
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    
    logger.info(f"Training with {len(train_dataset)} examples, evaluating with {len(eval_dataset)} examples")
    
    # Define preprocessing function (using the standard approach)
    def prepare_features(examples):
        # Tokenize questions and contexts
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        # Since one example might give us several features if it has a long context
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # Map offsets to original text position
        offset_mapping = tokenized_examples["offset_mapping"]
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["yes_no_labels"] = []  # For yes/no questions
        
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = 0
            
            # Get sample index
            sample_index = sample_mapping[i]
            
            # Get the answers
            answers = examples["answers"][sample_index]
            
            # If no answers, set to CLS index
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["yes_no_labels"].append(-1)  # Not a yes/no question
            else:
                # Get first answer start and text
                answer_start = answers["answer_start"][0]
                answer_text = answers["text"][0]
                
                # Check if it's a yes/no question
                is_yes_no = answer_text.lower() in ["yes", "no"]
                
                if is_yes_no:
                    # For yes/no questions, set positions to CLS and set yes/no label
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["yes_no_labels"].append(1 if answer_text.lower() == "yes" else 0)
                else:
                    # Regular answer span
                    answer_end = answer_start + len(answer_text)
                    
                    # Set to CLS at first (default if answer not in this span)
                    start_position = cls_index
                    end_position = cls_index
                    
                    # Check if the answer is in this span
                    for j, (offset_start, offset_end) in enumerate(offsets):
                        if offset_start <= answer_start < offset_end:
                            start_position = j
                        if offset_start < answer_end <= offset_end:
                            end_position = j
                            break
                    
                    tokenized_examples["start_positions"].append(start_position)
                    tokenized_examples["end_positions"].append(end_position)
                    tokenized_examples["yes_no_labels"].append(-1)  # Not a yes/no question
        
        return tokenized_examples
    
    # Process the datasets
    tokenized_train = train_dataset.map(
        prepare_features,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_eval = eval_dataset.map(
        prepare_features,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Initialize model
    logger.info("Initializing model")
    model = MedicalQAModel(pretrained_model_name=model_name)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )
if __name__ == "__main__":
    main()
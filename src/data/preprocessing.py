"""
SQuAD dataset preprocessing for QA model fine-tuning.

This script handles preprocessing of the SQuAD dataset for training
a question-answering model with both span extraction and yes/no capabilities.
"""

from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_squad_examples(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Preprocess SQuAD examples for QA training.
    
    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer for encoding inputs
        max_length: Maximum sequence length
        doc_stride: Stride for handling overlapping windows
        
    Returns:
        dict: Preprocessed features
    """
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
    
    # Since one example might give us several features
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]
    
    # Labels for our model
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["yes_no_labels"] = []
    
    for i, offsets in enumerate(offset_mapping):
        # CLS token index (typically 0)
        cls_index = 0
        
        # Get the corresponding example
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # If no answers, set CLS token as answer position
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["yes_no_labels"].append(-1)  # Not a yes/no question
            continue
        
        # Get the first answer's text and start position
        answer_text = answers["text"][0]
        answer_start = answers["answer_start"][0]
        answer_end = answer_start + len(answer_text)
        
        # Check if it's a yes/no question (rare in SQuAD but supported by our model)
        is_yes_no = answer_text.lower() in ["yes", "no"]
        
        if is_yes_no:
            # For yes/no questions, use CLS token position
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["yes_no_labels"].append(1 if answer_text.lower() == "yes" else 0)
        else:
            # Default to CLS token positions (for "impossible" answers)
            start_position = cls_index
            end_position = cls_index
            
            # Find token positions for the answer
            for j, (start, end) in enumerate(offsets):
                if start <= answer_start < end:
                    start_position = j
                
                if start < answer_end <= end:
                    end_position = j
                    break
            
            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)
            tokenized_examples["yes_no_labels"].append(-1)  # Not a yes/no question
    
    return tokenized_examples

def prepare_squad_dataset(tokenizer, subset_size=None, max_length=384, doc_stride=128):
    """
    Load and preprocess the SQuAD dataset.
    
    Args:
        tokenizer: Tokenizer for encoding inputs
        subset_size: Optional size limit for quick testing (default: use full dataset)
        max_length: Maximum sequence length
        doc_stride: Stride for handling overlapping windows
        
    Returns:
        tuple: (tokenized_train, tokenized_validation) datasets
    """
    logger.info("Loading SQuAD dataset")
    dataset = load_dataset("squad")
    
    # Take subset if specified
    if subset_size:
        train_dataset = dataset["train"].select(range(subset_size))
        eval_dataset = dataset["validation"].select(range(subset_size // 5))
        logger.info(f"Using subset: {len(train_dataset)} training examples, {len(eval_dataset)} validation examples")
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        logger.info(f"Using full dataset: {len(train_dataset)} training examples, {len(eval_dataset)} validation examples")
    
    # Preprocess the datasets
    logger.info("Preprocessing training dataset")
    tokenized_train = train_dataset.map(
        lambda examples: preprocess_squad_examples(
            examples, tokenizer, max_length, doc_stride
        ),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    logger.info("Preprocessing validation dataset")
    tokenized_eval = eval_dataset.map(
        lambda examples: preprocess_squad_examples(
            examples, tokenizer, max_length, doc_stride
        ),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    logger.info("Dataset preprocessing complete")
    return tokenized_train, tokenized_eval

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_data, eval_data = prepare_squad_dataset(tokenizer, subset_size=10)
    
    print(f"Train features: {len(train_data)}")
    print(f"Sample features: {list(train_data[0].keys())}")
from transformers import AutoModelForQuestionAnswering

def model_init(model_name="dmis-lab/biobert-base-cased-v1.1"):
    """Initialize a pretrained BioBERT model for QA."""
    return AutoModelForQuestionAnswering.from_pretrained(model_name)
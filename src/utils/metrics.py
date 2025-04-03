from typing import Dict
from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    """Compute metrics for question answering evaluation."""
    predictions, labels = eval_pred
    start_logits, end_logits = predictions
    start_positions, end_positions = labels[:, 0], labels[:, 1]
    
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)
    
    exact_match = compute_exact_match(start_preds, end_preds, start_positions, end_positions)
    f1_score = compute_f1_score(start_preds, end_preds, start_positions, end_positions)
    
    return {
        "exact_match": exact_match,
        "f1": f1_score
    }

def compute_exact_match(start_preds, end_preds, start_positions, end_positions):
    """Compute exact match score"""
    correct = ((start_preds == start_positions) & (end_preds == end_positions)).mean()
    return float(correct)

def compute_f1_score(start_preds, end_preds, start_positions, end_positions):
    """Compute token-level F1 score"""
    f1_scores = []
    for start_pred, end_pred, start_pos, end_pos in zip(
        start_preds, end_preds, start_positions, end_positions
    ):
        pred_tokens = set(range(start_pred, end_pred + 1))
        true_tokens = set(range(start_pos, end_pos + 1))
        
        common_tokens = pred_tokens & true_tokens
        
        if len(pred_tokens) == 0 and len(true_tokens) == 0:
            f1_scores.append(1.0)
            continue
        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            f1_scores.append(0.0)
            continue
            
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    
    return float(np.mean(f1_scores))
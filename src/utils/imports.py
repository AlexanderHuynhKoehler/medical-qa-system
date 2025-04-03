import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from torch.nn import MultiheadAttention, TransformerEncoderLayer
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    BertConfig,
    BertModel,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EvalPrediction
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from datasets import load_dataset
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm
from .metrics import compute_exact_match, compute_f1_score, compute_metrics
from .base_model import MedicalQAModel
from src.retrieval.vector_db import VectorDB
from src.models.QAhead import QAHead
from src.models.base_model import (
    LayerNorm,
    PositionalEncoding,
    EncoderLayer,
    CustomTransformerEncoder,
    MedicalQAModel
)
from src.data.preprocessing import preprocess_newsqa
from src.data.dataset import MedicalQADataset

__all__ = [
    'os',
    'torch',
    'nn',
    'F',
    'MultiheadAttention',
    'TransformerEncoderLayer',
    'DataLoader',
    'TensorDataset',
    'AutoTokenizer',
    'AutoModelForQuestionAnswering',
    'BertConfig',
    'BertModel',
    'BertTokenizer',
    'TrainingArguments',
    'Trainer',
    'default_data_collator',
    'EvalPrediction',
    'QuestionAnsweringModelOutput',
    'load_dataset',
    'np',
    'pd',
    'logging',
    'datetime',
    'Dict',
    'List',
    'Tuple',
    'Optional',
    'Counter',
    'fetch_20newsgroups',
    'train_test_split',
    'TSNE',
    'plt',
    'sns',
    'math',
    'tqdm',
    'compute_exact_match',
    'compute_f1_score',
    'MultiHeadAttention',
    'MedicalQAModel',
    'VectorDB',
    'QAHead',
    'LayerNorm',
    'PositionalEncoding',
    'EncoderLayer',
    'CustomTransformerEncoder',
    'MedicalQAModel',
    'preprocess_newsqa',
    'MedicalQADataset'
]

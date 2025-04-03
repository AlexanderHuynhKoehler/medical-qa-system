# MedBot

A medical question-answering system using transformer-based architecture.

## Project Structure

The project is organized as follows:
- `config/`: Configuration files for model and training
- `data/`: Data storage and processing
- `src/`: Main source code
- `notebooks/`: Jupyter notebooks for exploration
- `scripts/`: Training and evaluation scripts
- `tests/`: Unit tests

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Training:
```bash
python scripts/train.py --config config/train_config.yaml
```

Evaluation:
```bash
python scripts/evaluate.py --model-path checkpoints/latest
```

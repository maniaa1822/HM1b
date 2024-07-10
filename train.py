from src.lstm_classification_model import LSTMClassifier, train_model,train_and_evaluate, evaluate_model
from src.lstm_dataset_class import LSTMTextClassificationDataset

import torch

train_dataset = LSTMTextClassificationDataset('HASPEEDE-20240708T122905Z-001/HASPEEDE/train-taskA.jsonl', split='train', split_ratio=0.8)
val_dataset = LSTMTextClassificationDataset('HASPEEDE-20240708T122905Z-001/HASPEEDE/train-taskA.jsonl', split='validation', split_ratio=0.8)
test_dataset1 = LSTMTextClassificationDataset('HASPEEDE-20240708T122905Z-001/HASPEEDE/test-news-taskA.jsonl')
test_dataset2 = LSTMTextClassificationDataset('HASPEEDE-20240708T122905Z-001/HASPEEDE/test-news-taskA.jsonl')

VOCAB_SIZE = train_dataset.get_vocab_size()
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(train_dataset.label_map)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

train_and_evaluate(
    model, 
    train_dataset, 
    val_dataset, 
    {'test1': test_dataset1, 'test2': test_dataset2},
    batch_size=128, 
    n_epochs=5, 
    lr=0.001, 
    device='cuda' if torch.cuda.is_available() else 'cpu',
    train_only=True,
    eval_only=False
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from lstm_dataset_class import LSTMTextClassificationDataset

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        return self.fc(self.dropout(hidden))

def train_model(model, train_iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(train_iterator, desc="Training"):
        optimizer.zero_grad()
        text = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += (predictions.argmax(1) == labels).float().mean().item()
    
    return epoch_loss / len(train_iterator), epoch_acc / len(train_iterator)

def evaluate_model(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            text = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += (predictions.argmax(1) == labels).float().mean().item()
            
            all_predictions.extend(predictions.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    return epoch_loss / len(iterator), epoch_acc / len(iterator), precision, recall, f1

def train_and_evaluate(model, train_dataset, val_dataset, test_datasets, batch_size=32, n_epochs=10, lr=0.001, device='cuda'):
    train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_iterator = DataLoader(val_dataset, batch_size=batch_size)
    test_iterators = {name: DataLoader(dataset, batch_size=batch_size) for name, dataset in test_datasets.items()}

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Add weight decay for L2 regularization
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        train_loss, train_acc = train_model(model, train_iterator, optimizer, criterion, device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_iterator, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')
        print(f'\t Val. Precision: {val_precision:.3f} | Val. Recall: {val_recall:.3f} | Val. F1: {val_f1:.3f}')

    model.load_state_dict(torch.load('best_model.pt'))

    for name, test_iterator in test_iterators.items():
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_iterator, criterion, device)
        print(f'Test Results ({name}):')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        print(f'\tTest Precision: {test_precision:.3f} | Test Recall: {test_recall:.3f} | Test F1: {test_f1:.3f}')

# Usage example:
# Assuming you have your datasets prepared
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
    n_epochs=12, 
    lr=0.001, 
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

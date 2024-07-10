#%%
import json
from torch.utils.data import Dataset
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import random

# Set a fixed seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
random.seed(seed_value)

class LSTMTextClassificationDataset(Dataset):
    def __init__(self, file_path, max_seq_length=100, min_freq=2, split_ratio=0.8, split='train'):
        self.data = []
        self.max_seq_length = max_seq_length
        
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Shuffle and split data
        random.shuffle(self.data)
        split_index = int(len(self.data) * split_ratio)
        if split == 'train':
            self.data = self.data[:split_index]
        elif split == 'validation':
            self.data = self.data[split_index:]
        else:
            raise ValueError("split must be 'train' or 'validation'")
        
        # Set up tokenizer
        self.tokenizer = get_tokenizer("spacy", language="it_core_news_sm")
        
        # Build vocabulary
        def yield_tokens(data_iter):
            for item in data_iter:
                yield self.tokenizer(item['text'])
        
        self.vocab = build_vocab_from_iterator(yield_tokens(self.data), 
                                               min_freq=min_freq, 
                                               specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        self.label_map = {label: idx for idx, label in enumerate(self.data[0]['choices'])}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # Tokenize and convert to indices
        tokens = self.tokenizer(text)
        indices = [self.vocab[token] for token in tokens]
        
        # Pad or truncate sequence
        if len(indices) < self.max_seq_length:
            indices += [self.vocab['<pad>']] * (self.max_seq_length - len(indices))
        else:
            indices = indices[:self.max_seq_length]
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(self.label_map[item['choices'][label]], dtype=torch.long)
        }
    
    def get_vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

# %%

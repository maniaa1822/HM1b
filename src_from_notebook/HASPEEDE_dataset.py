### Imports ###
import json
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, vocab
from torch.nn.utils.rnn import pack_padded_sequence
import random

# Set a fixed seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
random.seed(seed_value)

class HASPEEDE_Dataset(Dataset):
    def __init__(
        self,
        input_file_path,
        max_length=1028,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        tokenizer = get_tokenizer("spacy", language="it_core_news_sm")
        
        # Save samples from input file
        self.data = []
        with open(input_file_path, "r") as f:
            for line in f:
                sample = json.loads(line.strip())
                # the sample dictionary contains the following key: idx, sentence and label (integer)
                sample["tokens"] = tokenizer(sample["text"])
                self.data.append(sample)
                
        # Initialize indexed data attribute but leave it None
        # Must be filled through the `.index(vocabulary, label_vocabulary)` method
        # each dictionary represents a sentence with two keys: "input_ids" and "label"
        self.indexed_data: list[dict] | None = None
        
        #keep track of the maximum length to allow for a batch
        self.max_length = max_length
        #save the device
        self.device = torch.device(device)
        #keep track of padding id
        self.padding_id: int | None = None
        
    def get_raw_element(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.indexed_data[idx]
    
    def get_vocabulary(
        self,
        pad_token: str = '<pad>',
        unk_token: str = '<unk>',
        extra_tokens: list[str] = []
    )->Vocab:
        """Builds a `torchtext.vocab.Vocab` object from data stored in this object."""
        # most_common() returns a list of (token, count) pairs, so we convert them back into dictionary
        vocab_counter = dict(Counter(token for sent in self.data for token in sent["tokens"]).most_common())
        # We build the vocabulary through a dictionary like {token: frequency, ...}
        vocabulary = vocab(vocab_counter, min_freq=1, specials=[pad_token, unk_token, *extra_tokens])
        # vocabulary(list of tokens) returns a list of values, so get the only one
        vocabulary.set_default_index(vocabulary([unk_token])[0])
        return vocabulary

    def set_padding_id(self, value: int) -> None:
        self.padding_id = value

    def index(self, vocabulary: Vocab) -> None:
        """Builds `self.indexed_data` by converting raw data to input_ids following `vocabulary`"""
        if self.indexed_data is not None:
            print("Dataset has already been indexed. Keeping old index...")
        else:
            indexed_data = []
            for sample in self.data:
                # append the dictionary containing ids of the input tokens and label
                indexed_data.append({"input_ids": vocabulary(sample["tokens"]), "label": sample["label"]})
            self.indexed_data = indexed_data

    def _collate_fn(self, raw_batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batches together single elements of the dataset.
        This function:
        - converts each sentence made up of single input_ids to a padded tensor,
        - keeps track of the length of each sentence through `sequence_lengths`
        - builds a `labels` tensor storing the label for each sentence

        Args:
            raw_batch (list[dict]): a list of elements, as returned by the `__getitem__()` function.

        Returns:
            A tuple of three tensors, respectively `(sequence_lengths, padded_sequence, labels)`
        """
        if self.padding_id is None:
            raise RuntimeError("Padding value not set! Set it through .set_padding_id method.")

        # We need these sequence lengths to construct a `torch.nn.utils.rnn.PackedSequence` in the model
        sequence_lengths = torch.tensor([len(sample["input_ids"]) for sample in raw_batch], dtype=torch.long)
        padded_sequence = pad_sequence(
            (
                torch.tensor(sample["input_ids"], dtype=torch.long, device=self.device)
                for sample in raw_batch
            ),
            batch_first=True,
            padding_value=self.padding_id
        )
        labels = torch.tensor([sample["label"] for sample in raw_batch], device=self.device, dtype=torch.long)
        return sequence_lengths, padded_sequence, labels
        
        
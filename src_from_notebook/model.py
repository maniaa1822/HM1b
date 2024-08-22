# This file contains the model definition for the BiLSTM model, the baseline models and the BOW baseline model
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

### Imports ###


class BiLSTMModel(torch.nn.Module):
    """
    BiLSTMModel is a PyTorch module that implements a Bidirectional LSTM model for classification tasks.

    Args:
        vocabulary_length (int): The length of the vocabulary.
        hidden_dim (int): The dimensionality of the hidden state of the LSTM.
        bilstm_layers (int): The number of layers in the Bidirectional LSTM.
        bilstm_dropout (float): The dropout probability for the Bidirectional LSTM.
        num_classes (int): The number of output classes.
        padding_id (int): The index of the padding token in the vocabulary.
        device (str, optional): The device to run the model on (default: "cuda").

    Attributes:
        embedding (torch.nn.Embedding): The embedding layer.
        bilstm (torch.nn.LSTM): The Bidirectional LSTM layer.
        layer_norm (torch.nn.LayerNorm): The layer normalization layer.
        hidden_layer (torch.nn.Linear): The hidden layer.
        projection (torch.nn.Linear): The projection layer.
        relu (torch.nn.ReLU): The ReLU activation function.

    """

    def __init__(
        self,
        vocabulary_length: int,
        hidden_dim: int,
        bilstm_layers: int,
        bilstm_dropout: float,
        num_classes: int,
        padding_id: int,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        # Prepare the device
        self.device = torch.device(device)

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_length,
            embedding_dim=hidden_dim,
            padding_idx=padding_id, # avoid updating the gradient of padding entries
            device=self.device
            
        )
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=bilstm_layers,
            batch_first=True,
            dropout=bilstm_dropout,
            bidirectional=True,
            device=self.device
        )

        #layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2,
                                       device=self.device)
        
        self.hidden_layer = nn.Linear(
            in_features=hidden_dim * 2,
            out_features= hidden_dim,
            device=self.device
        )
            
        # Projection layer
        self.projection = nn.Linear(
            in_features=hidden_dim,
            out_features=num_classes,
            device=device
        )
    
        self.relu = nn.ReLU()
        
    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the BiLSTMModel.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing the sequence lengths and input IDs.

        Returns:
            torch.Tensor: The logits of each class.

        """
        # Get the different parts of the batch
        sequence_lengths, input_ids = batch

        # First we embed the input tokens
        embeds = self.embedding(input_ids) # [B, S, H]
        # where B is the batch size, S is the sequence length and H is the hidden dimension

        # Pack the sequence to avoid gradient descent on padding tokens.
        # An alternative to packing sequences is using masking.
        packed = pack_padded_sequence(embeds, sequence_lengths, batch_first=True, enforce_sorted=False)

        # Then we pass it to the BiLSTM
        # The first output of the BiLSTM tuple, packed_output, is of size B x S x 2H,
        # where B is the batch size, S is the sequence length and H is the hidden dimension
        # hidden_state is of size [2 * num_layers, B, H], where the 2 is because we are using BiLSTMs instead of LSTMs.
        # cell_state has size [2 * num_layers, B, C] where C is the cell dimension of the internal LSTMCell.
        packed_output, (hidden_state, cell_state) = self.bilstm(packed)

        # We take the last two hidden representations of the BiLSTM (the second-to-last layer's output is forward; last
        # layer's is backward) by concatenating forward and backward over dimension 1.
        # Both tensors have shapes of [B, H], so concatenating them along the second dimension (dim 1) results in a new
        # tensor of shape [B, 2 * H]
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)

        #layer normalization
        hidden = self.layer_norm(hidden)
        #hidden layer projection
        hidden = self.relu(self.hidden_layer(hidden))
        # Finally we project to the two final classes and return the logits of each class
        logits = self.projection(hidden) # [B, 2]
        return logits
    
class BaselineModels(nn.Module):
    """
    BaselineModels class represents a baseline model for text classification.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_classes (int): The number of classes for classification.
        baseline_type (str, optional): The type of baseline model. Defaults to 'majority'.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Attributes:
        baseline_type (str): The type of baseline model.
        num_classes (int): The number of classes for classification.
        device (torch.device): The device to use for computation.

    Methods:
        forward(batch): Performs forward pass of the model.
        majority_baseline(input_ids): Computes predictions using the majority baseline.
        random_baseline(input_ids): Computes predictions using the random baseline.
        prepare_batch(batch, device): Prepares the batch for model input.
        collate_fn(batch): Collates the batch for model input.

    """

    def __init__(self, vocab_size, num_classes, baseline_type='majority', device='cuda'):
        super().__init__()
        self.baseline_type = baseline_type
        self.num_classes = num_classes
        self.device = torch.device(device)

        if baseline_type == 'random':
            self.dummy = nn.Parameter(torch.randn(1))
        elif baseline_type == 'majority':
            self.dummy = nn.Parameter(torch.randn(1))
            
        self.to(self.device)
    
    def forward(self, batch):
        """
        Performs forward pass of the model.

        Args:
            batch (tuple): A tuple containing sequence lengths and input IDs.

        Returns:
            torch.Tensor: The predicted class probabilities.

        Raises:
            ValueError: If the baseline type is unknown.

        """
        sequence_lengths, input_ids = batch
        
        if self.baseline_type == 'majority':
            return self.majority_baseline(input_ids)
        elif self.baseline_type == 'random':
            return self.random_baseline(input_ids)
        elif self.baseline_type == 'bow':
            return self.bow_baseline(input_ids)
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")
    
    def majority_baseline(self, input_ids):
        """
        Computes predictions using the majority baseline.

        Args:
            input_ids (torch.Tensor): The input IDs.

        Returns:
            torch.Tensor: The predicted class probabilities.

        """
        batch_size = input_ids.size(0)
        # Always predict the majority class (assuming class 0 is the majority)
        return torch.zeros(batch_size, self.num_classes, device=self.device)
    
    def random_baseline(self, input_ids):
        """
        Computes predictions using the random baseline.

        Args:
            input_ids (torch.Tensor): The input IDs.

        Returns:
            torch.Tensor: The predicted class probabilities.

        """
        batch_size = input_ids.size(0)
        # Randomly predict classes
        return torch.randn(batch_size, self.num_classes, device=self.device)
    
    @staticmethod
    def prepare_batch(batch, device):
        """
        Prepares the batch for model input.

        Args:
            batch (tuple): A tuple containing sequence lengths, input IDs, and labels.
            device (str): The device to use for computation.

        Returns:
            tuple: A tuple containing the prepared batch and labels.

        """
        sequence_lengths, input_ids, labels = batch
        return (sequence_lengths, input_ids), labels

    @staticmethod
    def collate_fn(batch):
        """
        Collates the batch for model input.

        Args:
            batch (list): A list of tuples containing input IDs and labels.

        Returns:
            tuple: A tuple containing the collated sequence lengths, padded input IDs, and labels.

        """
        input_ids, labels = zip(*batch)
        sequence_lengths = torch.tensor([len(seq) for seq in input_ids])
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        return sequence_lengths, input_ids_padded, torch.tensor(labels)
    
    
class BOWBaseline(nn.Module):
    """
    Bag-of-Words Baseline model for text classification.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_classes (int): The number of classes for classification.
        device (str): The device to run the model on.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        num_classes (int): The number of classes for classification.
        device (str): The device to run the model on.
        linear (nn.Linear): Linear layer to project the BoW features to the number of classes.

    """

    def __init__(self, vocab_size, num_classes, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.device = device
        # Create a linear layer to project the BoW features to the number of classes
        self.linear = nn.Linear(vocab_size, num_classes, device=device)

    def forward(self, input_seq):
        """
        Forward pass of the BOWBaseline model.

        Args:
            input_seq (tuple): A tuple containing the sequence lengths and input IDs.

        Returns:
            torch.Tensor: The logits for each class.

        """
        batch_size = input_seq[0].size(0)
        seq_lengths, input_ids = input_seq

        # Create the Bag-of-Words representation
        bow_features = torch.zeros(batch_size, self.vocab_size, device=self.device)
        for i, ids in enumerate(input_ids):
            bow_features[i].index_add_(0, ids, torch.ones_like(ids,dtype=torch.float, device=self.device))
        seq_lengths = seq_lengths.to(self.device)    
        bow_features = bow_features / seq_lengths.unsqueeze(1)

        # Apply the linear layer to get the logits
        logits = self.linear(bow_features)
        return logits
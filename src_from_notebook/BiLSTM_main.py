#import
import torch
from torch import nn
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from HASPEEDE_dataset import HASPEEDE_Dataset
from model import BiLSTMModel
from trainer import Trainer
from torchsummary import summary

if __name__ == "__main__":
    pad_token, unk_token = "<pad>", "<unk>"
    device = "cuda"

    train_dataset = HASPEEDE_Dataset("/home/matteo/AI and Robotics/NLP/HM1b/HASPEEDE-20240708T122905Z-001/HASPEEDE/train-taskA.jsonl", device=device)
    validation_dataset = HASPEEDE_Dataset("/home/matteo/AI and Robotics/NLP/HM1b/HASPEEDE-20240708T122905Z-001/HASPEEDE/test-news-taskA.jsonl", device=device)
    test_dataset = HASPEEDE_Dataset("/home/matteo/AI and Robotics/NLP/HM1b/HASPEEDE-20240708T122905Z-001/HASPEEDE/test-tweets-taskA.jsonl", device=device)

    vocabulary = train_dataset.get_vocabulary(pad_token=pad_token, unk_token=unk_token)
    padding_id = vocabulary([pad_token])[0]

    train_dataset.set_padding_id(padding_id)
    validation_dataset.set_padding_id(padding_id)
    test_dataset.set_padding_id(padding_id)

    train_dataset.index(vocabulary)
    validation_dataset.index(vocabulary)
    test_dataset.index(vocabulary)

    print(f"Training len: {len(train_dataset)}")
    print(f"Validation len: {len(validation_dataset)}")
    print(f"Test len: {len(test_dataset)}")

    training_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=train_dataset._collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False, collate_fn=validation_dataset._collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=test_dataset._collate_fn)

    sentiment_classifier = BiLSTMModel(
        vocabulary_length=len(vocabulary),
        hidden_dim=128,
        bilstm_layers=8,
        bilstm_dropout=0.3,
        num_classes=2,
        padding_id=padding_id,
        device=device
    )

    trainer = Trainer(
        model=sentiment_classifier,
        optimizer=torch.optim.Adam(sentiment_classifier.parameters(), lr=0.0001),
        log_steps=10
    )

    losses = trainer.train(training_dataloader, validation_dataloader, epochs=10)
    test_loss, test_acc = trainer.evaluate(test_dataloader)
    
    print(f"BiLSTM - Test loss: {test_loss}, Test accuracy: {test_acc}")
    #save the results and model parameters and summary
    with open("BiLSTM_model_summary.txt", "w") as f:
        f.write(str(sentiment_classifier))
        f.write(f"\nTest loss: {test_loss}, Test accuracy: {test_acc}")
    with open("BiLSTM_losses.txt", "w") as f:
        f.write("\n".join(map(str, losses)))
    print("Model saved")
    print("Training and evaluation completed    ")
    
    
    
    
        

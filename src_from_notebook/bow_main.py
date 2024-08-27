from model import BiLSTMModel, BaselineModels, BOWBaseline
from HASPEEDE_dataset import HASPEEDE_Dataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch

pad_token, unk_token = "<pad>", "<unk>"
device = "cuda"

train_dataset = HASPEEDE_Dataset("/home/matteo/AI and Robotics/NLP/HM1b/HASPEEDE-20240708T122905Z-001/HASPEEDE/train-taskA.jsonl", device=device)
validation_dataset = HASPEEDE_Dataset("/home/matteo/AI and Robotics/NLP/HM1b/HASPEEDE-20240708T122905Z-001/HASPEEDE/test-tweets-taskA.jsonl", device=device)
test_dataset = HASPEEDE_Dataset("/home/matteo/AI and Robotics/NLP/HM1b/HASPEEDE-20240708T122905Z-001/HASPEEDE/test-news-taskA.jsonl", device=device)

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


bow_baseline = BOWBaseline(
    vocab_size=len(vocabulary),
    num_classes=2,
    device=device
)
# Function to evaluate a model without training
def evaluate_model(model, dataloader):
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
        log_steps=10
    )
    test_loss, test_acc = trainer.evaluate(dataloader)
    return test_loss, test_acc

def train_and_test(model, train_dataloader, valid_dataloader, test_dataloader, epochs=10, lr=0.001, log_steps=10):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        log_steps=log_steps
    )

    losses = trainer.train(train_dataloader, valid_dataloader, epochs=epochs)

    test_loss, test_acc = trainer.evaluate(test_dataloader)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    return losses, test_loss, test_acc

if __name__ == "__main__":
    #train and evaluate using the bow baseline
    bow_losses, bow_test_loss, bow_test_acc = train_and_test(bow_baseline, training_dataloader, validation_dataloader, test_dataloader)
    print(f"BoW Baseline - Test loss: {bow_test_loss}, Test accuracy: {bow_test_acc}")


    
    #save the results
    with open("BoW_results.txt", "w") as f:
        f.write(f"BoW Baseline - Test loss: {bow_test_loss}, Test accuracy: {bow_test_acc}\n")
        f.write(f"BoW Baseline - Losses: {bow_losses}\n")
        #write models hyperparameters
        f.write(f"BoW Baseline - Hyperparameters: {bow_baseline}\n")








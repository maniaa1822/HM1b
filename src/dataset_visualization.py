#%%
from lstm_dataset_class import LSTMTextClassificationDataset
import random
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_dataset(dataset, num_samples=10):
    tokenizer = get_tokenizer("spacy", language="it_core_news_sm")
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    fig.suptitle("Visualizing Processed Dataset Samples", fontsize=16)
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        
        original_text = dataset.data[idx]['text']
        processed_ids = sample['input_ids'].tolist()
        label = dataset.data[idx]['choices'][sample['label'].item()]
        
        # Decode processed ids back to tokens
        processed_tokens = [dataset.vocab.get_itos()[id] for id in processed_ids]
        
        # Visualize
        ax = axes[i]
        sns.heatmap([processed_ids], ax=ax, cmap='YlOrRd', cbar=False)
        ax.set_yticks([])
        ax.set_xticks(range(len(processed_tokens)))
        ax.set_xticklabels(processed_tokens, rotation=45, ha='right')
        ax.set_title(f"Sample {i+1} (Label: {label})")
        
        # Add original text as annotation
        ax.annotate(f"Original: {original_text[:100]}{'...' if len(original_text) > 100 else ''}", 
                    xy=(0, -0.3), xycoords='axes fraction', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Vocabulary size: {dataset.get_vocab_size()}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Max sequence length: {dataset.max_seq_length}")
    
    # Print label distribution
    label_dist = {}
    for item in dataset.data:
        label = item['choices'][item['label']]
        label_dist[label] = label_dist.get(label, 0) + 1
    print("Label distribution:")
    for label, count in label_dist.items():
        print(f"  {label}: {count} ({count/len(dataset)*100:.2f}%)")

# Usage:
# visualize_dataset(your_dataset_instance)
#split into two files for train and test

# %%
dataset = LSTMTextClassificationDataset('HASPEEDE-20240708T122905Z-001/HASPEEDE/train-taskA.jsonl', split='train', split_ratio=0.8)
# %%
visualize_dataset(dataset, num_samples=5)
# %%

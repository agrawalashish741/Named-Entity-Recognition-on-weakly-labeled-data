import torch
from transformers import (
    BertTokenizer,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    BertForMaskedLM,
    Trainer,
    TrainingArguments
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42  # Set your seed for reproducibility
setup_seed(SEED)

# Define training parameters directly in the script
pretrained_weights_path = 'bert-base-cased'  # Path to pre-trained BERT weights
unsupervised_train_data_path = '/kaggle/input/test-data/output_text (2).txt'  # Path to training data
do_lower_case = True  # Use lowercased text
max_len = 512  # Maximum token length
mlm_probability = 0.15  # Probability for masking tokens
lr = 5e-5  # Learning rate
epoch_num = 1  # Number of epochs
batch_size = 8  # Batch size

# Select the device (GPU if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_weights_path, do_lower_case=do_lower_case)

# Load the dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=unsupervised_train_data_path,
    block_size=max_len
)

# Data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=mlm_probability
)

# Load the pre-trained BERT model for masked language modeling
mlm = BertForMaskedLM.from_pretrained(pretrained_weights_path).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    learning_rate=lr,
    num_train_epochs=epoch_num,
    per_device_train_batch_size=batch_size,
    save_strategy='epoch',
    seed=SEED
)

# Initialize the Trainer
trainer = Trainer(
    model=mlm,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save the model weights
mlm.save_pretrained('./pretrained_weights')
tokenizer.save_pretrained('./pretrained_weights')

pip install pytorch-crf

import torch
from transformers import BertTokenizerFast, BertModel, Trainer, TrainingArguments
from torchcrf import CRF
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
# Custom Dataset class
class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) if key != 'word_ids' else val[idx]
            for key, val in self.encodings.items()
        }
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    collated = {key: [] for key in keys}
    for b in batch:
        for key in keys:
            collated[key].append(b[key])
    # Stack tensors where applicable
    for key in ['input_ids', 'attention_mask', 'labels']:
        collated[key] = torch.stack(collated[key])
    return collated

def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r') as file:
        sentence = []
        label_seq = []
        for line in file:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label_seq)
                    sentence = []
                    label_seq = []
            else:
                parts = line.split()
                if len(parts) == 2:  # Ensure there are exactly two parts
                    word, label = parts
                    sentence.append(word)
                    label_seq.append(label)
                else:
                    continue
        if sentence:
            sentences.append(sentence)
            labels.append(label_seq)
    return sentences, labels


# Function to tokenize and align labels
def tokenize_and_align_labels(sentences, labels, label_map):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_offsets_mapping=True,
    )
    all_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Assign the label for 'O' to special tokens
                label_ids.append(label_map['O'])
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]])
            else:
                # For subword tokens, assign -100 so they are ignored in the loss computation
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

class BERT_CRF_NER(nn.Module):
    def __init__(self, num_labels, pretrained_weights_path):
        super(BERT_CRF_NER, self).__init__()
        # Load custom pretrained BERT weights
        self.bert = BertModel.from_pretrained(pretrained_weights_path, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = label_map['O']
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            return emissions, predictions
        
from tqdm import tqdm
# Load custom pretrained tokenizer
tokenizer = BertTokenizerFast.from_pretrained('./pretrained_weights')

# BERT + CRF Model
# class BERT_CRF_NER(nn.Module):
#     def __init__(self, num_labels, pretrained_weights_path):
#         super(BERT_CRF_NER, self).__init__()
#         # Load custom pretrained BERT weights
#         self.bert = BertModel.from_pretrained(pretrained_weights_path)
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
#         self.crf = CRF(num_labels, batch_first=True)

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         sequence_output = self.dropout(outputs[0])
#         emissions = self.classifier(sequence_output)
        
#         if labels is not None:
#             # CRF cannot handle -100 labels, so replace with label for 'O'
#             labels = labels.clone()
#             labels[labels == -100] = label_map['O']
#             mask = attention_mask.bool()
#             loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
#             return loss
#         else:
#             mask = attention_mask.bool()
#             predictions = self.crf.decode(emissions, mask=mask)
#             return predictions

# Load data

train_sentences, train_labels = load_data('/kaggle/input/bc5cdr-disease/train.txt')
# dev_sentences, dev_labels = load_data('/kaggle/input/ncbi-ka-disease/dev.txt')
# test_sentences, test_labels = load_data('/kaggle/input/ncbi-ka-disease/test.txt')

# Create label mappings
unique_labels = set(label for doc in train_labels for label in doc)
label_list = sorted(unique_labels)
label_map = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label_map.items()}
num_labels = len(label_list)

# Ensure 'O' label exists in label_map
if 'O' not in label_map:
    label_map['O'] = len(label_map)
    id2label[len(label_map) - 1] = 'O'
    num_labels += 1

# Tokenize and align labels
train_encodings = tokenize_and_align_labels(train_sentences, train_labels, label_map)
# dev_encodings = tokenize_and_align_labels(dev_sentences, dev_labels, label_map)
# test_encodings = tokenize_and_align_labels(test_sentences, test_labels, label_map)

# Create datasets
train_dataset = NERDataset(train_encodings)
# dev_dataset = NERDataset(dev_encodings)
# test_dataset = NERDataset(test_encodings)

# Initialize model
model = BERT_CRF_NER(num_labels, './pretrained_weights')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training setup
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

# Evaluate model
# test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)
# true_labels, predictions = evaluate_model(model, test_loader, id2label)

# # Print evaluation results
# print(classification_report(true_labels, predictions))
# print(f'F1 Score: {f1_score(true_labels, predictions)}')

# model.save_pretrained('./pretrained_weights_with_CRF')
# tokenizer.save_pretrained('./pretrained_weights_with_CRF')

import os

# Specify the directory to save the model
output_dir = './pretrained_weights_with_CRF'
os.makedirs(output_dir, exist_ok=True)

# Save the model's state_dict (parameters)
torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

# Save the tokenizer
tokenizer.save_pretrained(output_dir)
print(f"Model weights and tokenizer saved to {output_dir}")

# Load the sentences and labels from the file
weak_train_sentences, weak_train_labels = load_data("/kaggle/input/weak-labelled-data/l3.txt")

weak_unique_labels = set(label for doc in weak_train_labels for label in doc)
weak_label_list = sorted(weak_unique_labels)
weak_label_map = {label: i for i, label in enumerate(weak_label_list)}
weak_id2label = {i: label for label, i in weak_label_map.items()}
weak_num_labels = len(weak_label_list)

# Ensure 'O' label exists in label_map
if 'O' not in weak_label_map:
    weak_label_map['O'] = len(weak_label_map)
    weak_id2label[len(weak_label_map) - 1] = 'O'
    weak_num_labels += 1

# Tokenize and align labels
weak_train_encodings = tokenize_and_align_labels(weak_train_sentences, weak_train_labels, weak_label_map)

# print("Original:", weak_train_encodings["labels"])

# Create datasets
weak_train_dataset = NERDataset(weak_train_encodings)

tokenizer = BertTokenizerFast.from_pretrained("./pretrained_weights_with_CRF")
# model = BertForTokenClassification.from_pretrained("./pretrained_weights_with_CRF", ignore_mismatched_sizes=True)  # Load your BERT-CRF model
weak_model = BERT_CRF_NER(weak_num_labels, "./pretrained_weights_with_CRF").to(device)

from torch.utils.data import DataLoader

def complete_labels(dataset, weak_model, tokenizer, label_map, device):
    model.eval()
    completed_labels = []

    # Prepare DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Iterate through batches and predict labels
    for batch in tqdm(dataloader):
        # Move batch to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        original_labels = batch['labels']  # Keeping original labels for reference
        
        # Get model predictions
        with torch.no_grad():
            emissions, predictions = weak_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Loop through predictions and original labels to complete labels
        for i, pred in enumerate(predictions):
            completed_label_seq = []
            for j, token_label in enumerate(original_labels[i]):
                if j < len(pred):
                    token_label = original_labels[i][j].item()

                    if token_label == label_map['O'] and pred[j] != label_map['O'] and pred[j] != -100:
                        # Replace 'O' label in weak data with predicted label if not -100
                        completed_label_seq.append(pred[j])
                    else:
                        # Keep the original label
                        completed_label_seq.append(token_label)
                else:
                    # If no prediction, default to original label (likely padding)
                    completed_label_seq.append(original_labels[i][j].item())
            completed_labels.append(completed_label_seq)
    
    return completed_labels

# Complete the labels for weak_train_dataset
completed_labels = complete_labels(weak_train_dataset, weak_model, tokenizer, weak_label_map, device)

# Add the completed labels back into the encodings
weak_train_encodings["labels"] = completed_labels

# print("Completed:", weak_train_encodings["labels"])

# Update the dataset with completed labels
completed_weak_train_dataset = NERDataset(weak_train_encodings)

import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    # Extract input_ids, attention_mask, labels, and confidences from each item in the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    confidences = [item['confidences'] for item in batch]

    # Pad sequences to the same length within the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    confidences_padded = pad_sequence(confidences, batch_first=True, padding_value=0.0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'confidences': confidences_padded
    }

from torch.utils.data import ConcatDataset

import torch
from transformers import AutoTokenizer, BertForTokenClassification
from torch.nn.functional import softmax
import numpy as np

# Load the BERT tokenizer and BERT-CRF model (replace 'your-bert-crf-model' with your model path)
tokenizer = BertTokenizerFast.from_pretrained("./pretrained_weights_with_CRF")
# model = BertForTokenClassification.from_pretrained("./pretrained_weights_with_CRF", ignore_mismatched_sizes=True)  # Load your BERT-CRF model

# Load the sentences and labels from the file
# weak_train_sentences, weak_train_labels = load_data("/kaggle/input/test-weak/test_weak.txt")

combined_sentences = train_sentences + weak_train_sentences
combined_labels = train_labels + weak_train_labels

combined_unique_labels = set(label for doc in combined_labels for label in doc)
combined_label_list = sorted(combined_unique_labels)
combined_label_map = {label: i for i, label in enumerate(combined_label_list)}
combined_id2label = {i: label for label, i in combined_label_map.items()}
combined_num_labels = len(combined_label_list)

# print(combined_num_labels)

model = BERT_CRF_NER(combined_num_labels, "./pretrained_weights_with_CRF").to(device)

# Ensure 'O' label exists in label_map
if 'O' not in weak_label_map:
    weak_label_map['O'] = len(weak_label_map)
    weak_id2label[len(weak_label_map) - 1] = 'O'
    weak_num_labels += 1

# Tokenize and align labels
# combined_encodings = tokenize_and_align_labels(combined_sentences, combined_labels, combined_label_map)

# Create datasets
# combined_dataset = NERDataset(combined_encodings)

# Function to calculate confidence for each label
def get_confidence_scores(logits):
    probs = softmax(logits, dim=2)  # Convert logits to probabilities
    confidences = torch.max(probs, dim=2)[0].detach().cpu().numpy()  # Get max probability for each token
    return confidences

combined_dataset = ConcatDataset([train_dataset, completed_weak_train_dataset])
# # Load the sentences from the file
# sentences = read_and_tokenize("/kaggle/input/test-weak/test_weak.txt")

# # Iterate through sentences, perform predictions, and estimate confidence
# for sentence in sentences:
#     inputs = tokenizer(sentence, return_tensors="pt", is_split_into_words=True, truncation=True)
#     inputs.pop("token_type_ids", None)  # Remove token_type_ids if it exists
#     with torch.no_grad():
#         # Pass through the model and get logits from the CRF layer
#         emissions, predictions = model(**inputs)
#         # logits = outputs.logits  # Assuming model outputs logits for CRF
#         confidences = get_confidence_scores(emissions)
        
#         # Print results for each token in the sentence
#         for token, confidence in zip(sentence, confidences[0]):
#             print(f"Token: {token}, Confidence: {confidence:.4f}")

from torch.utils.data import DataLoader

class ConfidenceDatasetWrapper:
    def __init__(self, dataset, model, tokenizer, device):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original data from the dataset
        batch = self.dataset[idx]

        # Extract input_ids and attention_mask
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Process the data through the model to get confidence scores
        with torch.no_grad():
            emissions, predictions = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            confidences = torch.tensor(get_confidence_scores(emissions)).squeeze(0)   # Remove batch dimension for confidences

        # Add confidence scores to the batch
        batch['confidences'] = confidences  # Convert tensor to list for compatibility
        
        return batch

# Wrap your original dataset with the confidence wrapper
train_dataset_with_confidences = ConfidenceDatasetWrapper(combined_dataset, model, tokenizer, device)
train_loader_with_confidences = DataLoader(train_dataset_with_confidences, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

# Now each batch in train_loader_with_confidences will have 'input_ids', 'attention_mask', 'labels', and 'confidences'
model = BERT_CRF_NER(num_labels, "./pretrained_weights_with_CRF").to(device)
train_dataset_with_confidences = ConfidenceDatasetWrapper(train_dataset, model, tokenizer, device)
train_loader_with_confidences = DataLoader(train_dataset_with_confidences, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

import matplotlib.pyplot as plt
import numpy as np

# Collect CRF scores and corresponding confidence scores
all_crf_scores = []
all_confidences = []

# Iterate through the data loader to extract CRF scores and confidence scores
for batch in train_loader_with_confidences:
    confidences = batch['confidences']  # Extract confidences
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        # Get CRF scores (emissions) from the model
        emissions, _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # Process each token's score
    for crf_score, confidence in zip(emissions.cpu().numpy(), confidences):
        all_crf_scores.extend(crf_score.max(axis=1).tolist())  # Max CRF score per token
        all_confidences.extend(confidence.cpu().numpy().tolist())  # Confidence score per token

# Bin CRF scores and calculate average confidence per bin
bin_edges = np.linspace(min(all_crf_scores), max(all_crf_scores), 50)  # 50 bins for CRF scores
bin_indices = np.digitize(all_crf_scores, bin_edges) - 1  # Get bin index for each CRF score

average_confidence_per_bin = [
    np.mean([all_confidences[i] for i in range(len(all_crf_scores)) if bin_indices[i] == j])
    for j in range(len(bin_edges) - 1)
]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], average_confidence_per_bin, width=(bin_edges[1] - bin_edges[0]), edgecolor="black", align="edge")
plt.xlabel("CRF Score")
plt.ylabel("Average Confidence Score")
plt.title("Average Confidence Score by CRF Score")
plt.show()




def noise_aware_loss_function(emissions, weak_labels, confidence_scores, crf_layer, attention_mask):
    """
    Noise-aware loss function with CRF layer.
    
    Parameters:
    - emissions: The output logits from the model (before CRF).
    - weak_labels: Weakly labeled target labels for tokens.
    - confidence_scores: Confidence scores for each weak label.
    - crf_layer: The CRF layer used to calculate log-likelihood.
    - attention_mask: Attention mask to ignore padding tokens.

    Returns:
    - Mean noise-aware loss across the batch.
    """

     # Clone and replace -100 in weak_labels with label_map['O']
    weak_labels = weak_labels.clone()
    ignore_label = label_map['O']
    weak_labels[weak_labels == -100] = ignore_label
    
    # Apply attention mask and mask for valid labels (ignoring -100 replacements)
    # mask = attention_mask.bool() & (weak_labels != ignore_label)
    # Apply attention mask
    mask = attention_mask.bool()
    
    # Negative log-likelihood for correct labels (per sequence)
    nll_loss = -crf_layer(emissions, weak_labels, mask=mask, reduction='none')  # Shape: (batch_size)
    
    # Compute per-instance confidence by averaging over valid tokens
    valid_confidences = confidence_scores * mask  # Shape: (batch_size, seq_length)
    sum_confidences = valid_confidences.sum(dim=1)  # Shape: (batch_size)
    num_valid_tokens = mask.sum(dim=1) + 1e-8  # Avoid division by zero
    per_instance_confidence = sum_confidences / num_valid_tokens  # Shape: (batch_size)
    
    # Calculate the noise-aware loss
    noise_aware_loss = (per_instance_confidence * nll_loss).mean()
    return noise_aware_loss
    
# Training loop with noise-aware loss
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 1

for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_loader_with_confidences):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        weak_labels = batch["labels"].to(device)
        confidence_scores = batch["confidences"].to(device)
        
        # Forward pass through model to get emissions
        emissions = model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = model.classifier(emissions)  # Logits before CRF

        # Calculate noise-aware loss
        loss = noise_aware_loss_function(emissions, weak_labels, confidence_scores, model.crf, attention_mask)
        total_loss += loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader_with_confidences)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

import os

# Specify the directory to save the model
output_dir = './noise_aware_pretrained_weights_with_CRF'
os.makedirs(output_dir, exist_ok=True)

# Save the model's state_dict (parameters)
torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

# Save the tokenizer
tokenizer.save_pretrained(output_dir)
print(f"Model weights and tokenizer saved to {output_dir}")

pip install seqeval

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from torchcrf import CRF
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import classification_report, f1_score

# Update collate_fn to handle word_ids
def collate_fn(batch):
    keys = batch[0].keys()
    collated = {key: [] for key in keys}
    for b in batch:
        for key in keys:
            collated[key].append(b[key])
    # Stack tensors where applicable, excluding 'word_ids' which should remain a list of lists
    for key in ['input_ids', 'attention_mask', 'labels']:
        collated[key] = torch.stack(collated[key])
    return collated

# Update __getitem__ in NERDataset
class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) if key != 'word_ids' else val[idx]
            for key, val in self.encodings.items()
        }
        return item

    
# BERT + CRF Model
class BERT_CRF_NER(nn.Module):
    def __init__(self, num_labels):
        super(BERT_CRF_NER, self).__init__()
        self.bert = BertModel.from_pretrained('/kaggle/working/noise_aware_pretrained_weights_with_CRF', ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # Replace -100 in labels as CRF cannot handle -100 labels
            labels = labels.clone()
            labels[labels == -100] = label_map['O']  # Assuming 'O' is the label for non-entity tokens
            # Use attention_mask as the mask for CRF
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            return emissions, predictions

# Function to load data
def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r') as file:
        sentence = []
        label_seq = []
        for line in file:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label_seq)
                    sentence = []
                    label_seq = []
            else:
                word, label = line.split()
                sentence.append(word)
                label_seq.append(label)
        if sentence:
            sentences.append(sentence)
            labels.append(label_seq)
    return sentences, labels

# Function to tokenize and align labels
def tokenize_and_align_labels(sentences, labels, label_map):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_offsets_mapping=True,
    )
    all_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Assign the label for 'O' to special tokens
                label_ids.append(label_map['O'])
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]])
            else:
                # For subword tokens, assign -100 so they are ignored in the loss computation
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

# Load data
train_sentences, train_labels = load_data('/kaggle/input/bc5cdr-disease/train.txt')
dev_sentences, dev_labels = load_data('/kaggle/input/bc5cdr-disease/dev.txt')
test_sentences, test_labels = load_data('/kaggle/input/bc5cdr-disease/test.txt')

# Create label mappings
unique_labels = set(label for doc in train_labels for label in doc)
label_list = sorted(unique_labels)
label_map = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label_map.items()}
num_labels = len(label_list)

# Ensure 'O' label exists in label_map
if 'O' not in label_map:
    label_map['O'] = len(label_map)
    id2label[len(label_map) - 1] = 'O'
    num_labels += 1

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('/kaggle/working/noise_aware_pretrained_weights_with_CRF')

# Tokenize and align labels
train_encodings = tokenize_and_align_labels(train_sentences, train_labels, label_map)
dev_encodings = tokenize_and_align_labels(dev_sentences, dev_labels, label_map)
test_encodings = tokenize_and_align_labels(test_sentences, test_labels, label_map)

# Create datasets
train_dataset = NERDataset(train_encodings)
dev_dataset = NERDataset(dev_encodings)
test_dataset = NERDataset(test_encodings)

# Initialize model
model = BERT_CRF_NER(num_labels)

# Training setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

# Create DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Function to evaluate model
def evaluate(model, dataloader, label_map, id2label):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get predictions
            _, preds = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Convert predictions and labels to lists
            preds = [[id2label[label] for label in pred] for pred in preds]
            labels = [[id2label[label.item()] if label.item() != -100 else 'O' for label in sent] for sent in labels]
            
            # Remove padding and subword labels
            for i in range(len(labels)):
                true_sent, pred_sent = [], []
                for j in range(len(labels[i])):
                    if labels[i][j] != 'O' or attention_mask[i][j].item() == 1:
                        true_sent.append(labels[i][j])
                        pred_sent.append(preds[i][j])
                true_labels.append(true_sent)
                predictions.append(pred_sent)
    
    # Calculate metrics
    report = classification_report(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return report, f1

# Run evaluation
report, f1 = evaluate(model, test_loader, label_map, id2label)
print("Classification Report:\n", report)
print("F1 Score:", f1)





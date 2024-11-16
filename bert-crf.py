!pip install simpletransformers
pip install transformers torch seqeval pytorch-crf

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from torchcrf import CRF
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import classification_report, f1_score

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


    
# BERT + CRF Model
class BERT_CRF_NER(nn.Module):
    def __init__(self, num_labels):
        super(BERT_CRF_NER, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
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
            return predictions

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
def tokenize_and_align_labels(tokenizer, sentences, labels, label_map):
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

# Evaluation function
def evaluate_model(model, dataloader, id2label):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            word_ids_batch = batch['word_ids']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                label_ids = labels[i].numpy()
                pred_ids = outputs[i]
                word_ids = word_ids_batch[i]
                true_labels_seq = []
                pred_labels_seq = []
                previous_word_idx = None
                for idx, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        continue
                    if word_idx != previous_word_idx:
                        label_id = label_ids[idx]
                        if label_id == -100:
                            continue
                        true_labels_seq.append(id2label[label_id])
                        pred_labels_seq.append(id2label[pred_ids[idx]])
                    previous_word_idx = word_idx
                true_labels.append(true_labels_seq)
                predictions.append(pred_labels_seq)
    return true_labels, predictions


def train_and_evaluate_model(train_file, dev_file, test_file, num_epochs=5, batch_size=16, lr=1e-5):
    # Load data
    train_sentences, train_labels = load_data(train_file)
    dev_sentences, dev_labels = load_data(dev_file)
    test_sentences, test_labels = load_data(test_file)

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
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    # Tokenize and align labels
    train_encodings = tokenize_and_align_labels(tokenizer, train_sentences, train_labels, label_map)
    dev_encodings = tokenize_and_align_labels(tokenizer, dev_sentences, dev_labels, label_map)
    test_encodings = tokenize_and_align_labels(tokenizer, test_sentences, test_labels, label_map)

    # Create datasets
    train_dataset = NERDataset(train_encodings)
    dev_dataset = NERDataset(dev_encodings)
    test_dataset = NERDataset(test_encodings)

    # Initialize model
    model = BERT_CRF_NER(num_labels)

    # Training setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
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

    # Evaluate model on test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    true_labels, predictions = evaluate_model(model, test_loader, id2label)

    # Print evaluation results
    print(classification_report(true_labels, predictions))
    print(f'F1 Score: {f1_score(true_labels, predictions)}')

# Example usage
train_and_evaluate_model('/kaggle/input/ncbi_disease/train.txt',
                   '/kaggle/input/ncbi_disease/dev.txt',
                   '/kaggle/input/ncbi_disease/test.txt')

train_and_evaluate_model('/kaggle/input/BC5CDR_chem/train.txt',
                   '/kaggle/input/BC5CDR_chem/dev.txt',
                   '/kaggle/input/BC5CDR_chem/test.txt')

train_and_evaluate_model('/kaggle/input/BC5CDR_disease/train.txt',
                   '/kaggle/input/BC5CDR_disease/dev.txt',
                   '/kaggle/input/BC5CDR_disease/test.txt')
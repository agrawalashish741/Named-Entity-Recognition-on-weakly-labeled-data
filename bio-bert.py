pip install transformers datasets
!pip install seqeval

# Import required libraries
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=3)  # We will update num_labels later

# Load and preprocess the data from text file
def load_data(file_path):
    tokens, labels = [], []
    current_tokens, current_labels = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "":  # End of sentence
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                current_tokens, current_labels = [], []
            else:
                splits = line.strip().split()
                if len(splits) >= 2:
                    word, tag = splits[0], splits[-1]
                    current_tokens.append(word)
                    current_labels.append(tag)
        # Add the last sentence if file doesn't end with a newline
        if current_tokens:
            tokens.append(current_tokens)
            labels.append(current_labels)
    return tokens, labels

def get_unique_labels(*label_lists):
    unique_labels = set()
    for labels in label_lists:
        for label_seq in labels:
            unique_labels.update(label_seq)
    return list(unique_labels)

def convert_labels_to_ids(labels, label_to_id):
    return [[label_to_id[label] for label in sentence] for sentence in labels]

# Prepare dataset
def prepare_dataset(tokens, labels):
    dataset = Dataset.from_dict({"tokens": tokens, "labels": labels})
    return dataset

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # For subword tokens, we set the label to -100
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(pred, id_to_label):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        seq_true_labels = []
        seq_predictions = []
        for pred_label_id, true_label_id in zip(prediction, label):
            if true_label_id != -100:
                seq_true_labels.append(id_to_label[true_label_id])
                seq_predictions.append(id_to_label[pred_label_id])
        true_labels.append(seq_true_labels)
        true_predictions.append(seq_predictions)

    # Compute metrics
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def train_and_evaluate_model(train_file, val_file, test_file, output_dir="./results", num_epochs=3, learning_rate=2e-5, batch_size=16):
    # Load train, validation, and test datasets
    train_tokens, train_labels = load_data(train_file)
    val_tokens, val_labels = load_data(val_file)
    test_tokens, test_labels = load_data(test_file)

    # Collect all unique labels
    unique_labels = get_unique_labels(train_labels, val_labels, test_labels)

    # Define label mappings (dynamically)
    label_list = sorted(unique_labels)  # Sort for consistency
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    # Update num_labels in the model based on the unique labels
    model.config.num_labels = len(label_list)

    # Prepare datasets
    train_dataset = prepare_dataset(train_tokens, convert_labels_to_ids(train_labels, label_to_id))
    val_dataset = prepare_dataset(val_tokens, convert_labels_to_ids(val_labels, label_to_id))
    test_dataset = prepare_dataset(test_tokens, convert_labels_to_ids(test_labels, label_to_id))

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['tokens', 'labels'])
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['tokens', 'labels'])
    test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['tokens', 'labels'])

    # Initialize the data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test dataset
    results = trainer.evaluate(test_dataset)
    print("Test Metrics:")
    for key, value in results.items():
        if key.startswith("eval_"):
            print(f"{key}: {value:.4f}")

    # Optionally, print the classification report
    predictions, labels, _ = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        seq_true_labels = []
        seq_predictions = []
        for pred_label_id, true_label_id in zip(prediction, label):
            if true_label_id != -100:
                seq_true_labels.append(id_to_label[true_label_id])
                seq_predictions.append(id_to_label[pred_label_id])
        true_labels.append(seq_true_labels)
        true_predictions.append(seq_predictions)

    print("\nClassification Report:")
    print(classification_report(true_labels, true_predictions))

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
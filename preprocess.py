import xml.etree.ElementTree as ET

def convert_xml_to_text_preserving_format(xml_path, output_text_file):
    """
    Converts an XML file into a plain text file, preserving the original format (without adding any new lines).
    
    :param xml_path: Path to the XML file
    :param output_text_file: Path to the output text file (.txt)
    """
    all_text = []

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract text from all XML elements and concatenate without adding new lines
    for elem in root.iter():
        text = elem.text
        if text:
            text = text.strip()
            if text != "":
                all_text.append(text)

    # Join the text into a single string, separated by spaces
    final_text = ' '.join(all_text)

    # Write the extracted text to a text file as a single block of text
    with open(output_text_file, 'w') as out_file:
        out_file.write(final_text)

    print(f"Successfully extracted text to {output_text_file}")

# Usage example
convert_xml_to_text_preserving_format('/kaggle/input/pubmed-data/pubmed24n0009.xml', 'output_text.txt')

!pip install flashtext
import numpy as np
import random
from itertools import chain
from tqdm import tqdm
import ray
from flashtext import KeywordProcessor
ray.init(ignore_reinit_error=True)

import spacy

random.seed(0)
np.random.seed(seed=0)

sampling_rate_no_weak_labels = 0.1

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
nlp_tokenizer = spacy.blank('en')

with open('/kaggle/input/abcder/chem_dict.txt', 'r') as f:
    dict_chem = [x.strip() for x in f if x.strip() != ""]
    
with open('/kaggle/input/abcder/disease_dict.txt', 'r') as f:
    dict_disease = [x.strip() for x in f if x.strip() != ""]
    
print(len(dict_chem))
print(len(dict_disease))

entity_to_type = {}
for entity in dict_chem:
    entity_to_type[entity] = 'Chemical'
for entity in dict_disease:
    entity_to_type[entity] = 'Disease'

@ray.remote
def process_chunk(text_lines, entity_to_type):
    labeled_lines = []
    unlabeled_lines = []

    # Initialize KeywordProcessor
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for entity, entity_type in entity_to_type.items():
        keyword_processor.add_keyword(entity, entity_type)

    for line in text_lines:
        doc = nlp_tokenizer(line)
        tokens = [token.text for token in doc]
        labels = ['O'] * len(tokens)
        token_positions = [(token.idx, token.idx + len(token)) for token in doc]

        matches = keyword_processor.extract_keywords(line, span_info=True)
        for match in matches:
            matched_entity_type = match[0]  # 'Chemical' or 'Disease'
            start_pos, end_pos = match[1], match[2]

            # Find the tokens that correspond to this match
            for i, (token_start, token_end) in enumerate(token_positions):
                if token_end <= start_pos:
                    continue
                if token_start >= end_pos:
                    break
                if labels[i] == 'O':
                    if token_start == start_pos:
                        labels[i] = 'B-' + matched_entity_type
                    else:
                        labels[i] = 'I-' + matched_entity_type

        if all(label == 'O' for label in labels):
            unlabeled_lines.append([tokens, labels])
        else:
            labeled_lines.append([tokens, labels])

    return labeled_lines, unlabeled_lines

# Load the input file
with open('/kaggle/input/abcder/testing.txt', 'r') as input_file:
    all_text = input_file.read()

# Convert the text into sentences using NLTK
all_sentences = sent_tokenize(all_text)

# Adjust the number of chunks based on CPU count
num_cpus = multiprocessing.cpu_count()
num_chunks = num_cpus * 2  # Adjust this multiplier based on performance
chunk_size = len(all_sentences) // num_chunks + 1

# Prepare the chunks
chunks = [all_sentences[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

# Start parallel processing
futures = [process_chunk.remote(chunk, entity_to_type) for chunk in chunks]

# Collect results with progress bar
all_processed_data = []
for result in tqdm(ray.get(futures), total=len(futures)):
    all_processed_data.append(result)

# Combine results from all workers
labeled_lines = list(chain.from_iterable([x[0] for x in all_processed_data]))
unlabeled_lines = list(chain.from_iterable([x[1] for x in all_processed_data]))

# Write the labeled data to a file
with open('l1.txt', 'w') as output_file:
    for tokens, labels in labeled_lines:
        for word, label in zip(tokens, labels):
            output_file.write(f"{word}\t{label}\n")
        output_file.write("\n")  # Separate sentences by a blank line

print("Processing completed. Labeled data saved to 'l1.txt'.")
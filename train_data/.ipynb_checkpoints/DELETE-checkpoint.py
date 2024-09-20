import json
import random
import re

# Function to clean edition text
def clean_edition_text(text):
    text = text.replace("⟨⟩", "")  # Replace "⟨⟩" with ""
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()

# File path to the processed_iphi.jsonl file
input_file = 'data/processed_iphi.jsonl'

# Output file paths
train_output_file = 'data/inscr_train.jsonl'
test_output_file = 'data/inscr_test.jsonl'

# Read and process the data
processed_entries = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)

        # Rename fields as per the new requirements
        entry['PHI_ID'] = entry.pop('id')
        entry['Edition_with_brackets'] = clean_edition_text(entry.pop('text', ''))
        entry['Edition_without_brackets'] = clean_edition_text(entry.pop('text_nobrackets', ''))
        entry['place'] = entry.pop('region')

        # Check if both editions are shorter than 30 characters
        if len(entry['Edition_with_brackets']) >= 30 or len(entry['Edition_without_brackets']) >= 30:
            processed_entries.append(entry)

# Shuffle the processed entries to randomize the train/test split
random.shuffle(processed_entries)

# Determine the split point for 95%/5% split
split_index = int(0.95 * len(processed_entries))

# Split the data into train and test sets
train_entries = processed_entries[:split_index]
test_entries = processed_entries[split_index:]

# Write the train entries to the inscr_train.jsonl file
with open(train_output_file, 'w', encoding='utf-8') as train_file:
    for entry in train_entries:
        train_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Write the test entries to the inscr_test.jsonl file
with open(test_output_file, 'w', encoding='utf-8') as test_file:
    for entry in test_entries:
        test_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Train and test files created successfully!")
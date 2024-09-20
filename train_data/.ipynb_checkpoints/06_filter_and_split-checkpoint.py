import json
import random
import re

# Function to clean edition text
def clean_edition_text(text):
    text = text.replace("⟨⟩", "")  # Replace "⟨⟩" with ""
    text = re.sub(r"\s+", " ", text)  # Replace "\s*" with a single space
    return text.strip()

# File path to the united.jsonl file
input_file = 'data/cleaned_united.jsonl'

# Output file paths
train_output_file = 'data/train.jsonl'
test_output_file = 'data/test.jsonl'

# Read and filter the data
filtered_entries = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        edition_with_brackets = clean_edition_text(entry.get('Edition_with_brackets', ''))
        edition_without_brackets = clean_edition_text(entry.get('Edition_without_brackets', ''))

        # Check if both editions are shorter than 30 characters
        if len(edition_with_brackets) >= 30 or len(edition_without_brackets) >= 30:
            entry['Edition_with_brackets'] = edition_with_brackets
            entry['Edition_without_brackets'] = edition_without_brackets
            filtered_entries.append(entry)

# Shuffle the filtered entries to randomize the train/test split
random.shuffle(filtered_entries)

# Determine the split point for 95%/5% split
split_index = int(0.95 * len(filtered_entries))

# Split the data into train and test sets
train_entries = filtered_entries[:split_index]
test_entries = filtered_entries[split_index:]

# Write the train entries to the train.jsonl file
with open(train_output_file, 'w', encoding='utf-8') as train_file:
    for entry in train_entries:
        train_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Write the test entries to the test.jsonl file
with open(test_output_file, 'w', encoding='utf-8') as test_file:
    for entry in test_entries:
        test_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Train and test files created successfully!")
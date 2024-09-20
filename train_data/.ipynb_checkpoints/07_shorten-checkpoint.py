import json
import random
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# File paths
train_output_file = 'data/train.jsonl'
test_output_file = 'data/test.jsonl'
shortened_train_file = 'data/shortened_train.jsonl'
shortened_test_file = 'data/shortened_test.jsonl'

# Load the data
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Save data to jsonl
def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Function to split text considering " " and "· "
def split_text(text, parts):
    approx_part_length = len(text) // parts
    splits = []
    last_split = 0

    for _ in range(parts - 1):
        split_point = last_split + approx_part_length
        # Try to find the best splitting point near the middle
        space_idx = text.rfind(' ', last_split, split_point + 50)
        dot_idx = text.rfind('· ', last_split, split_point + 50)

        # Choose the best split point, preferring "· " if it's within range
        if dot_idx != -1 and abs(dot_idx - split_point) <= 50:
            best_split = dot_idx + 2
        elif space_idx != -1:
            best_split = space_idx + 1
        else:
            best_split = split_point

        splits.append(text[last_split:best_split])
        last_split = best_split

    splits.append(text[last_split:])  # Add the remaining text
    return splits

# Function to process and shorten entries in a file
def process_entries(data):
    max_tokens_per_part = 699
    processed_data = []

    for entry in data:
        edition_with_brackets = entry.get('Edition_with_brackets', '')
        edition_without_brackets = entry.get('Edition_without_brackets', '')

        tokens_with_brackets = len(tokenizer.tokenize(edition_with_brackets))

        # Calculate the number of parts based on a maximum of 699 tokens per part
        parts = max(1, (tokens_with_brackets + max_tokens_per_part - 1) // max_tokens_per_part)  # Rounding up

        if parts > 1:
            splits_with_brackets = split_text(edition_with_brackets, parts)
            splits_without_brackets = split_text(edition_without_brackets, parts)

            # Create new entries for the splits
            for i in range(parts):
                new_entry = entry.copy()
                new_entry['Edition_with_brackets'] = splits_with_brackets[i]
                new_entry['Edition_without_brackets'] = splits_without_brackets[i]
                processed_data.append(new_entry)
        else:
            processed_data.append(entry)

    return processed_data

# Process and save shortened train and test data
train_data = load_jsonl(train_output_file)
test_data = load_jsonl(test_output_file)

shortened_train_data = process_entries(train_data)
shortened_test_data = process_entries(test_data)

save_jsonl(shortened_train_data, shortened_train_file)
save_jsonl(shortened_test_data, shortened_test_file)

print("Processing complete! Files saved as 'shortened_train.jsonl' and 'shortened_test.jsonl'.")
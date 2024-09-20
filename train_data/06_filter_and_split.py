import json
import random
import re

def clean_edition_text(text):
    text = text.replace("⟨⟩", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

input_file = 'data/normalized_united.jsonl'

train_output_file = 'data/pap_train.jsonl'
test_output_file = 'data/pap_test.jsonl'

filtered_entries = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        edition_with_brackets = clean_edition_text(entry.get('Edition_with_brackets', ''))
        edition_without_brackets = clean_edition_text(entry.get('Edition_without_brackets', ''))

        if len(edition_with_brackets) >= 30 or len(edition_without_brackets) >= 30:
            entry['Edition_with_brackets'] = edition_with_brackets
            entry['Edition_without_brackets'] = edition_without_brackets
            filtered_entries.append(entry)

random.shuffle(filtered_entries)

split_index = int(0.95 * len(filtered_entries))

train_entries = filtered_entries[:split_index]
test_entries = filtered_entries[split_index:]

with open(train_output_file, 'w', encoding='utf-8') as train_file:
    for entry in train_entries:
        train_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

with open(test_output_file, 'w', encoding='utf-8') as test_file:
    for entry in test_entries:
        test_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Train and test files created successfully!")
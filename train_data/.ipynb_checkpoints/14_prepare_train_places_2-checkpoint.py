import os
import json
import re
import random

# Create the output directory if it doesn't exist
output_dir = 'data/data_prefinal/round_2'
os.makedirs(output_dir, exist_ok=True)

# Helper function to strip "⟨" and "⟩" from the text
def strip_brackets(text):
    return text.replace("⟨", "").replace("⟩", "")

# Helper function to rearrange sentences randomly
def rearrange_sentences(text):
    sentences = re.split(r'(?<=·)|$', text)  # Splitting on "·" or the end of string
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty and whitespace-only strings
    random.shuffle(sentences)  # Randomly shuffle the sentences
    return ' '.join(sentences)

# Helper function to replace characters with '-'
def replace_characters(text, percent):
    text_length = len(text)
    num_replacements = int(text_length * (percent / 100))
    indices = random.sample(range(text_length), num_replacements)
    text_list = list(text)
    for index in indices:
        text_list[index] = '-'
    return ''.join(text_list).replace('- -', '--')

def process_entry(entry):
    results = []

    place = entry.get("place", "")
    # Skip 'null' or empty entries
    if not place or place.lower() == "null":
        return results

    # Process the 'place' field
    place_processed = re.sub(r'^(.+?)(\(.+?$)', r'\1', place)

    # Process the 'Edition_with_brackets' and 'Edition_without_brackets' fields
    edition_with_brackets = strip_brackets(entry.get("Edition_with_brackets", ""))
    edition_without_brackets = strip_brackets(entry.get("Edition_without_brackets", ""))

    # Produce rearranged versions
    rearranged_with_brackets = rearrange_sentences(edition_with_brackets)
    rearranged_without_brackets = rearrange_sentences(edition_without_brackets)

    # Define the percentages for replacement
    percentages = [0, 5, 15, 25]

    # Create versions with characters replaced by '-'
    for percent in percentages:
        replaced_with_brackets = replace_characters(edition_with_brackets, percent)
        replaced_rearranged_with_brackets = replace_characters(rearranged_with_brackets, percent)
        replaced_without_brackets = replace_characters(edition_without_brackets, percent)
        replaced_rearranged_without_brackets = replace_characters(rearranged_without_brackets, percent)

        # Construct result structure
        results.append({
            "messages": [
                {"role": "system", "content": "Assign this papyrus fragment to an exact place!"},
                {"role": "user", "content": replaced_with_brackets},
                {"role": "assistant", "content": place_processed}
            ]
        })
        results.append({
            "messages": [
                {"role": "system", "content": "Assign this papyrus fragment to an exact place!"},
                {"role": "user", "content": replaced_rearranged_with_brackets},
                {"role": "assistant", "content": place_processed}
            ]
        })
        results.append({
            "messages": [
                {"role": "system", "content": "Assign this papyrus fragment to an exact place!"},
                {"role": "user", "content": replaced_without_brackets},
                {"role": "assistant", "content": place_processed}
            ]
        })
        results.append({
            "messages": [
                {"role": "system", "content": "Assign this papyrus fragment to an exact place!"},
                {"role": "user", "content": replaced_rearranged_without_brackets},
                {"role": "assistant", "content": place_processed}
            ]
        })

    return results

# Repeat the process 10 times
for round_number in range(1, 11):
    output_entries = []

    # Read all entries from the JSONL file
    with open('data/shortened_train.jsonl', 'r', encoding='utf-8') as file:
        entries = [json.loads(line.strip()) for line in file]

    # Process each entry
    for entry in entries:
        output_entries.extend(process_entry(entry))

    # Save the processed entries to a JSONL file
    output_file_path = os.path.join(output_dir, f'DDbDP_train_places_{round_number}.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in output_entries:
            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
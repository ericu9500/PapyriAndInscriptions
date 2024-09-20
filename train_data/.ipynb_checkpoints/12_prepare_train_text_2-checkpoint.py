import os
import re
import json
import random
import numpy as np

# Create the output directory if it doesn't exist
output_dir = 'data/data_prefinal/round_2'
os.makedirs(output_dir, exist_ok=True)

# Load all entries from shortened_train.jsonl
with open('data/shortened_train.jsonl', 'r', encoding='utf-8') as file:
    entries = [json.loads(line) for line in file]

# Function to segment the text
def segment_text(text):
    segments = []
    
    # Replace "…" with "----------"
    text = text.replace("…", "----------")

    # Define the regex patterns based on the rules given
    numeral_pattern = r"⟨[^-…⟩]+⟩(?:· | )?"  # Numerals within brackets, including spaces or "· " attached
    lost_letters_pattern = r"-·? |----------·? "  # Lost letters "- ", "-· ", "---------- ", "----------· "
    preserved_letters_pattern = r"(?<![⟨⟩])[^-…⟨⟩](?![⟨⟩])(?:· | )?"  # Preserved letters, with attached spaces or "· "

    # Combine all patterns
    combined_pattern = f"({numeral_pattern})|({lost_letters_pattern})|({preserved_letters_pattern})"

    # Find all matches and add them to the segments list
    for match in re.findall(combined_pattern, text):
        segments.append("".join(match))

    return segments

# Function to identify and scramble sentences
def scramble_sentences(text):
    # Find all sentences ending with "·" or other ending characters
    sentences = re.findall(r"(.+?·|.+?[^·]$)", text, re.MULTILINE)

    # Ensure that there are sentences to scramble
    if not sentences:
        return text  # Return the original text if no sentences are found

    # Scramble the order of sentences randomly
    random.shuffle(sentences)

    # Reassemble the text from scrambled sentences
    scrambled_text = ''.join(sentences)
    
    return scrambled_text

# Function to group and count sequences of tokens
def group_and_count(segments):
    grouped_sequences = []
    current_type = None
    current_group = []
    
    for token in segments:
        if "-" in token or "----------" in token:  # Identify lost tokens
            token_type = "lost"
        else:
            token_type = "preserved"

        # Group tokens by type
        if token_type == current_type:
            current_group.append(token)
        else:
            if current_group:
                # Add the previous group to the result
                grouped_sequences.append((len(current_group), current_type, current_group))
            # Start a new group
            current_type = token_type
            current_group = [token]

    # Add the final group
    if current_group:
        grouped_sequences.append((len(current_group), current_type, current_group))
    
    return grouped_sequences

# Function to apply statistically biased masking
def mask_preserved_tokens(grouped_sequences):
    # Initialize masked_segment in case no masking occurs
    masked_segment = []

    # Get all preserved sequences that are at least 3 tokens long
    preserved_sections = [(i, seq) for i, seq in enumerate(grouped_sequences) if seq[1] == "preserved" and seq[0] >= 3]
    
    # If no eligible sections exist, return without masking
    if not preserved_sections:
        return grouped_sequences, masked_segment
    
    # Calculate weights based on the length of each preserved section (favoring longer sections)
    lengths = [seq[0] for _, seq in preserved_sections]
    weights = np.array(lengths) ** 2  # Square the lengths to heavily favor longer sections

    # Randomly choose a section based on the calculated weights
    chosen_index = random.choices(range(len(preserved_sections)), weights=weights, k=1)[0]
    chosen_section_index, chosen_section = preserved_sections[chosen_index]
    
    total_preserved_tokens = chosen_section[0]
    
    # Decide how many tokens to mask (between 3 and up to 50% of the chosen section's length)
    max_maskable_tokens = min(20, total_preserved_tokens // 2)
    if max_maskable_tokens >= 3:
        mask_count = random.randint(3, max_maskable_tokens)
        
        # Determine the starting point for masking
        start_index = random.randint(0, total_preserved_tokens - mask_count)
        
        # Extract the masked portion for later revealing
        masked_segment = chosen_section[2][start_index:start_index + mask_count]
        
        # Replace the selected contiguous tokens with "X"
        for i in range(start_index, start_index + mask_count):
            chosen_section[2][i] = 'X'
    
    return grouped_sequences, masked_segment

# Function to randomly replace a percentage of valid characters with "-"
def replace_percentage_with_dash(text, percentage):
    characters = list(text)
    indices_to_replace = [i for i, c in enumerate(characters) if c not in ['-', ' ', '…', '·']]
    
    # Exclude '[X letters missing]' parts from being replaced
    x_indices = []
    for match in re.finditer(r'\[\d+ letters missing\]', text):
        x_indices.extend(range(match.start(), match.end()))

    indices_to_replace = [i for i in indices_to_replace if i not in x_indices]
    
    # Calculate the number of characters to replace based on the percentage
    num_to_replace = int(len(indices_to_replace) * (percentage / 100))
    chosen_indices = random.sample(indices_to_replace, num_to_replace)

    # Replace chosen characters with "-"
    for i in chosen_indices:
        characters[i] = '-'
    
    return ''.join(characters)

# Function to reassemble tokens into text and insert the "[X tokens missing]" message
def reassemble_text_with_placeholder(grouped_sequences):
    reassembled_text = ""
    placeholder_inserted = False

    for count, token_type, tokens in grouped_sequences:
        for token in tokens:
            if token == "X" and not placeholder_inserted:
                # Count the number of Xs in this segment
                masked_count = tokens.count("X")
                reassembled_text += f"[{masked_count} letters missing]"
                placeholder_inserted = True
            elif token != "X":
                reassembled_text += token

    return reassembled_text

# Function to count valid characters
def count_valid_characters(text):
    # Remove invalid characters ("…", " ", "·", "-")
    clean_text = re.sub(r"[ …·-]", "", text)
    return len(clean_text)

# Function for final post-processing stage
def post_process_text(text):
    # Replace occurrences of "- -" with "--"
    return re.sub(r"-\s-", "--", text)

# Function to process the entry with different masking percentages and varying masks
def process_entry(edition_text, scramble=False):
    results = []

    # Scramble sentences if required
    text_to_process = scramble_sentences(edition_text) if scramble else edition_text

    # Ensure the scrambled text is not empty
    if not text_to_process.strip():
        return results  # Return empty if the text is empty after scrambling

    # Process with varying masking percentages
    for percent in [0, 5, 10, 15, 20, 25]:
        # Re-segment and re-group the text for each variation
        segmented_text = segment_text(text_to_process)
        grouped_sequences = group_and_count(segmented_text)
        
        # Apply a new masking each time to ensure variation
        masked_sequences, masked_segment = mask_preserved_tokens(grouped_sequences)
        final_text_with_placeholder = reassemble_text_with_placeholder(masked_sequences)

        # Post-Processing Steps
        final_text_with_placeholder = re.sub(r"[⟨⟩]", "", final_text_with_placeholder)  # 1. Remove brackets
        revealed_masked_string = "".join(masked_segment)
        revealed_masked_string = re.sub(r"⟨[^⟩]+?⟩", "0", revealed_masked_string)  # 2. Replace inner content with "0"
        
        # Replace a percentage of valid characters with "-"
        masked_text = replace_percentage_with_dash(final_text_with_placeholder, percent)
        masked_text = post_process_text(masked_text)
        
        results.append({
            "messages": [
                {"role": "system", "content": "Fill in the missing letters in this papyrus fragment!"},
                {"role": "user", "content": masked_text},
                {"role": "assistant", "content": revealed_masked_string}
            ]
        })
    
    return results

# Repeat the process 10 times
for round_number in range(1, 11):
    output_entries = []
    
    # Process each entry in the JSONL file
    for entry in entries:
        for edition_field in ["Edition_with_brackets", "Edition_without_brackets"]:
            if edition_field in entry:
                edition_text = entry[edition_field]
                
                # Filter out entries with fewer than 50 valid characters
                if count_valid_characters(edition_text) >= 50:
                    # Process without scrambling
                    output_entries.extend(process_entry(edition_text, scramble=False))
                    
                    # Process with scrambling
                    output_entries.extend(process_entry(edition_text, scramble=True))
                    break  # Exit after processing one entry 

    # Save the processed entries to a JSONL file
    output_file_path = os.path.join(output_dir, f'DDbDP_train_eds_{round_number}.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in output_entries:
            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
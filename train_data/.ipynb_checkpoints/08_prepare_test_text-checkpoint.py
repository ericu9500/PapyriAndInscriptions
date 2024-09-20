import re
import json
import random
import numpy as np
import os

# Load all entries from shortened_test.jsonl
with open('data/shortened_test.jsonl', 'r', encoding='utf-8') as file:
    entries = [json.loads(line) for line in file]

# Function to segment the text
def segment_text(text):
    segments = []
    
    # Define the regex patterns based on the rules given
    numeral_pattern = r"⟨[^-…⟩]+⟩(?:· | )?"  # Numerals within brackets, including spaces or "· " attached
    lost_letters_pattern = r"-·? |…·? "  # Lost letters "- ", "-· ", "… ", "…· "
    preserved_letters_pattern = r"(?<![⟨⟩])[^-…⟨⟩](?![⟨⟩])(?:· | )?"  # Preserved letters, with attached spaces or "· "

    # Combine all patterns
    combined_pattern = f"({numeral_pattern})|({lost_letters_pattern})|({preserved_letters_pattern})"

    # Find all matches and add them to the segments list
    for match in re.findall(combined_pattern, text):
        segments.append("".join(match))

    return segments

# Function to group and count sequences of tokens
def group_and_count(segments):
    grouped_sequences = []
    current_type = None
    current_group = []
    
    for token in segments:
        if "-" in token or "…" in token:  # Identify lost tokens
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

# Process each entry and generate output
output_entries = []

for entry in entries:
    for edition_field in ["Edition_with_brackets", "Edition_without_brackets"]:
        if edition_field in entry:
            edition_text = entry[edition_field]
            
            # Segment the text
            segmented_text = segment_text(edition_text)
            
            # Group and count the sequences
            grouped_sequences = group_and_count(segmented_text)
            
            # Apply statistically biased masking and get the masked segment
            masked_sequences, masked_segment = mask_preserved_tokens(grouped_sequences)
            
            # Reassemble the text with placeholders for the masked tokens
            final_text_with_placeholder = reassemble_text_with_placeholder(masked_sequences)
            
            # Reassemble the masked portion for revealing
            revealed_masked_string = "".join(masked_segment)
            
            # --- Post-Processing Steps ---
            # 1. Replace regex [⟨⟩] with "" in final_text_with_placeholder
            final_text_with_placeholder = re.sub(r"[⟨⟩]", "", final_text_with_placeholder)
            
            # 2. Replace regex "⟨[^⟩]+?⟩" with "0" in revealed_masked_string
            revealed_masked_string = re.sub(r"⟨[^⟩]+?⟩", "0", revealed_masked_string)
            
            # Create the JSON object for output in the specified structure
            output_entry = {
                "messages": [
                    {"role": "system", "content": "Fill in the missing letters in this papyrus fragment!"},
                    {"role": "user", "content": final_text_with_placeholder},
                    {"role": "assistant", "content": revealed_masked_string}
                ]
            }
            output_entries.append(output_entry)

# Ensure the output directory exists
output_dir = 'data/data_prefinal/'
os.makedirs(output_dir, exist_ok=True)

# Write the output to a JSONL file
output_file_path = os.path.join(output_dir, 'DDbDPtest_eds.jsonl')
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for output_entry in output_entries:
        json.dump(output_entry, output_file, ensure_ascii=False)  # Add ensure_ascii=False here
        output_file.write("\n")

print(f"Processing complete! Results saved to {output_file_path}.")
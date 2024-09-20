import re
import json
import random
import numpy as np
from tqdm import tqdm 
import os

with open('data/shortened_pap_test.jsonl', 'r', encoding='utf-8') as file:
    entries = [json.loads(line) for line in file]

def segment_text(text):
    segments = []
    
    numeral_pattern = r"⟨[^-…⟩]+⟩(?:· | )?"   
    lost_letters_pattern = r"-{2,}|-·? |…·? " 
    preserved_letters_pattern = r"(?<![⟨⟩])[^-…⟨⟩](?![⟨⟩])(?:· | )?"

    combined_pattern = f"({numeral_pattern})|({lost_letters_pattern})|({preserved_letters_pattern})"

    for match in re.findall(combined_pattern, text):
        segments.append("".join(match))

    return segments

def group_and_count(segments):
    grouped_sequences = []
    current_type = None
    current_group = []
    
    for token in segments:
        if "-" in token or "…" in token:  
            token_type = "lost"
        else:
            token_type = "preserved"

        if token_type == current_type:
            current_group.append(token)
        else:
            if current_group:
                grouped_sequences.append((len(current_group), current_type, current_group))
            current_type = token_type
            current_group = [token]

    if current_group:
        grouped_sequences.append((len(current_group), current_type, current_group))
    
    return grouped_sequences

def mask_preserved_tokens(grouped_sequences):
    masked_segment = []
    preserved_sections = [(i, seq) for i, seq in enumerate(grouped_sequences) if seq[1] == "preserved" and seq[0] >= 3]
    
    if not preserved_sections:
        return grouped_sequences, masked_segment
    
    lengths = [seq[0] for _, seq in preserved_sections]
    weights = np.array(lengths) ** 2
    chosen_index = random.choices(range(len(preserved_sections)), weights=weights, k=1)[0]
    chosen_section_index, chosen_section = preserved_sections[chosen_index]
    
    total_preserved_tokens = chosen_section[0]
    
    max_characters_to_mask = 10
    mask_count = random.randint(1, min(total_preserved_tokens, max_characters_to_mask))
    
    start_index = random.randint(0, total_preserved_tokens - mask_count)
    
    masked_segment = chosen_section[2][start_index:start_index + mask_count]
    
    masked_length = sum(len(token) for token in masked_segment)
    while masked_length > max_characters_to_mask:
        mask_count -= 1
        masked_segment = chosen_section[2][start_index:start_index + mask_count]
        masked_length = sum(len(token) for token in masked_segment)
    
    for i in range(start_index, start_index + mask_count):
        chosen_section[2][i] = 'X'
    
    return grouped_sequences, masked_segment

def reassemble_text_with_placeholder(grouped_sequences):
    reassembled_text = ""
    placeholder_inserted = False

    for count, token_type, tokens in grouped_sequences:
        for token in tokens:
            if token == "X" and not placeholder_inserted:
                masked_count = sum(1 for t in tokens if t == "X" and t not in [" ", "·"])
                reassembled_text += f"[{masked_count} letters missing]"
                placeholder_inserted = True
            elif token != "X":
                reassembled_text += token

    return reassembled_text
    
output_dir = 'data/test'
os.makedirs(output_dir, exist_ok=True)

# Output file path
output_file_path = 'data/test/test_pap_text.jsonl'

# Open output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    count = 0  

    for entry in tqdm(entries, desc="Processing entries", unit="entry"):
        if "Edition_with_brackets" in entry:
            edition_text = entry["Edition_with_brackets"]
            
            # Replace "…" with "----------"
            edition_text = edition_text.replace("…", "----------")
            
            if len(re.findall(r"[^\-…]", edition_text)) >= 90:
                segmented_text = segment_text(edition_text)
                
                grouped_sequences = group_and_count(segmented_text)
                
                masked_sequences, masked_segment = mask_preserved_tokens(grouped_sequences)
                
                final_text_with_placeholder = reassemble_text_with_placeholder(masked_sequences)
                
                revealed_masked_string = "".join(masked_segment)
                
                final_text_with_placeholder = re.sub(r"[⟨⟩]", "", final_text_with_placeholder)
                final_text_with_placeholder = re.sub(r" \-", "-", final_text_with_placeholder)
                final_text_with_placeholder = re.sub(r"\- ", "-", final_text_with_placeholder)
                revealed_masked_string = re.sub(r"⟨[^⟩]+?⟩", "0", revealed_masked_string)
                
                output_entry = {
                    "messages": [
                        {"role": "system", "content": "Fill in the missing letters in this papyrus fragment!"},
                        {"role": "user", "content": final_text_with_placeholder},
                        {"role": "assistant", "content": revealed_masked_string}
                    ]
                }

                json.dump(output_entry, output_file, ensure_ascii=False)
                output_file.write("\n")

                count += 1
                if count >= 7811:
                    break

print(f"Processing complete! {count} entries saved to {output_file_path}.")
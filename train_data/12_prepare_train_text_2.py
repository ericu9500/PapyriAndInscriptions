import os
import re
import json
import random
import numpy as np


output_dir = 'data/train_round_2'
os.makedirs(output_dir, exist_ok=True)


with open('data/shortened_pap_train.jsonl', 'r', encoding='utf-8') as file:
    entries = [json.loads(line) for line in file]

def segment_text(text):
    segments = []    
    text = text.replace("…", "----------")
    numeral_pattern = r"⟨[^-…⟩]+⟩(?:· | )?"  
    lost_letters_pattern = r"-·? |----------·? "  
    preserved_letters_pattern = r"(?<![⟨⟩])[^-…⟨⟩](?![⟨⟩])(?:· | )?" 
    combined_pattern = f"({numeral_pattern})|({lost_letters_pattern})|({preserved_letters_pattern})"

    
    for match in re.findall(combined_pattern, text):
        segments.append("".join(match))

    return segments


def scramble_sentences(text):
    
    sentences = re.findall(r"(.+?·|.+?[^·]$)", text, re.MULTILINE)

    
    if not sentences:
        return text  

    
    random.shuffle(sentences)

    
    scrambled_text = ''.join(sentences)
    
    return scrambled_text


def group_and_count(segments):
    grouped_sequences = []
    current_type = None
    current_group = []
    
    for token in segments:
        if "-" in token or "----------" in token:  
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
    
    
    max_maskable_tokens = min(20, total_preserved_tokens // 2)
    if max_maskable_tokens >= 1:
        mask_count = random.randint(1, max_maskable_tokens)
        
        
        start_index = random.randint(0, total_preserved_tokens - mask_count)
        
        
        masked_segment = chosen_section[2][start_index:start_index + mask_count]
        
        
        for i in range(start_index, start_index + mask_count):
            chosen_section[2][i] = 'X'
    
    return grouped_sequences, masked_segment


def replace_percentage_with_dash(text, percentage):
    characters = list(text)
    indices_to_replace = [i for i, c in enumerate(characters) if c not in ['-', ' ', '…', '·']]
    
    
    x_indices = []
    for match in re.finditer(r'\[\d+ letters missing\]', text):
        x_indices.extend(range(match.start(), match.end()))

    indices_to_replace = [i for i in indices_to_replace if i not in x_indices]
    
    
    num_to_replace = int(len(indices_to_replace) * (percentage / 100))
    chosen_indices = random.sample(indices_to_replace, num_to_replace)

    
    for i in chosen_indices:
        characters[i] = '-'
    
    return ''.join(characters)


def reassemble_text_with_placeholder(grouped_sequences):
    reassembled_text = ""
    placeholder_inserted = False

    for count, token_type, tokens in grouped_sequences:
        for token in tokens:
            if token == "X" and not placeholder_inserted:
                
                masked_count = tokens.count("X")
                reassembled_text += f"[{masked_count} letters missing]"
                placeholder_inserted = True
            elif token != "X":
                reassembled_text += token

    return reassembled_text


def count_valid_characters(text):
    
    clean_text = re.sub(r"[ …·-]", "", text)
    return len(clean_text)


def post_process_text(text):
    
    return re.sub(r"-\s-", "--", text)


def process_entry(edition_text, scramble=False):
    results = []

    
    text_to_process = scramble_sentences(edition_text) if scramble else edition_text

    
    if not text_to_process.strip():
        return results  

    
    for percent in [0, 5, 10, 15, 20, 25]:
        
        segmented_text = segment_text(text_to_process)
        grouped_sequences = group_and_count(segmented_text)
        
        
        masked_sequences, masked_segment = mask_preserved_tokens(grouped_sequences)
        final_text_with_placeholder = reassemble_text_with_placeholder(masked_sequences)

        
        final_text_with_placeholder = re.sub(r"[⟨⟩]", "", final_text_with_placeholder)  
        revealed_masked_string = "".join(masked_segment)
        revealed_masked_string = re.sub(r"⟨[^⟩]+?⟩", "0", revealed_masked_string)  
        
        
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


for round_number in range(1, 11):
    output_entries = []
    
    
    for entry in entries:
        for edition_field in ["Edition_with_brackets", "Edition_without_brackets"]:
            if edition_field in entry:
                edition_text = entry[edition_field]
                
                
                if count_valid_characters(edition_text) >= 50:
                    
                    output_entries.extend(process_entry(edition_text, scramble=False))
                    
                    
                    output_entries.extend(process_entry(edition_text, scramble=True))
                    break  

    
    output_file_path = os.path.join(output_dir, f'train_pap_text_{round_number}.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in output_entries:
            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
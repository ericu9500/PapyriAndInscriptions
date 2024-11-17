import re
import json
import random
import os

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

input_file = 'synthetic_editions_with_ithaca_text.jsonl'
output_file = os.path.join(output_dir, '1.jsonl')

with open(input_file, 'r', encoding='utf-8') as file:
    entries = [json.loads(line) for line in file]

def segment_text(text):
    segments = []
    
    numeral_pattern = r"⟨[^-…⟩]+⟩(?:· | )?"
    lost_letters_pattern = r"-{2,}|-·? |…·? "
    preserved_letters_pattern = r"(?<![⟨⟩])[^-…⟨⟩](?![⟨⟩])(?:· | )?"

    combined_pattern = f"({numeral_pattern})|({lost_letters_pattern})|({preserved_letters_pattern})"

    for match in re.findall(combined_pattern, text):
        segments.append("".join(filter(None, match)))

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

def mask_preserved_tokens(grouped_sequences, mask_count):
    masked_segment = []
    preserved_sections = [(i, seq) for i, seq in enumerate(grouped_sequences) if seq[1] == "preserved" and seq[0] >= mask_count]
    
    if not preserved_sections:
        return grouped_sequences, masked_segment
    
    chosen_section_index, chosen_section = random.choice(preserved_sections)
    total_preserved_tokens = chosen_section[0]
    
    start_index = random.randint(0, total_preserved_tokens - mask_count)
    masked_segment = chosen_section[2][start_index:start_index + mask_count]
    
    \n    if all(char in [" ", "·"] for char in masked_segment):
        return grouped_sequences, []
    
    for i in range(start_index, start_index + mask_count):
        chosen_section[2][i] = 'X'
    
    return grouped_sequences, masked_segment

def reassemble_text_with_placeholder(grouped_sequences):
    reassembled_text = ""
    placeholder_inserted = False

    for count, token_type, tokens in grouped_sequences:
        for token in tokens:
            if token == "X" and not placeholder_inserted:
                reassembled_text += "[Ö letters missing]"
                placeholder_inserted = True
            elif token != "X":
                reassembled_text += token

    return reassembled_text

def process_entry(edition_text, mask_count):
    if len(edition_text) > 749:
        choice = random.choice(['beginning', 'middle', 'end'])
        if choice == 'beginning':
            edition_text = edition_text[:749]
        elif choice == 'middle':
            start = len(edition_text) // 2 - 374
            edition_text = edition_text[start:start+749]
        elif choice == 'end':
            edition_text = edition_text[-749:]

    edition_text = edition_text.replace("…", "----------")
    edition_text = edition_text.replace(".", "·")

    if len(re.findall(r"[^\-]", edition_text)) >= 50:
        segmented_text = segment_text(edition_text)
        grouped_sequences = group_and_count(segmented_text)
        masked_sequences, masked_segment = mask_preserved_tokens(grouped_sequences, mask_count)
        
        if not masked_segment:
            return None, None
        
        final_text_with_placeholder = reassemble_text_with_placeholder(masked_sequences)

        final_text_with_placeholder = re.sub(r"[⟨⟩]", "", final_text_with_placeholder)
        final_text_with_placeholder = re.sub(r" \-", "-", final_text_with_placeholder)
        final_text_with_placeholder = re.sub(r"\- ", "-", final_text_with_placeholder)

        revealed_masked_string = "".join(masked_segment).replace("·", "")

        return final_text_with_placeholder, revealed_masked_string
    else:
        return None, None


with open(output_file, 'w', encoding='utf-8') as output:
    for entry in entries:
        phi_id = str(entry.get('PHI_ID', ''))

        if phi_id and phi_id[-1] in '01256789':
            random_field = random.choice(['without_diacritics', 'synthetic', 'synthetic_2', 'ithaca_text'])
            edition_text = entry.get(random_field, "")

            for mask_count in range(1, 21):
                for _ in range(40): 
                    final_text_with_placeholder, revealed_masked_string = process_entry(edition_text, mask_count)
                    if final_text_with_placeholder and revealed_masked_string:
                        missing_letter_count = len(re.sub(r'[ ·]', '', revealed_masked_string)) 
                        final_text_with_placeholder = final_text_with_placeholder.replace("Ö", str(missing_letter_count))

                        messages = {
                            "messages": [
                                {"role": "system", "content": "Fill in the missing characters in this inscription!"},
                                {"role": "user", "content": final_text_with_placeholder},
                                {"role": "assistant", "content": revealed_masked_string}
                            ]
                        }
                        output.write(json.dumps(messages, ensure_ascii=False) + '\n')
                        break

print(f"Masked data saved to {output_file}")
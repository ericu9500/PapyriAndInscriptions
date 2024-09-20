import os
import json
import re
import random


output_dir = 'data/train_round_2'
os.makedirs(output_dir, exist_ok=True)



def strip_brackets(text):
    return text.replace("⟨", "").replace("⟩", "")

def rearrange_sentences(text):
    sentences = re.split(r'(?<=·)|$', text)  
    sentences = [s.strip() for s in sentences if s.strip()]  
    random.shuffle(sentences)  
    return ' '.join(sentences)

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
    if not place or place.lower() == "null":
        return results

    
    place_processed = re.sub(r'^(.+?)(\(.+?$)', r'\1', place)

    
    edition_with_brackets = strip_brackets(entry.get("Edition_with_brackets", ""))
    edition_without_brackets = strip_brackets(entry.get("Edition_without_brackets", ""))

    
    rearranged_with_brackets = rearrange_sentences(edition_with_brackets)
    rearranged_without_brackets = rearrange_sentences(edition_without_brackets)

    
    percentages = [0, 5, 15, 25]

    
    for percent in percentages:
        replaced_with_brackets = replace_characters(edition_with_brackets, percent)
        replaced_rearranged_with_brackets = replace_characters(rearranged_with_brackets, percent)
        replaced_without_brackets = replace_characters(edition_without_brackets, percent)
        replaced_rearranged_without_brackets = replace_characters(rearranged_without_brackets, percent)

        
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


for round_number in range(1, 11):
    output_entries = []

    
    with open('data/shortened_pap_train.jsonl', 'r', encoding='utf-8') as file:
        entries = [json.loads(line.strip()) for line in file]

    
    for entry in entries:
        output_entries.extend(process_entry(entry))

    
    output_file_path = os.path.join(output_dir, f'DDbDP_train_places_{round_number}.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in output_entries:
            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
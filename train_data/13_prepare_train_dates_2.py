import os
import re
import json
import random


output_dir = 'data/round_2'
os.makedirs(output_dir, exist_ok=True)

def process_date(date):
    
    if re.match(r'^-\d+±\d+$', date):
        return int(date.split('±')[0])  
    elif re.match(r'^\d+±\d+$', date):
        return int(date.split('±')[0])  
    elif re.match(r'^-\d+\+$', date):
        return int(date[:-1]) + 25  
    elif re.match(r'^-\d+-$', date):
        return int(date[:-1]) - 25  
    elif re.match(r'^\d+\+$', date):
        return int(date[:-1]) + 25  
    elif re.match(r'^\d+-$', date):
        return int(date[:-1]) - 25  
    elif re.match(r'^-\d+$', date):
        return int(date)  
    elif date == 'null':
        return None  
    else:
        return date  

def strip_brackets(text):
    
    return text.replace('⟨', '').replace('⟩', '')

def rearrange_sentences(text):
    
    sentences = re.findall(r'(.+?·|.+?$)', text)
    random.shuffle(sentences)
    return ''.join(sentences)

def replace_characters(text, percentage):
    
    num_chars_to_replace = int(len(text) * percentage)
    text_list = list(text)
    indices = random.sample(range(len(text)), num_chars_to_replace)
    for idx in indices:
        if text_list[idx] != ' ':
            text_list[idx] = '-'
    return ''.join(text_list).replace('- -', '--')  

def process_entry(entry):
    results = []

    date = entry.get("date", "")
    if date == 'null':
        return results  

    processed_date = process_date(date)
    edition_with_brackets = strip_brackets(entry.get("Edition_with_brackets", "Not available"))
    edition_without_brackets = strip_brackets(entry.get("Edition_without_brackets", "Not available"))

    
    editions = {
        'Original': (edition_with_brackets, edition_without_brackets),
        'Rearranged': (rearrange_sentences(edition_with_brackets), rearrange_sentences(edition_without_brackets))
    }

    
    percentages = [0.0, 0.05, 0.15, 0.25]
    for version_name, (version_with, version_without) in editions.items():
        for percentage in percentages:
            replaced_with = replace_characters(version_with, percentage)
            replaced_without = replace_characters(version_without, percentage)

            
            results.append({
                "messages": [
                    {"role": "system", "content": "Date this papyrus fragment to an exact year!"},
                    {"role": "user", "content": replaced_with},
                    {"role": "assistant", "content": str(processed_date)}
                ]
            })
            results.append({
                "messages": [
                    {"role": "system", "content": "Date this papyrus fragment to an exact year!"},
                    {"role": "user", "content": replaced_without},
                    {"role": "assistant", "content": str(processed_date)}
                ]
            })

    return results


for round_number in range(1, 11):
    output_entries = []

    
    with open('data/shortened_pap_train.jsonl', 'r', encoding='utf-8') as f:
        entries = [json.loads(line.strip()) for line in f]

    
    for entry in entries:
        output_entries.extend(process_entry(entry))

    
    output_file_path = os.path.join(output_dir, f'DDbDP_train_dates_{round_number}.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in output_entries:
            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
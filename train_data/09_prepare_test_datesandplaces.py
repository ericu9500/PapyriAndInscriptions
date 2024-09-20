import json
import re
import os

def clean_edition_text(text):
    text = text.replace("⟨", "").replace("⟩", "")
    text = re.sub(r"\s+", " ", text)  
    text = re.sub(r"…", "----------", text)
    text = re.sub(r"\- ", "-", text) 
    text = re.sub(r" \-", "-", text)
    text = re.sub(r" \·", "· ", text) 
    text = re.sub(r"\s+", " ", text) 

    return text.strip()

def process_entries(input_file, output_file_dates, output_file_places):
    with open(input_file, 'r', encoding='utf-8') as file:
        entries = [json.loads(line) for line in file]
    
    entries = [entry for entry in entries if len(re.sub(r'-', '', entry.get("Edition_with_brackets", ""))) >= 90]

    with open(output_file_dates, 'w', encoding='utf-8') as file_dates, open(output_file_places, 'w', encoding='utf-8') as file_places:
        for entry in entries:
            edition_with_brackets = clean_edition_text(entry.get("Edition_with_brackets", ""))

            real_date = entry.get("date", "")
            real_date = real_date.strip() if real_date else None
            
            real_place = entry.get("place", "")
            real_place = real_place.strip() if real_place else None

            if real_date:
                output_entry = {
                    "messages": [
                        {"role": "system", "content": "Date this inscription to an exact year!"},
                        {"role": "user", "content": edition_with_brackets},
                        {"role": "assistant", "content": real_date}
                    ]
                }
                file_dates.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

            if real_place:
                output_entry = {
                    "messages": [
                        {"role": "system", "content": "Assign this inscription to an exact place!"},
                        {"role": "user", "content": edition_with_brackets},
                        {"role": "assistant", "content": real_place}
                    ]
                }
                file_places.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

input_file = 'shortened_pap_test.jsonl'
output_dir = 'data/test'
os.makedirs(output_dir, exist_ok=True)
output_file_dates = 'data/test/test_pap_dates.jsonl'
output_file_places = 'data/test/test_pap_places.jsonl'

process_entries(input_file, output_file_dates, output_file_places)
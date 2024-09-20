import json
import re
import os  

def clean_edition_text(text):
    
    text = text.replace("⟨", "").replace("⟩", "")
    text = re.sub(r"\s+", " ", text)  
    return text.strip()  

def process_train_file(input_file, output_file_dates, output_file_places):
    date_entries = []
    place_entries = []
    
    
    with open(input_file, 'r', encoding='utf-8') as file:
        entries = [json.loads(line) for line in file]
    
    for entry in entries:
        
        edition_with_brackets = clean_edition_text(entry.get("Edition_with_brackets", ""))
        edition_without_brackets = clean_edition_text(entry.get("Edition_without_brackets", ""))
        real_date = entry.get("date", "").strip()
        real_place = entry.get("place", "").strip()

        
        if not real_date or real_date.lower() == "null":
            real_date = None
        if not real_place or real_place.lower() == "null":
            real_place = None

        
        if real_date:
            date_entries.append({
                "messages": [
                    {"role": "system", "content": "Date this papyrus fragment to an exact year!"},
                    {"role": "user", "content": edition_with_brackets},
                    {"role": "assistant", "content": real_date}
                ]
            })
            date_entries.append({
                "messages": [
                    {"role": "system", "content": "Date this papyrus fragment to an exact year!"},
                    {"role": "user", "content": edition_without_brackets},
                    {"role": "assistant", "content": real_date}
                ]
            })

        
        if real_place:
            place_entries.append({
                "messages": [
                    {"role": "system", "content": "Assign this papyrus fragment to an exact place!"},
                    {"role": "user", "content": edition_with_brackets},
                    {"role": "assistant", "content": real_place}
                ]
            })
            place_entries.append({
                "messages": [
                    {"role": "system", "content": "Assign this papyrus fragment to an exact place!"},
                    {"role": "user", "content": edition_without_brackets},
                    {"role": "assistant", "content": real_place}
                ]
            })

    
    output_dir_dates = os.path.dirname(output_file_dates)
    output_dir_places = os.path.dirname(output_file_places)
    os.makedirs(output_dir_dates, exist_ok=True)
    os.makedirs(output_dir_places, exist_ok=True)

    
    with open(output_file_dates, 'w', encoding='utf-8') as file:
        for entry in date_entries:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")

    
    with open(output_file_places, 'w', encoding='utf-8') as file:
        for entry in place_entries:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")


process_train_file('data/shortened_pap_train.jsonl', 'data/train_round_1/train_pap_dates.jsonl', 'data/train_round_1/train_pap_places.jsonl')

print("Processing complete! Results saved to train_pap_dates.jsonl and train_pap_places.jsonl.")
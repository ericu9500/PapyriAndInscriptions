import json
import re
import os  # Import the os module to handle directory creation

def clean_edition_text(text):
    # Remove "⟨" and "⟩" and replace multiple spaces with a single space
    text = text.replace("⟨", "").replace("⟩", "")
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()  # Strip any leading or trailing spaces

def process_file(input_file, output_file):
    output_entries = []
    
    # Load entries from the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as file:
        entries = [json.loads(line) for line in file]
    
    for entry in entries:
        # Get the raw editions
        edition_with_brackets = clean_edition_text(entry.get("Edition_with_brackets", ""))
        edition_without_brackets = clean_edition_text(entry.get("Edition_without_brackets", ""))
        real_date = entry.get("date", "").strip()
        real_place = entry.get("place", "").strip()

        # Skip entries where the date or place is empty or "null"
        if not real_date or real_date.lower() == "null":
            real_date = None
        if not real_place or real_place.lower() == "null":
            real_place = None

        # Only create entries if real_date and real_place are valid
        if real_date:
            output_entries.append({
                "messages": [
                    {"role": "system", "content": "Date this papyrus fragment!"},
                    {"role": "user", "content": edition_with_brackets},
                    {"role": "assistant", "content": real_date}
                ]
            })
            output_entries.append({
                "messages": [
                    {"role": "system", "content": "Date this papyrus fragment!"},
                    {"role": "user", "content": edition_without_brackets},
                    {"role": "assistant", "content": real_date}
                ]
            })

        if real_place:
            output_entries.append({
                "messages": [
                    {"role": "system", "content": "Assign this papyrus fragment to a place!"},
                    {"role": "user", "content": edition_with_brackets},
                    {"role": "assistant", "content": real_place}
                ]
            })
            output_entries.append({
                "messages": [
                    {"role": "system", "content": "Assign this papyrus fragment to a place!"},
                    {"role": "user", "content": edition_without_brackets},
                    {"role": "assistant", "content": real_place}
                ]
            })

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Write the output entries to the specified output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for output_entry in output_entries:
            json.dump(output_entry, file, ensure_ascii=False)
            file.write("\n")

# Process train and test files separately
process_file('data/shortened_test.jsonl', 'data/data_prefinal/DDbDPtest_dates.jsonl')

print("Processing complete! Results saved to train_dates_and_places.jsonl and test_dates_and_places.jsonl.")
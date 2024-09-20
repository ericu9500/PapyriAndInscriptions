import json

# File paths
places_and_dates_file = 'data/Places_and_dates.tsv'
output_with_brackets_file = 'data/output.jsonl'
output_without_brackets_file = 'data/output_without_brackets.jsonl'
united_output_file = 'data/united.jsonl'

# Load data from Places_and_dates.tsv
places_and_dates = {}
with open(places_and_dates_file, 'r', encoding='utf-8') as file:
    for line in file:
        tm_number, place, date = line.strip().split('\t')
        places_and_dates[tm_number] = {
            'place': place,
            'date': date
        }

# Load data from output.jsonl (with brackets)
output_with_brackets = {}
with open(output_with_brackets_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        output_with_brackets[entry['TM_Number']] = entry['Edition_with_brackets']

# Load data from output_without_brackets.jsonl (without brackets)
output_without_brackets = {}
with open(output_without_brackets_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        output_without_brackets[entry['TM_Number']] = entry['Edition_without_brackets']

# Combine data and save to united.jsonl
with open(united_output_file, 'w', encoding='utf-8') as file:
    for tm_number in places_and_dates:
        if tm_number in output_with_brackets and tm_number in output_without_brackets:
            combined_entry = {
                'TM_Number': tm_number,
                'place': places_and_dates[tm_number]['place'],
                'date': places_and_dates[tm_number]['date'],
                'Edition_with_brackets': output_with_brackets[tm_number],
                'Edition_without_brackets': output_without_brackets[tm_number]
            }
            file.write(json.dumps(combined_entry, ensure_ascii=False) + '\n')

print(f"Data combined and saved to {united_output_file}")
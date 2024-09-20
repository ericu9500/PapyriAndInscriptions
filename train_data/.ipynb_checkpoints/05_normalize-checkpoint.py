import json
import re

# The replacement map provided
replacement_map = {
    'ἋΆΑάαἀἁἂἃἄἅἆἈἉἊἌἍἎἏὰᾁᾈᾲᾳᾴᾶᾷᾼ': 'α',
    'Ββ': 'β',
    'Γγ': 'γ',
    'Δδ∆': 'δ',
    'ΈΕὲέέέεἐἑἒἓἔἕἘἙἛἜἝ\u1F73': 'ε',
    'Ζζ': 'ζ',
    'ΗήηἠἡἢἣἤἥἦἧἨἩἫἬἭἮὴᾐᾑᾒᾓᾔᾕᾖᾗῂῃῄῆῇ': 'η',
    'Θθ': 'θ',
    'ΙΊΐίιϊἰἱἲἳἴἵἶἷἸἹἼἽἾὶῑῒῖῗ': 'ι',
    'Κκ': 'κ',
    'Λλ': 'λ',
    'Μμ': 'μ',
    'Νν': 'ν',
    'Ξξ': 'ξ',
    'ΟοόὀὁὂὃὄὅὈὉὊὋὌὍὸόό\u1F79': 'ο',
    'Ππ': 'π',
    'ΡρῤῥῬ': 'ρ',
    'Σςσ': 'σ',
    'Ττ': 'τ',
    'ΥΰυϋύὐὑὓὔὕὖὗὙὝὺῢῦῧ': 'υ',
    'Φφ': 'φ',
    'Χχ': 'χ',
    'Ψψ': 'ψ',
    'ΩΏώῲῳῴῶῷωώὠὡὢὣὤὥὦὧὨὩὪὫὬὭὮὯὼώᾠᾡᾤᾥᾦᾧ\u1F7D': 'ω',
    'Ϙϙ': 'ϙ',
    'Ϛϛ': 'ϛ',
    'ṇ\'\\/`´΄̣‵′᾿᾽᾽': '',
    ';····\u00B7\u0387': '·',
    'Ϡϡ': 'ϡ',
    '†‡': '†'
}

# Convert the replacement_map to a usable regex pattern
replacement_pattern = {}
for chars, replacement in replacement_map.items():
    for char in chars:
        replacement_pattern[char] = replacement

# Function to clean text based on the replacement map
def clean_text(text):
    return ''.join(replacement_pattern.get(char, char) for char in text)

# File paths
united_output_file = 'data/united.jsonl'
cleaned_output_file = 'data/cleaned_united.jsonl'

# Process and clean the JSONL file
with open(united_output_file, 'r', encoding='utf-8') as infile, open(cleaned_output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        entry = json.loads(line)
        entry['Edition_with_brackets'] = clean_text(entry.get('Edition_with_brackets', ''))
        entry['Edition_without_brackets'] = clean_text(entry.get('Edition_without_brackets', ''))
        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Cleaned data saved to {cleaned_output_file}")
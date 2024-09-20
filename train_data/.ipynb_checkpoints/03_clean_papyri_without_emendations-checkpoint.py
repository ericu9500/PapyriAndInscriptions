import os
import re
import json
import random
from bs4 import BeautifulSoup

def handle_tag(soup):
    for div in soup.find_all('div', {'type': 'edition'}):
        children = div.contents
        for child in children:
            div.insert_before(child)
        div.decompose()

def replace_with_hyphens(tag):
    text_inside = tag.get_text()
    # Count non-space characters
    char_count = len(text_inside.replace(" ", ""))
    # Replace the content with hyphens equal to the count of characters
    tag.replace_with('-' * char_count)

def clean_text(soup):
    # Decompose unwanted tags
    for note in soup.find_all('note'):
        note.decompose()
    for ab in soup.find_all('ab'):
        ab.unwrap()
    for expan in soup.find_all('expan'):
        expan.unwrap()
    for ex in soup.find_all('ex'):
        ex.unwrap()
    for supplied in soup.find_all('supplied'):
        replace_with_hyphens(supplied)
    for unclear in soup.find_all('unclear'):
        replace_with_hyphens(unclear)
    for handShift in soup.find_all('handShift'):
        handShift.unwrap()
    for g in soup.find_all('g'):
        g.unwrap()
    for app in soup.find_all('app'):
        app.unwrap()
    for lem in soup.find_all('lem'):
        lem.unwrap()
    for milestone in soup.find_all('milestone'):
        milestone.unwrap()
    for rdg in soup.find_all('rdg'):
        rdg.decompose()
    for add in soup.find_all('add'):
        add.unwrap()
    for space in soup.find_all('space'):
        space.unwrap()
    for hi in soup.find_all('hi'):
        hi.unwrap()
    for tag in soup.find_all('del'):
        tag.decompose()
    for surplus in soup.find_all('surplus'):
        surplus.decompose()
    for choice in soup.find_all('choice'):
        for orig in choice.find_all('orig'):
            orig.decompose()
        for reg in choice.find_all('reg'):
            reg.unwrap()
        choice.unwrap()
    for lb in soup.find_all('lb'):
        if lb.get('break') == 'no':
            lb.replace_with('€€')
        else:
            lb.replace_with(' ')
    for gap in soup.find_all('gap'):
        if gap.get('extent') == 'unknown':
            gap.replace_with('…')
        elif gap.get('unit') == 'line':
            gap.replace_with('…')
        elif gap.get('unit') == 'character' and gap.get('quantity'):
            try:
                quantity = int(gap['quantity'])
                gap.replace_with('-' * quantity)
            except ValueError:
                gap.replace_with('-')
    for div in soup.find_all('div'):
        if div.get('type') == 'textpart':
            div.unwrap()
    for num in soup.find_all('num'):
        unwrapped_content = num.get_text()  # Get the text inside <num>
        num.replace_with(f"⟨{unwrapped_content}⟩")  # Replace the <num> tag with the modified text
    for abbr in soup.find_all('abbr'):
        abbr.replace_with(f"{abbr.get_text()}…")

    # Extract and clean the text content
    text_content = ''.join(
        str(content) if isinstance(content, str) else content.get_text()
        for content in soup.strings
    ).strip()

    # Perform additional text replacements and formatting
    text_content = ' '.join(text_content.split())
    text_content = text_content.replace("€€ ", "").replace(" €€", "").replace("€€", "").replace(" ,", ",").replace(" .", ".").replace("\"", "").replace("#", "")
    
    # Regular expressions for additional cleaning
    replacements = [
        (r'[\n ]+', ' '),
        (r'[ﬂⲁⲂⲃⲅⲇⲉⲋⲍⲏⲑ\?ⲓⲕⲗöﬁ\／ⲙⲛⲝⲟⲡⲣⲥⲧⲩⲫⲭⲯⲱⲻⳉⳓ⳨⳿⸌⸍⸗ꜢꜣꜤꜥ⟦⤚⦿⟧●⎛⎜⎝⎞⎟⎠⎧⎨⎩⎫⎬⎭بةتث!"#$&<=>ß@ABC῎῾῏῞῟DEF‘’‚“”„GHIJKLMNOPQRSTUVWXYZ§¨±_abcdefghijklmnopqrstuvwxyz\{\|\}\~áâäçéìíîïòóôõúüıšʹʼʽˉ˙̱̀́̃̅̇̈̉̒̓̔͂͗ͅ]', ''),  # Remove special characters
        (r'[›※‾⁓⁢⁩­ ‪–—―‖]', ' '),
        (r'[∶⋮‧•\··\,\.\:]', '·'),
        (r'[-]', '-'),
        (r'[0-9]', ''),
        (r'[]', 'ε'),
        (r'[]', 'η'),
        (r'[∂θϑ]', 'θ'),
        (r'[􏰂]', 'ι'),
        (r'[]', 'ο'),
        (r'[὘]', 'υ'),
        (r'[µ]', 'μ'),
        (r'[ϕ]', 'φ'),
        (r'[]', 'ω'),
        (r'[\n]', ' '),
        (r'[ϢϣϥϨϩ⁦ϪϫϬϭ𐅵ϮϯϲϹׂء􏰁أؤإئابةتثجح<>خدذرزسشصضطعغـفقكụلمنهوىيٍّپᐧḍḎḏḤḥḪḫṃṭṯṱẖẠẹỈỉ]', '')
    ]

    for pattern, replacement in replacements:
        text_content = re.sub(pattern, replacement, text_content)

    return text_content

def extract_tm_number(soup):
    idno_tm_tag = soup.find('idno', type='TM')
    return idno_tm_tag.get_text(strip=True) if idno_tm_tag else 'null'

def process_xml_files(base_dir):
    output_data = []
    xml_files = [os.path.join(root, file)
                 for root, _, files in os.walk(base_dir)
                 for file in files if file.endswith('.xml')]

    for xml_file in xml_files:
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'xml')

            for div in soup.find_all('div', {'xml:lang': 'grc', 'type': 'edition'}):
                handle_tag(div)
                edition_text = clean_text(div)
                tm_number = extract_tm_number(soup)
                
                output_data.append({
                    "TM_Number": tm_number,
                    "Edition_without_brackets": edition_text
                })

        except Exception as e:
            print(f"Error processing file {xml_file}: {e}")

    # Create output directory if it doesn't exist
    output_dir = os.path.join('data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the output as a JSONL file in UTF-8 encoding
    output_file = os.path.join(output_dir, 'output_without_brackets.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    base_directory_ddb = 'DDB_EpiDoc_XML'
    process_xml_files(base_directory_ddb)


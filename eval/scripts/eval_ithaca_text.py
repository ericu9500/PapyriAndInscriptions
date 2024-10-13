import json
import re
import functools
import pickle
import os
from ithaca.eval import inference
from ithaca.models.model import Model
from ithaca.util.alphabet import GreekAlphabet
import jax
import argparse

def load_jsonl_file(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            entries = [json.loads(line) for line in file]
        return entries
    except FileNotFoundError:
        print(f"Error: The input file {input_file} does not exist.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse the JSONL file. JSONDecodeError: {e}")
        exit(1)

def replace_letters_missing(text, real_response):
    def replacement(match):
        missing_count = len(real_response)
        return '?' * missing_count
    return re.sub(r'\[(\d+) letters missing\]', replacement, text)

def replace_bullet_with_dot(text):
    return text.replace('·', '.')

def load_checkpoint(path):
    try:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        params = jax.device_put(checkpoint['params'])
        model = Model(**checkpoint['model_config'])
        forward = functools.partial(model.apply, params)

        region_map = checkpoint['region_map']
        alphabet = GreekAlphabet()
        alphabet.idx2word = checkpoint['alphabet']['idx2word']
        alphabet.word2idx = checkpoint['alphabet']['word2idx']

        return checkpoint['model_config'], region_map, alphabet, params, forward
    except FileNotFoundError:
        print(f"Error: The checkpoint file {path} does not exist.")
        exit(1)
    except pickle.UnpicklingError as e:
        print(f"Error: Failed to load checkpoint file. Pickle Error: {e}")
        exit(1)

def find_first_missing_region(text):
    match = re.search(r'\?+', text)
    if match:
        return match.start(), len(match.group())
    else:
        return None, 0

def get_isolated_restorations(restoration, missing_region_start, missing_region_length):
    ranked_predictions = sorted(restoration.predictions, key=lambda x: x.score, reverse=True)

    top_20_predictions = {}
    for i, pred in enumerate(ranked_predictions[:20], start=1):
        restored_part = pred.text[missing_region_start:missing_region_start + missing_region_length]
        top_20_predictions[f"rank_{i}"] = restored_part

    return top_20_predictions

def save_jsonl_output(data, output_file):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as file:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')
    except OSError as e:
        print(f"Error: Could not write to the output file {output_file}. OSError: {e}")
        exit(1)

def process_entry(entry, model_config, region_map, alphabet, params, forward, output_file):
    try:
        user_content = entry["messages"][1]["content"]
        assistant_content = entry["messages"][2]["content"]
        user_content_replaced = replace_letters_missing(user_content, assistant_content)
        user_content_replaced = replace_bullet_with_dot(user_content_replaced)

        if not 50 <= len(user_content_replaced) <= 750:
            raise ValueError(f'Text should be between 50 and 750 chars long, but the input is {len(user_content_replaced)} characters')

        missing_region_start, missing_region_length = find_first_missing_region(user_content_replaced)
        if missing_region_start is None:
            raise ValueError('No missing "?" region found in the input text.')

        restoration = inference.restore(
            user_content_replaced,
            forward=forward,
            params=params,
            alphabet=alphabet,
            vocab_char_size=model_config['vocab_char_size'],
            vocab_word_size=model_config['vocab_word_size']
        )

        top_20_predictions = get_isolated_restorations(restoration, missing_region_start, missing_region_length)

        result_data = {
            "real_response": assistant_content,
            "predictions": top_20_predictions
        }
        save_jsonl_output(result_data, output_file)
    except KeyError as e:
        print(f"Error processing entry: Missing key {e}")
    except ValueError as e:
        print(f"Skipping entry due to error: {e}")
    except Exception as e:
        print(f"Unexpected error during processing: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process JSONL file for restoration and save results.")
    parser.add_argument('--input_jsonl', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint file')
    parser.add_argument('--start_row', type=int, required=True, help='Start row index (inclusive) for processing')
    parser.add_argument('--end_row', type=int, required=True, help='End row index (inclusive) for processing')

    args = parser.parse_args()
    entries = load_jsonl_file(args.input_jsonl)
    model_config, region_map, alphabet, params, forward = load_checkpoint(args.checkpoint_path)
    entries_to_process = entries[args.start_row - 1: args.end_row]
    output_filename = re.sub(r'\.jsonl$', f'_rows_{args.start_row}_{args.end_row}.jsonl', args.output_jsonl)
    for entry in entries_to_process:
        process_entry(entry, model_config, region_map, alphabet, params, forward, output_filename)

if __name__ == '__main__':
    main()
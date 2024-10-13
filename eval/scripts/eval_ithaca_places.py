import json
import re
import functools
import pickle
import os
import random
import numpy as np
import jax
from ithaca.eval import inference
from ithaca.models.model import Model
from ithaca.util.alphabet import GreekAlphabet
from absl import flags
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
def load_region_map(region_file_path):
    """Load the region file and create a mapping of location_id to region name."""
    region_map = {}
    try:
        with open(region_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(';')
                region_name_id = parts[0]
                location_id = int(region_name_id.split('_')[-1])
                region_name = region_name_id.rsplit('_', 1)[0]
                region_map[location_id] = region_name
    except FileNotFoundError:
        print(f"Error: Region file {region_file_path} not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading region map: {e}")
        exit(1)

    return region_map

def process_entry(entry, model_config, region_map, alphabet, params, forward, location_map):
    try:
        user_content = entry.get("messages", [{}])[1].get("content", "")
        assistant_content = entry.get("messages", [{}])[2].get("content", "")

        if not user_content or not assistant_content:
            raise ValueError("User or assistant content is missing or empty")

        user_content_replaced = replace_letters_missing(user_content, assistant_content)
        user_content_replaced = replace_bullet_with_dot(user_content_replaced)

        if not 50 <= len(user_content_replaced) <= 750:
            raise ValueError(f'Text should be between 50 and 750 chars long, but the input is {len(user_content_replaced)} characters')
        attribution = inference.attribute(
            user_content_replaced,
            forward=forward,
            params=params,
            alphabet=alphabet,
            region_map=region_map,
            vocab_char_size=model_config['vocab_char_size'],
            vocab_word_size=model_config['vocab_word_size']
        )

        if not hasattr(attribution, 'locations') or not isinstance(attribution.locations, list):
            raise ValueError("Attribution locations not found or not in the correct format")

        sorted_attributions = sorted(attribution.locations, key=lambda x: x[1], reverse=True)
        top_3_attributions = sorted_attributions[:3]

        predictions = {
            f"rank_{i+1}": location_map.get(attr[0], "Unknown Region")
            for i, attr in enumerate(top_3_attributions)
        }

        result_data = {
            "real_response": assistant_content,
            "predictions": predictions
        }

        return result_data

    except KeyError as e:
        print(f"Error processing entry: Missing key {e}")
    except ValueError as e:
        print(f"Skipping entry due to error: {e}")
    except Exception as e:
        print(f"Unexpected error during processing: {e}")

    return None

def save_to_jsonl(output_file, data):
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Process JSONL files with region attribution.")
    parser.add_argument('--input_jsonl', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint file')
    parser.add_argument('--region_file_path', type=str, required=True, help='Path to region map file')
    
    args = parser.parse_args()

    entries = load_jsonl_file(args.input_jsonl)
    model_config, region_map, alphabet, params, forward = load_checkpoint(args.checkpoint_path)
    location_map = load_region_map(args.region_file_path)
    processed_entries = []
    for entry in entries:
        result = process_entry(entry, model_config, region_map, alphabet, params, forward, location_map)
        if result:
            processed_entries.append(result)
    save_to_jsonl(args.output_jsonl, processed_entries)

if __name__ == '__main__':
    main()
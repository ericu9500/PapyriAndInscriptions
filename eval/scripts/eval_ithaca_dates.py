import json
import re
import functools
import pickle
import os
import random
import numpy as np
import argparse
from ithaca.eval import inference
from ithaca.models.model import Model
from ithaca.util.alphabet import GreekAlphabet
import jax

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


def get_year_scores(entry, model_config, region_map, alphabet, params, forward):
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
        date_pred_y = np.array(attribution.year_scores)
        date_pred_x = np.arange(
          dataset_config.date_min + dataset_config.date_interval / 2,
          dataset_config.date_max + dataset_config.date_interval / 2,
          dataset_config.date_interval)

        date_pred_avg = np.dot(date_pred_y, date_pred_x)
        
        return assistant_content, date_pred_avg

    except KeyError as e:
        print(f"Error processing entry: Missing key {e}")
    except ValueError as e:
        print(f"Skipping entry due to error: {e}")
    except Exception as e:
        print(f"Unexpected error during processing: {e}")


def save_output_to_jsonl(output_file, results):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Process JSONL files and a model checkpoint for attribution.')
    parser.add_argument('--input_jsonl', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to the output JSONL file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file')
    
    args = parser.parse_args()
    input_file = args.input_jsonl
    checkpoint_path = args.checkpoint_path
    output_file = args.output_jsonl

    entries = load_jsonl_file(input_file)
    model_config, region_map, alphabet, params, forward = load_checkpoint(checkpoint_path)

    results = []
    
    for entry in entries:
        try:
            real_response, prediction = get_year_scores(entry, model_config, region_map, alphabet, params, forward)
            results.append({
                "real_response": real_response,
                "prediction": prediction
            })
        except Exception as e:
            print(f"Skipping entry due to error: {e}")
    
    save_output_to_jsonl(output_file, results)


if __name__ == '__main__':
    main()


import numpy as np

class dataset_config:
    date_interval = 10
    date_max = 800
    date_min = -800

def calculate_mean_date(year_scores):
    date_pred_x = np.arange(
        dataset_config.date_min + dataset_config.date_interval / 2,
        dataset_config.date_max + dataset_config.date_interval / 2,
        dataset_config.date_interval)
    year_scores = np.array(year_scores)
    year_scores /= year_scores.sum()
    date_pred_avg = np.dot(year_scores, date_pred_x)
    
    return date_pred_avg
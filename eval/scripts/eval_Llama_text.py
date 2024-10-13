"""
Run e.g.
python /mimer/NOBACKUP/groups/naiss2024-22-201/try/ToGit/eval/scripts/eva_llama_text_1.py \
    --model_path "/mimer/NOBACKUP/groups/naiss2024-22-1168/Winners_HF/3/PHI-minemerged05_reconstructer_3" \
    --dataset_path "/mimer/NOBACKUP/groups/naiss2024-22-201/try/ToGit/eval/def_test/test_files/test_PHI3_text.jsonl" \
    --output_dir "/mimer/NOBACKUP/groups/naiss2024-22-201/try/ToGit/eval/def_test/results/NyttEvalScript" \
    --beams 50 \
    --repetition_penalty 1 \
    --percentage 100
"""


import torch
from transformers import pipeline
import random
import json
import re
import difflib
import argparse
import os
parser = argparse.ArgumentParser(description="Run beam search on dataset and save results.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files")
parser.add_argument("--beams", type=int, default=20, help="Number of beams to use in the test")
parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty in top-20 test")
parser.add_argument("--percentage", type=float, default=5, help="Percentage of entries to test for each length")
args = parser.parse_args()
pipe = pipeline(
    "text-generation",
    model=args.model_path,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.bfloat16},
)
with open(args.dataset_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
def calculate_content_length(content):
    content = re.sub(r"[ \.·]", "", content)
    return len(content)

def normalize_content(content):
    content = content.replace("ς", "σ")
    content = re.sub(r"[ \.·0]", "", content)
    return content

def truncate_to_real_length(prediction, reference):
    reference_len = len(reference)
    return prediction[:reference_len]

def calculate_cer(reference, hypothesis):
    reference = normalize_content(reference)
    hypothesis = normalize_content(hypothesis)
    hypothesis = truncate_to_real_length(hypothesis, reference)
    matcher = difflib.SequenceMatcher(None, reference, hypothesis)
    return 1 - matcher.ratio()
def collect_entries_by_percentage(percentage):
    total_lengths = {i: 0 for i in range(1, 11)}
    collected_entries = {i: [] for i in range(1, 11)}
    
    for line in lines:
        data = json.loads(line)
        assistant_content = data['messages'][2]['content']
        content_length = calculate_content_length(assistant_content)
        if 1 <= content_length <= 10:
            total_lengths[content_length] += 1
    
    to_collect = {length: max(1, int(total_lengths[length] * (percentage / 100))) for length in range(1, 11)}

    for line in lines:
        if all(len(collected_entries[i]) >= to_collect[i] for i in range(1, 11)):
            break
        data = json.loads(line)
        assistant_content = data['messages'][2]['content']
        content_length = calculate_content_length(assistant_content)
        if 1 <= content_length <= 10 and len(collected_entries[content_length]) < to_collect[content_length]:
            collected_entries[content_length].append(data)

    return collected_entries

collected_entries = collect_entries_by_percentage(args.percentage)

cer_stats = {i: {'total_cer': 0, 'count': 0} for i in range(1, 11)}
correct_top_1 = 0
correct_top_20 = 0
total_entries = 0
model_name = os.path.basename(args.model_path).replace('.', '_')
dataset_name = os.path.basename(args.dataset_path).replace('.', '_')
output_filename_jsonl = f"{model_name}_{dataset_name}_{args.beams}_{args.repetition_penalty}_{args.percentage}.jsonl"
output_filename_txt = f"{model_name}_{dataset_name}_{args.beams}_{args.repetition_penalty}_{args.percentage}.txt"
output_path_jsonl = os.path.join(args.output_dir, output_filename_jsonl)
output_path_txt = os.path.join(args.output_dir, output_filename_txt)
def process_entries():
    global correct_top_1, correct_top_20, total_entries
    
    results = []
    
    for length, entries in collected_entries.items():
        print(f"\nProcessing entries with assistant content length = {length}:")
        for idx, data in enumerate(entries):
            messages = [
                {"role": "system", "content": data['messages'][0]['content']},
                {"role": "user", "content": data['messages'][1]['content']}
            ]
            correct_assistant_content = data['messages'][2]['content']
            generated_beam_1_content = pipe(
                messages,
                max_new_tokens=13,
                num_beams=args.beams,
                num_return_sequences=1,
                early_stopping=True
            )[0]['generated_text'][-1]["content"]

            cer_beam_1 = calculate_cer(correct_assistant_content, generated_beam_1_content)
            cer_stats[length]['total_cer'] += cer_beam_1
            cer_stats[length]['count'] += 1
            total_entries += 1

            is_top_1_correct = normalize_content(truncate_to_real_length(generated_beam_1_content, correct_assistant_content)) == normalize_content(correct_assistant_content)
            top_1_result = {"system": data['messages'][0], "user": data['messages'][1], "assistant": data['messages'][2], "top_1_prediction": generated_beam_1_content, "CER": cer_beam_1, "top_1_correct": is_top_1_correct}

            if is_top_1_correct:
                correct_top_1 += 1
                correct_top_20 += 1
                top_1_result["top_20_correct"] = True
                results.append(top_1_result)
            else:
                additional_beams = pipe(
                    messages,
                    max_new_tokens=13,
                    num_beams=args.beams,
                    num_return_sequences=args.beams - 1,
                    early_stopping=True,
                    repetition_penalty=args.repetition_penalty
                )

                found_in_top_20 = False
                for beam_output in additional_beams:
                    generated_beam_content = beam_output['generated_text'][-1]["content"]
                    truncated_beam_content = truncate_to_real_length(generated_beam_content, correct_assistant_content)
                    if normalize_content(truncated_beam_content) == normalize_content(correct_assistant_content):
                        found_in_top_20 = True
                        correct_top_20 += 1
                        break
                
                top_1_result["top_20_correct"] = found_in_top_20
                top_1_result["top_20_predictions"] = [beam['generated_text'][-1]["content"] for beam in additional_beams]
                results.append(top_1_result)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_path_jsonl, 'w', encoding='utf-8') as jsonl_file:
        for result in results:
            json.dump(result, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")
    with open(output_path_txt, 'w', encoding='utf-8') as txt_file:
        overall_total_cer = sum([cer_stats[length]['total_cer'] for length in range(1, 11)])
        overall_count = sum([cer_stats[length]['count'] for length in range(1, 11)])
    
        txt_file.write("Average CER for each length:\n")
        for length in range(1, 11):
            if cer_stats[length]['count'] > 0:
                avg_cer = cer_stats[length]['total_cer'] / cer_stats[length]['count']
                txt_file.write(f"Length {length}: Average CER = {avg_cer:.4f}\n")
    
        overall_avg_cer = overall_total_cer / overall_count if overall_count > 0 else 0
        txt_file.write(f"\nOverall Average CER: {overall_avg_cer:.4f}\n")
    
        top_1_accuracy = (correct_top_1 / total_entries) * 100 if total_entries > 0 else 0
        top_20_accuracy = (correct_top_20 / total_entries) * 100 if total_entries > 0 else 0
        txt_file.write(f"Top-1 Accuracy: {top_1_accuracy:.2f}%\n")
        txt_file.write(f"Top-20 Accuracy: {top_20_accuracy:.2f}%\n")
process_entries()
import json
import random
import os
import argparse
from transformers import pipeline
import torch
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Run model evaluation with specified parameters.")
parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file (messages JSONL).")
parser.add_argument("--percentage", type=float, default=100, help="Percentage of the dataset to process (default: 100%).")
parser.add_argument("--model", type=str, required=True, help="Huggingface model identifier.")
parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder (will be created if it does not exist).")
parser.add_argument("--num_beams", type=int, default=15, help="Number of beams for beam search (default: 15).")
args = parser.parse_args()

pipeline = pipeline(
    "text-generation",
    model=args.model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

os.makedirs(args.output_folder, exist_ok=True)

with open(args.dataset, "r", encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

allowed_prompts = [
    "Date this papyrus fragment!",
    "Date this inscription!",
    "Date this papyrus fragment to an exact year!",
    "Date this inscription to an exact year!"
]

percentage = args.percentage
num_entries_to_process = int(len(data) * (percentage / 100))

filtered_entries = [entry for entry in data if entry["messages"][0]["content"] in allowed_prompts]
filtered_entries = random.sample(filtered_entries, min(num_entries_to_process, len(filtered_entries)))

dataset_name = os.path.basename(args.dataset).split('.')[0]
model_name = args.model.split('/')[-1]
output_file_path = os.path.join(args.output_folder, f"dates_{dataset_name}_{model_name}.jsonl")

with open(output_file_path, "w", encoding='utf-8') as output_file:
    for entry in filtered_entries:
        messages = entry["messages"]
        system_message = messages[0]
        user_message = messages[1]
        assistant_message = messages[2]["content"]

        input_messages = [
            {"role": "system", "content": system_message["content"]},
            {"role": "user", "content": user_message["content"]},
        ]

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            input_messages,
            max_new_tokens=10,
            eos_token_id=terminators,
            num_beams=args.num_beams,
            num_return_sequences=1,
            early_stopping=True,
        )

        beam_contents = []
        for output in outputs:
            generated_text = output.get('generated_text', [])
            for item in generated_text:
                if item.get('role') == 'assistant':
                    beam_contents.append(item.get('content'))

        result = {
            "real_response": assistant_message,
            "predictions": {f"rank_{i+1}": content for i, content in enumerate(beam_contents)}
        }

        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Results saved to {output_file_path}")
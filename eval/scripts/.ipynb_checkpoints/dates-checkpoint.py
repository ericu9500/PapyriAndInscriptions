import json
import random
import os
import argparse
from transformers import pipeline
import torch

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run model evaluation with specified parameters.")
parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file (JSONL).")
parser.add_argument("--percentage", type=float, default=100, help="Percentage of the dataset to process (default: 100%).")
parser.add_argument("--model", type=str, required=True, help="Path to the model directory.")
parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder (will be created if it does not exist).")
args = parser.parse_args()

# Define your model and pipeline
model_id = args.model
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Define the dataset path and output directory
file_path = args.dataset
output_dir = args.output_folder
os.makedirs(output_dir, exist_ok=True)

# Load data from the JSONL file
with open(file_path, "r", encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Define the allowed system prompts
allowed_prompts = [
    "Date this papyrus fragment!",
    "Date this inscription!",
    "Date this papyrus fragment to an exact year!",
    "Date this inscription to an exact year!"
]

# Define percentage of the dataset to process
percentage = args.percentage
num_entries_to_process = int(len(data) * (percentage / 100))

# Filter entries matching allowed prompts
filtered_entries = [entry for entry in data if entry["messages"][0]["content"] in allowed_prompts]
filtered_entries = random.sample(filtered_entries, min(num_entries_to_process, len(filtered_entries)))

# Prepare output file path
dataset_name = os.path.basename(file_path).split('.')[0]
model_name = model_id.split('/')[-1]
output_file_path = os.path.join(output_dir, f"dates_{dataset_name}_{model_name}.jsonl")

# Process each entry and save results
with open(output_file_path, "w", encoding='utf-8') as output_file:
    for entry in filtered_entries:
        messages = entry["messages"]
        system_message = messages[0]
        user_message = messages[1]
        assistant_message = messages[2]["content"]

        # Prepare the input for the model
        input_messages = [
            {"role": "system", "content": system_message["content"]},
            {"role": "user", "content": user_message["content"]},
        ]

        # Set termination tokens
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate model output using beam search with 20 beams
        outputs = pipeline(
            input_messages,
            max_new_tokens=10,
            eos_token_id=terminators,
            num_beams=45,  # Use beam search with 20 beams
            num_return_sequences=3,  # Ensure 20 sequences are returned
            early_stopping=True,
        )

        # Extract the content for the 'assistant' role from each beam
        beam_contents = []
        for output in outputs:
            generated_text = output.get('generated_text', [])
            for item in generated_text:
                if item.get('role') == 'assistant':
                    beam_contents.append(item.get('content'))

        # Prepare the result dictionary
        result = {
            "real_response": assistant_message,
            "predictions": {f"rank_{i+1}": content for i, content in enumerate(beam_contents)}
        }

        # Save the result to the output file
        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Results saved to {output_file_path}")
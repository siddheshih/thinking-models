import torch
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

# ----------------------------
# 1. Dataset loading & preprocessing
# ----------------------------
def load_and_preprocess_aime_dataset(model_name, split="train"):
    """
    Loads and preprocesses the AIME dataset from Hugging Face.
    
    Args:
        model_name: The name of the Hugging Face model (for tokenizer compatibility).
        split: The dataset split to load (default is 'train').
        
    Returns:
        A tuple (processed_data, tokenizer) where processed_data is a list of dictionaries.
    """
    try:
        dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split=split)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        processed_data = []
        for example in dataset:
            question = example['Question']
            answer = example['Answer']
            year = example['Year']
            problem_number = example['Problem Number']
            problem_id = f"{year}-{problem_number}"
            
            # Clean LaTeX if needed.
            question = question.replace("\\$", "$")
            answer = answer.replace("\\$", "$")
            
            # Create the prompt (without forced tokens; these will be inserted later).
            input_prompt = (
                f"AIME Problem {problem_number} ({year}):\n{question}\n\n"
                "Solution:\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
            )
            
            processed_data.append({
                'problem': question,
                'solution': answer,
                'year': year,
                'problem_number': problem_number,
                'problem_id': problem_id,
                'input_prompt': input_prompt,
            })
        
        print(f"Successfully processed {len(processed_data)} AIME problems")
        return processed_data, tokenizer
    
    except Exception as e:
        print(f"Error loading AIME dataset: {str(e)}")
        return None, None

def extract_boxed_answer(solution_text):
    """
    Extracts the answer that appears inside \\boxed{} in LaTeX solutions.
    
    Args:
        solution_text: The full solution text containing a boxed answer.
        
    Returns:
        The extracted answer as a string, or None if no boxed answer is found.
    """
    pattern = r'\\boxed\{([^{}]+)\}'
    match = re.search(pattern, solution_text)
    if match:
        return match.group(1).strip()
    else:
        alt_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        alt_match = re.search(alt_pattern, solution_text)
        if alt_match:
            return alt_match.group(1).strip()
        return None

# ----------------------------
# 2. Setup forced token values and generation stages
# ----------------------------
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
set_seed(42)

# Load and preprocess dataset.
processed_data, tokenizer = load_and_preprocess_aime_dataset(model_name, split="train")
if processed_data is None:
    raise ValueError("Dataset loading failed.")

# For demonstration, use a small batch.
num_examples = 100
batch_prompts = [example["input_prompt"] for example in processed_data[:num_examples]]

# Determine prompt token length using the Hugging Face tokenizer.
# (Assume that all prompts are similar in length.)
prompt_tokens = tokenizer(batch_prompts[0], add_special_tokens=False)["input_ids"]
prompt_length = len(prompt_tokens)

# Setup forced tokens.
# We wish to force two tokens: a newline ("\n") and "</think>".
forced_token1 = "\n"
forced_token2 = "</think>"

forced_token1_ids = tokenizer.encode(forced_token1, add_special_tokens=False)
forced_token2_ids = tokenizer.encode(forced_token2, add_special_tokens=False)

if len(forced_token1_ids) != 1:
    raise ValueError("The forced newline token is encoded as multiple tokens; adjust your approach.")
if len(forced_token2_ids) != 1:
    raise ValueError("The forced '</think>' token is encoded as multiple tokens; adjust your approach.")

# We use the string forms for concatenation.
forced_token_str = forced_token1 + forced_token2

# Define a list of forced offsets (number of generated tokens after the prompt)
# at which to insert the forced tokens.
forced_offsets = [500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000]

# Total desired new tokens (output sequence length) after the prompt.
total_new_tokens = 16000

# ----------------------------
# 3. Setup vLLM model and sampling parameters
# ----------------------------
# Initialize the vLLM model.
model = LLM(model=model_name, max_model_len=16048, tensor_parallel_size=2)

temperature = 0.6
top_p = 0.95

# Lists to store metrics for plotting.
accuracy_list = []
length_list = []

# ----------------------------
# 4. Loop over forced offsets and generate outputs in two stages.
# ----------------------------
for force_offset in forced_offsets:
    # Stage 1: Generate up to the forced offset.
    sampling_params_stage1 = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=force_offset
    )
    
    # Calculate the remaining tokens to generate in stage 2.
    forced_token_length = len(tokenizer.encode(forced_token_str, add_special_tokens=False))
    remaining_tokens = total_new_tokens - force_offset - forced_token_length
    if remaining_tokens < 0:
        raise ValueError("Total output tokens is less than forced offset plus forced token length.")
    
    sampling_params_stage2 = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=remaining_tokens
    )
    
    print(f"Generating for forced offset {force_offset} ...")
    # Stage 1 generation.
    stage1_results = model.generate(batch_prompts, sampling_params_stage1)
    
    # Build new prompts by appending the forced token string.
    new_prompts = []
    for res in stage1_results:
        # Note: using res.outputs[0].text to access generated text.
        new_prompt = res.outputs[0].text + forced_token_str
        new_prompts.append(new_prompt)
    
    # Stage 2 generation.
    stage2_results = model.generate(new_prompts, sampling_params_stage2)
    
    # Combine stage1 and stage2 outputs.
    final_outputs = []
    for res1, res2 in zip(stage1_results, stage2_results):
        final_text = res1.outputs[0].text + forced_token_str + res2.outputs[0].text
        final_outputs.append(final_text)
    
    # Evaluate accuracy and record average generated sequence length.
    correct_count = 0
    total_count = 0
    token_lengths = []
    
    for i, final_text in enumerate(final_outputs):
        final_answer = extract_boxed_answer(final_text)
        ground_truth = processed_data[i]["solution"].strip().lower()
        extracted = final_answer.strip().lower() if final_answer else ""
        if extracted == ground_truth:
            correct_count += 1
        total_count += 1
        
        # Compute generated sequence length (in tokens).
        token_len = len(tokenizer.encode(final_text, add_special_tokens=False))
        token_lengths.append(token_len)
    
    acc = (correct_count / total_count) * 100 if total_count > 0 else 0.0
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    
    accuracy_list.append(acc)
    length_list.append(avg_length)
    
    print(f"Forced offset: {force_offset} -> Accuracy: {acc:.2f}%, Avg length: {avg_length:.2f} tokens")

# ----------------------------
# 5. Plot metrics and save plots
# ----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(forced_offsets, accuracy_list, marker='o')
plt.xlabel("Forced Offset (tokens)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Forced Token Insertion Offset")

plt.subplot(1, 2, 2)
plt.plot(forced_offsets, length_list, marker='o', color='orange')
plt.xlabel("Forced Offset (tokens)")
plt.ylabel("Average Generated Sequence Length (tokens)")
plt.title("Generated Sequence Length vs Forced Offset")

plt.tight_layout()

# Save plots as PNG and PDF.
plt.savefig("accuracy_and_length_plots.png", bbox_inches='tight', dpi=300)
plt.savefig("accuracy_and_length_plots.pdf", bbox_inches='tight')

plt.show()

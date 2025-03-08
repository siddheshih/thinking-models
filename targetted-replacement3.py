"""
Name is miselading, change the name the later, actually huggingface AIME
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from accelerate import Accelerator
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from contextlib import nullcontext
import pickle
import os
import bitsandbytes as bnb
from torch.cuda.amp import autocast

# ----------------------------
# 1. Dataset loading & preprocessing
# ----------------------------
def load_and_preprocess_aime_dataset(model_name, split="train"):
    """Loads and preprocesses the AIME dataset."""
    try:
        dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split=split)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processed_data = []
        for example in dataset:
            question = example['Question']
            answer = example['Answer']
            year = example['Year']
            problem_number = example['Problem Number']
            question = question.replace("\\$", "$")
            answer = answer.replace("\\$", "$")
            input_prompt = (
                f"AIME Problem {problem_number} ({year}):\n{question}\n\n"
                "Solution:\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
            )
            processed_data.append({
                'problem': question,
                'solution': answer,
                'year': year,
                'problem_number': problem_number,
                'input_prompt': input_prompt,
            })
        print(f"Successfully processed {len(processed_data)} AIME problems")
        return processed_data, tokenizer
    except Exception as e:
        print(f"Error loading AIME dataset: {str(e)}")
        return None, None

def extract_boxed_answer(solution_text):
    """Extracts the answer inside \\boxed{...}."""
    pattern = r'\\boxed\{([^{}]+)\}'
    match = re.search(pattern, solution_text)
    return match.group(1).strip() if match else ""

# ----------------------------
# 2. LogitsProcessor for Forced Tokens
# ----------------------------
class MultiStepForceTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, forced_tokens: dict):
        """
        forced_tokens: dict mapping absolute generation step (prompt length + generated tokens)
                       to forced token id.
        """
        self.forced_tokens = forced_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_step = input_ids.shape[1]
        if current_step in self.forced_tokens:
            token_id = self.forced_tokens[current_step]
            # Use scatter_ to force the token by assigning a large positive value.
            scores.scatter_(1, torch.tensor([[token_id]], device=scores.device), 1e9)
        return scores

# ----------------------------
# 3. Model loading, Accelerator, and setup (for two H100s)
# ----------------------------
temperature = 0.6
top_p = 0.95
max_new_tokens = 16000  # Total new tokens per example
num_examples = 100      # Process first 100 examples
batch_size = 2          # Process in batches of 2

# Use the optimized quantized model from unsloth.
model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"

processed_data, tokenizer = load_and_preprocess_aime_dataset(model_name, split="train")
if processed_data is None:
    raise ValueError("Dataset loading failed.")

set_seed(42)
accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16")
device = accelerator.device

# Load the quantized model in 4-bit mode using bitsandbytes.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_quant_type="nf4",
    device_map="auto",
)
# If running on two H100s, ensure tensor parallelism is enabled if supported (often handled by device_map).
model = accelerator.prepare(model)
model.eval()

# Create a directory to save checkpoints.
save_dir = "hf_generation_checkpoints"
os.makedirs(save_dir, exist_ok=True)
results_file = os.path.join(save_dir, "generation_results.pkl")

# ----------------------------
# 4. Setup forced tokens
# ----------------------------
forced_token1 = "\n"
forced_token2 = "</think>"
forced_token1_id = tokenizer.encode(forced_token1, add_special_tokens=False)[0]
forced_token2_id = tokenizer.encode(forced_token2, add_special_tokens=False)[0]

# ----------------------------
# 5. Loop over forced offsets in batches; skip OOM batches; save checkpoints
# ----------------------------
forced_offsets = [500, 1000, 2000, 4000, 6000, 8000, 9000, 10000, 12000, 14000]
results_dict = {}   # Key: forced_offset, Value: list of generation strings
metrics_dict = {}   # Key: forced_offset, Value: (accuracy, avg_length)

# Use underlying generate function.
generate_fn = model.module.generate if hasattr(model, 'module') else model.generate
num_total = min(num_examples, len(processed_data))
accuracy_list = []
length_list = []

for force_offset in forced_offsets:
    all_correct = 0
    all_total = 0
    all_lengths = []
    gen_texts = []
    
    for i in tqdm(range(0, num_total, batch_size), desc=f"Forced offset {force_offset}"):
        batch_data = processed_data[i : i + batch_size]
        batch_prompts = [ex["input_prompt"] for ex in batch_data]
        encoded_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids_batch = encoded_inputs.input_ids.to(device)
        attention_mask_batch = encoded_inputs.attention_mask.to(device)
        batch_prompt_length = input_ids_batch.shape[1]
        
        forced_tokens = {
            batch_prompt_length + force_offset: forced_token1_id,
            batch_prompt_length + force_offset + 1: forced_token2_id
        }
        logits_processor = LogitsProcessorList([MultiStepForceTokenLogitsProcessor(forced_tokens)])
        
        try:
            with torch.no_grad(), autocast(enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                outputs = generate_fn(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    logits_processor=logits_processor,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at forced offset {force_offset}, batch {i//batch_size}; skipping this batch.")
                torch.cuda.empty_cache()
                continue  # Skip this batch
            else:
                raise e
        
        decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        for j, generated_text in enumerate(decoded_texts):
            final_answer = extract_boxed_answer(generated_text)
            ground_truth = batch_data[j]["solution"].strip().lower()
            extracted_ans = final_answer.strip().lower() if final_answer else ""
            if extracted_ans == ground_truth:
                all_correct += 1
            all_total += 1
            token_len = len(tokenizer.encode(generated_text, add_special_tokens=False))
            all_lengths.append(token_len)
            gen_texts.append(generated_text)
        
        del outputs, decoded_texts, input_ids_batch, attention_mask_batch
        torch.cuda.empty_cache()
    
    if all_total > 0:
        acc = (all_correct / all_total) * 100
        avg_length = sum(all_lengths) / len(all_lengths)
    else:
        acc = None
        avg_length = None
    
    accuracy_list.append(acc)
    length_list.append(avg_length)
    metrics_dict[force_offset] = (acc, avg_length)
    results_dict[force_offset] = gen_texts
    print(f"Forced offset: {force_offset} -> Accuracy: {acc if acc is not None else 'N/A'}%, Avg length: {avg_length if avg_length is not None else 'N/A'} tokens")
    
    # Save intermediate checkpoint.
    with open(results_file, "wb") as f:
        pickle.dump({"metrics": metrics_dict, "generations": results_dict}, f)

# ----------------------------
# 6. Plot metrics and save the plots.
# ----------------------------
valid_offsets = [fo for fo, (acc, _) in metrics_dict.items() if acc is not None]
valid_accuracy = [acc for (acc, _) in metrics_dict.values() if acc is not None]
valid_length = [l for (_, l) in metrics_dict.values() if l is not None]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(valid_offsets, valid_accuracy, marker='o')
plt.xlabel("Forced Offset (tokens)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Forced Token Insertion Offset")

plt.subplot(1, 2, 2)
plt.plot(valid_offsets, valid_length, marker='o', color='orange')
plt.xlabel("Forced Offset (tokens)")
plt.ylabel("Avg Generated Sequence Length (tokens)")
plt.title("Generated Sequence Length vs Forced Offset")

plt.tight_layout()
plt.savefig("hf_accuracy_and_length_plots.png", bbox_inches='tight', dpi=300)
plt.savefig("hf_accuracy_and_length_plots.pdf", bbox_inches='tight')
plt.show()

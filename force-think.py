from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from random import seed, randint, sample
import re
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

def set_seed(seed_value):
    """Set seed for reproducibility."""
    seed(seed_value)

def extract_boxed_answer(solution_text):
    """
    Extracts the answer that appears inside \boxed{} in LaTeX solutions.
    
    Args:
        solution_text: The full solution text containing a boxed answer.
        
    Returns:
        The extracted answer as a string, or None if no boxed answer is found.
    """
    import re
    
    # Pattern to match content inside \boxed{...}
    pattern = r'\\boxed\{([^{}]+)\}'
    
    # Search for the pattern
    match = re.search(pattern, solution_text)
    
    if match:
        # Return the content inside the boxed environment
        return match.group(1).strip()
    else:
        # Try alternative pattern for cases where there might be nested braces
        alt_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        alt_match = re.search(alt_pattern, solution_text)
        
        if alt_match:
            return alt_match.group(1).strip()
        
        return None

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
            problem_id = f"{year}-{problem_number}"
            question = question.replace("\\$", "$")
            answer = answer.replace("\\$", "$")
            input_prompt = (
                f"AIME Problem {problem_number} ({year}):\n{question}\n\n"
                "Solution:\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
                "<think>\n"
            )
            tokenized_input = tokenizer(input_prompt, return_tensors="pt")
            processed_data.append({
                'problem': question,
                'solution': answer,
                'year': year,
                'problem_number': problem_number,
                'problem_id': problem_id,
                'input_prompt': input_prompt,
                'tokenized_input': tokenized_input,
            })
        print(f"Successfully processed {len(processed_data)} AIME problems")
        return processed_data, tokenizer
    except Exception as e:
        print(f"Error loading AIME dataset: {str(e)}")
        return None, None

def extract_thinking_trace(generated_text):
    """Extracts the thinking trace."""
    think_end = generated_text.find("</think>")
    if think_end != -1:
        thinking_trace = generated_text[:think_end + 8]
        thinking_trace_len = think_end + 8
        non_thinking_text = generated_text[think_end + 8:]
        non_thinking_len = len(non_thinking_text)
        return thinking_trace, thinking_trace_len, non_thinking_len
    else:
        return "", len(generated_text), 0

def replace_with_random_tokens(text, fraction, tokenizer):
    """Replaces a fraction of tokens in the text with random tokens."""
    tokens = tokenizer.tokenize(text)
    num_to_replace = int(len(tokens) * fraction)
    replace_indices = sample(range(len(tokens)), num_to_replace)  # Randomly select indices

    for i in replace_indices:
        random_token_id = randint(0, tokenizer.vocab_size - 1)  # Choose a random token ID
        tokens[i] = tokenizer.decode([random_token_id]) # Replace with decoded token

    return "".join(tokens)


def run_experiment_original(processed_data, model, tokenizer, num_examples, sampling_params):
    """Runs the original experiment (no augmentation)."""
    batch_prompts = [example["input_prompt"] for example in processed_data[:num_examples]]
    results = model.generate(batch_prompts, sampling_params)
    thinking_trace_lengths = []
    non_thinking_lengths = []
    correct_count = 0
    total = 0
    generated_thinking_traces = []

    for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
        generated_text = result.outputs[0].text
        final_answer = extract_boxed_answer(generated_text)
        think_trace, think_len, non_think_len = extract_thinking_trace(generated_text)
        thinking_trace_lengths.append(think_len)
        non_thinking_lengths.append(non_think_len)
        generated_thinking_traces.append(think_trace)
        ground_truth = example["solution"].strip().lower()
        extracted = final_answer.strip().lower() if final_answer else ""
        if extracted == ground_truth:
            correct_count += 1
        total += 1

    avg_thinking_trace_len = sum(thinking_trace_lengths) / len(thinking_trace_lengths) if thinking_trace_lengths else 0
    avg_non_thinking_len = sum(non_thinking_lengths) / len(non_thinking_lengths) if non_thinking_lengths else 0
    accuracy = (correct_count / total) * 100 if total > 0 else 0.0

    return results, accuracy, avg_thinking_trace_len, avg_non_thinking_len, generated_thinking_traces


def run_experiment_random_replacement(processed_data, model, tokenizer, num_examples, sampling_params, thinking_prompts, trace_fraction, replacement_fraction):
    """Runs experiment with random token replacement in the thinking trace."""
    batch_prompts = []
    for i, example in enumerate(processed_data[:num_examples]):
        prompt_to_inject = thinking_prompts[i]
        inject_len = int(len(prompt_to_inject) * trace_fraction)
        injected_prompt = prompt_to_inject[:inject_len]

        # Replace tokens in the *injected* portion
        corrupted_prompt = replace_with_random_tokens(injected_prompt, replacement_fraction, tokenizer)

        input_prompt = (
            f"AIME Problem {example['problem_number']} ({example['year']}):\n{example['problem']}\n\n"
            f"Solution:\n<think>\n{corrupted_prompt}"  # Use the corrupted prompt
        )
        batch_prompts.append(input_prompt)

    results = model.generate(batch_prompts, sampling_params)
    thinking_trace_lengths = []
    non_thinking_lengths = []
    correct_count = 0
    total = 0

    for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
        generated_text = result.outputs[0].text
        final_answer = extract_boxed_answer(generated_text)
        think_trace, think_len, non_think_len = extract_thinking_trace(generated_text)
        thinking_trace_lengths.append(think_len)
        non_thinking_lengths.append(non_think_len)
        ground_truth = example["solution"].strip().lower()
        extracted = final_answer.strip().lower() if final_answer else ""
        if extracted == ground_truth:
            correct_count += 1
        total += 1

    avg_thinking_trace_len = sum(thinking_trace_lengths) / len(thinking_trace_lengths) if thinking_trace_lengths else 0
    avg_non_thinking_len = sum(non_thinking_lengths) / len(non_thinking_lengths) if non_thinking_lengths else 0
    accuracy = (correct_count / total) * 100 if total > 0 else 0.0

    return results, accuracy, avg_thinking_trace_len, avg_non_thinking_len


# --- Colab Setup and Execution ---

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
processed_data, tokenizer = load_and_preprocess_aime_dataset(model_name, split="train")
if processed_data is None:
    raise ValueError("Dataset loading failed.")
set_seed(42)
model = LLM(model=model_name, max_model_len=16048)
temperature = 0.6
top_p = 0.95
max_new_tokens = 15000
num_examples = 200 # Reduced for faster experimentation
sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

print("Running Experiment 1: No Prompt Augmentation")
results_exp1, accuracy_exp1, avg_think_len_exp1, avg_non_think_len_exp1, thinking_prompts_exp1 = run_experiment_original(
    processed_data, model, tokenizer, num_examples, sampling_params
)
print(f"Accuracy (Experiment 1): {accuracy_exp1:.2f}%")
print(f"Average thinking trace length (Experiment 1): {avg_think_len_exp1:.2f} characters")
print(f"Average non-thinking text length (Experiment 1): {avg_non_think_len_exp1:.2f} characters")


# --- Experiment 3: Random Token Replacement ---
print("\nRunning Experiment 3: Random Token Replacement")
trace_fractions = [0.7]  # Use 70% of the trace, as per the prompt
replacement_fractions = [0.3, 0.5, 0.7]  # Replace 10%, 30%, 50%, 70%, 90% of the *injected* trace
num_runs = 5

all_accuracies_replacement = []
all_avg_thinking_lengths_replacement = []
all_avg_non_thinking_lengths_replacement = []

for replacement_fraction in replacement_fractions:
    print(f"  Replacement Fraction: {replacement_fraction}")
    accuracies_runs = []
    thinking_lengths_runs = []
    non_thinking_lengths_runs = []

    for run in range(num_runs):
        set_seed(42 + run)
        results_exp3, accuracy_exp3, avg_think_len_exp3, avg_non_think_len_exp3 = run_experiment_random_replacement(
            processed_data, model, tokenizer, num_examples, sampling_params,
            thinking_prompts_exp1, trace_fractions[0], replacement_fraction  # Use the first trace_fraction
        )
        accuracies_runs.append(accuracy_exp3)
        thinking_lengths_runs.append(avg_think_len_exp3)
        non_thinking_lengths_runs.append(avg_non_think_len_exp3)
        print(f"    Run {run + 1}: Accuracy: {accuracy_exp3:.2f}%, Think Len: {avg_think_len_exp3:.2f}, Non-Think Len: {avg_non_think_len_exp3:.2f}")

    all_accuracies_replacement.append(accuracies_runs)
    all_avg_thinking_lengths_replacement.append(np.mean(thinking_lengths_runs))
    all_avg_non_thinking_lengths_replacement.append(np.mean(non_thinking_lengths_runs))

# --- Plotting (Random Replacement) ---
plt.figure(figsize=(15, 5))

# Accuracy Plot
plt.subplot(1, 3, 1)
mean_accuracies = np.mean(all_accuracies_replacement, axis=1)
std_accuracies = np.std(all_accuracies_replacement, axis=1)
plt.errorbar(replacement_fractions, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=5)
plt.xlabel('Replacement Fraction')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Replacement Fraction')
plt.grid(True)

# Average Thinking Length Plot
plt.subplot(1, 3, 2)
plt.plot(replacement_fractions, all_avg_thinking_lengths_replacement, marker='o', color='r')
plt.xlabel('Replacement Fraction')
plt.ylabel('Average Thinking Length')
plt.title('Thinking Length vs. Replacement Fraction')
plt.grid(True)

# Average Non-Thinking Length Plot
plt.subplot(1, 3, 3)
plt.plot(replacement_fractions, all_avg_non_thinking_lengths_replacement, marker='o', color='g')
plt.xlabel('Replacement Fraction')
plt.ylabel('Average Non-Thinking Length')
plt.title('Non-Thinking Length vs. Replacement Fraction')
plt.grid(True)

plt.tight_layout()

output_dir_replacement = "plot_results_replacement_2"
if not os.path.exists(output_dir_replacement):
    os.makedirs(output_dir_replacement)
plt.savefig(os.path.join(output_dir_replacement, "combined_plots_replacement.png"))
plt.show()


del model
del tokenizer
del processed_data
del sampling_params

import gc
gc.collect()
try:
    from numba import cuda
    cuda.select_device(0)
    cuda.close()
except ImportError:
    print("Numba CUDA not found. Skipping GPU memory release.")
except Exception as e:
    print(f"Error releasing GPU memory: {e}")
finally:
    print("Model and associated resources have been deleted.")
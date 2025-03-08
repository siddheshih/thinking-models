from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from random import seed, randint, sample
import re
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import pandas as pd


def set_seed(seed_value):
    """Set seed for reproducibility."""
    seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def extract_boxed_answer(solution_text):
    """Extracts the answer inside \boxed{}."""
    pattern = r'\\boxed\{([^{}]+)\}'
    match = re.search(pattern, solution_text)
    if match:
        return match.group(1).strip()
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
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token
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

def inject_random_segment(full_prompt, inject_len, tokenizer):
    """Injects a random segment from the *tokenized* prompt."""
    tokens = tokenizer.encode(full_prompt) # Encode
    if inject_len <= 0 or inject_len > len(tokens):
        return ""
    start_index = randint(0, len(tokens) - inject_len)
    injected_tokens = tokens[start_index:start_index + inject_len]
    return tokenizer.decode(injected_tokens) # Decode back into a string


def run_experiment_original(processed_data, model, tokenizer, num_examples, sampling_params):
    """Runs the original experiment (no augmentation)."""
    batch_prompts = [example["input_prompt"] for example in processed_data[:num_examples]]
    results = model.generate(batch_prompts, sampling_params)
    data = []

    for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
        generated_text = result.outputs[0].text
        final_answer = extract_boxed_answer(generated_text)
        think_trace, think_len, non_think_len = extract_thinking_trace(generated_text)
        ground_truth = example["solution"].strip().lower()
        extracted = final_answer.strip().lower() if final_answer else ""
        correct = (extracted == ground_truth)
        accuracy = 100.0 if correct else 0.0

        data.append({
            'type': 'original',
            'fraction': 0.0,
            'run': 0,
            'accuracy': accuracy,
            'thinking_length': think_len,
            'non_thinking_length': non_think_len,
            'problem_id': example['problem_id'] #Unique ID
        })
    return data, [r.outputs[0].text for r in results]


def run_experiment_random_injection(processed_data, model, tokenizer, num_examples, sampling_params, thinking_prompts, thinking_prompt_fraction, run_number):
    """Runs experiment with random injection."""
    batch_prompts = []
    for i, example in enumerate(processed_data[:num_examples]):
        prompt_to_inject = thinking_prompts[i]
        # Use the TOKEN length, get it before any processing
        prompt_token_length = len(tokenizer.encode(prompt_to_inject))
        inject_len = int(prompt_token_length * thinking_prompt_fraction)
        injected_prompt = inject_random_segment(prompt_to_inject, inject_len, tokenizer)  # Pass tokenizer

        input_prompt = (
            f"AIME Problem {example['problem_number']} ({example['year']}):\n{example['problem']}\n\n"
            "Solution:\n<think>\n"
            "******\n"
            f"\n{injected_prompt}\n"
            "******\n"
            "</think>\n"
        )
        batch_prompts.append(input_prompt)

    results = model.generate(batch_prompts, sampling_params)
    data = []

    for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
        generated_text = result.outputs[0].text
        final_answer = extract_boxed_answer(generated_text)
        think_trace, think_len, non_think_len = extract_thinking_trace(generated_text)
        ground_truth = example["solution"].strip().lower()
        extracted = final_answer.strip().lower() if final_answer else ""
        correct = (extracted == ground_truth)
        accuracy = 100.0 if correct else 0.0

        data.append({
            'type': 'injection',
            'fraction': thinking_prompt_fraction,  # Use consistent naming
            'run': run_number,
            'accuracy': accuracy,
            'thinking_length': think_len,
            'non_thinking_length': non_think_len,
            'problem_id': example['problem_id']
        })
    return data

# --- Main Experiment Execution ---
if __name__ == '__main__':
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    processed_data, tokenizer = load_and_preprocess_aime_dataset(model_name, split="train")
    if processed_data is None:
        raise ValueError("Dataset loading failed.")

    set_seed(42)

    # Use BitsAndBytesConfig for quantization (optional, but recommended)
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     # llm_int8_enable_fp32_cpu_offload=True  # Optional
    # )

    # Initialize vLLM model
    model = LLM(model=model_name, max_model_len=16048, dtype="half", tensor_parallel_size=2) #Quantization and dtype
    temperature = 0.6
    top_p = 0.95
    max_new_tokens = 15000
    num_examples = 200  # Adjust as needed
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

    print("Running Experiment 1: No Prompt Augmentation")
    original_data, thinking_prompts_exp1 = run_experiment_original(
        processed_data, model, tokenizer, num_examples, sampling_params
    )
    all_data = original_data

    # --- Experiment 2: Random Injection ---
    print("\nRunning Experiment 2: Random Injection")
    prompt_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]  # Example fractions
    num_runs = 3

    for fraction in prompt_fractions:
        print(f"  Injection Fraction: {fraction}")
        for run in range(num_runs):
            set_seed(42 + run)  # Different seed for each run
            injection_data = run_experiment_random_injection(
                processed_data, model, tokenizer, num_examples, sampling_params,
                thinking_prompts_exp1, fraction, run + 1  # Pass run number
            )
            all_data.extend(injection_data)
            print(f"    Run {run + 1}: Completed")

    # --- Data Aggregation and CSV Export ---
    df = pd.DataFrame(all_data)
    output_csv_path = "experiment_results_pi.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"Experiment results saved to: {output_csv_path}")

# --- Plotting ---
    plt.figure(figsize=(15, 5))

    # Accuracy Plot (with error bars and original accuracy)
    plt.subplot(1, 3, 1)
    # Calculate mean and std for injection runs
    injection_summary = df[df['type'] == 'injection'].groupby('fraction')['accuracy'].agg(['mean']).reset_index()
    plt.plot(injection_summary['fraction'], injection_summary['mean'], marker='o', linestyle='-', label='Injection')

    # Add original accuracy as a horizontal line
    original_accuracy = df[df['type'] == 'original']['accuracy'].mean()
    plt.axhline(y=original_accuracy, color='r', linestyle='--', label=f'Original ({original_accuracy:.2f}%)')

    plt.xlabel('Injection Fraction')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Injection Fraction')
    plt.legend()  # Add a legend
    plt.grid(True)
    plt.ylim(0, 100)  # Set y-axis limits for accuracy

    # Average Thinking Length Plot
    plt.subplot(1, 3, 2)
    thinking_length_means = df[df['type'] == 'injection'].groupby('fraction')['thinking_length'].mean().reset_index()
    plt.plot(thinking_length_means['fraction'], thinking_length_means['thinking_length'], marker='o', color='r')
    plt.xlabel('Injection Fraction')
    plt.ylabel('Average Thinking Length')
    plt.title('Thinking Length vs. Injection Fraction')
    plt.grid(True)

    # Average Non-Thinking Length Plot
    plt.subplot(1, 3, 3)
    non_thinking_length_means = df[df['type'] == 'injection'].groupby('fraction')['non_thinking_length'].mean().reset_index()
    plt.plot(non_thinking_length_means['fraction'], non_thinking_length_means['non_thinking_length'], marker='o', color='g')
    plt.xlabel('Injection Fraction')
    plt.ylabel('Average Non-Thinking Length')
    plt.title('Non-Thinking Length vs. Injection Fraction')
    plt.grid(True)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = "plot_results_prompt_injection_2"  # More descriptive name
    os.makedirs(output_dir, exist_ok=True)

    # Save the combined figure
    plt.savefig(os.path.join(output_dir, "combined_plots.png"))
    plt.savefig(os.path.join(output_dir, "combined_plots.pdf"))


    plt.figure(figsize=(5, 5))  # Create a new figure for the accuracy plot
    plt.errorbar(injection_summary['fraction'], injection_summary['mean'], yerr=injection_summary['std'], fmt='o-', capsize=5, label='Injection')
    plt.axhline(y=original_accuracy, color='r', linestyle='--', label=f'Original ({original_accuracy:.2f}%)')
    plt.xlabel('Injection Fraction')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Injection Fraction')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.savefig(os.path.join(output_dir, "accuracy_plot.pdf"))


    plt.figure(figsize=(5, 5))  # New figure for thinking length
    plt.plot(thinking_length_means['fraction'], thinking_length_means['thinking_length'], marker='o', color='r')
    plt.xlabel('Injection Fraction')
    plt.ylabel('Average Thinking Length')
    plt.title('Thinking Length vs. Injection Fraction')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "thinking_length_plot.png"))
    plt.savefig(os.path.join(output_dir, "thinking_length_plot.pdf"))


    plt.figure(figsize=(5, 5))  # New figure for non-thinking length
    plt.plot(non_thinking_length_means['fraction'], non_thinking_length_means['non_thinking_length'], marker='o', color='g')
    plt.xlabel('Injection Fraction')
    plt.ylabel('Average Non-Thinking Length')
    plt.title('Non-Thinking Length vs. Injection Fraction')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "non_thinking_length_plot.png"))
    plt.savefig(os.path.join(output_dir, "non_thinking_length_plot.pdf"))


    plt.show()
    
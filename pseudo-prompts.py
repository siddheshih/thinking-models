from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd

def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    import numpy as np
    random.seed(seed_value)
    np.random.seed(seed_value)


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
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token
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
            )  # Initial part of the prompt, before pseudo-thinking
            processed_data.append({
                'problem': question,
                'solution': answer,
                'year': year,
                'problem_number': problem_number,
                'problem_id': problem_id,
                'input_prompt_prefix': input_prompt,  # Store prefix separately
            })
        print(f"Successfully processed {len(processed_data)} AIME problems")
        return processed_data, tokenizer
    except Exception as e:
        print(f"Error loading AIME dataset: {str(e)}")
        return None, None

def generate_pseudo_thinking_prompt(prefix, num_lines):
    """Generates the full prompt with pseudo-thinking content."""
    pseudo_thinking = "\n".join(["*" * 10] * num_lines)  # Each line has 10 stars
    return f"{prefix}{pseudo_thinking}\n</think>\n"


def run_experiment(processed_data, model, tokenizer, num_examples, sampling_params, pseudo_thinking_lengths):
    """Runs the experiment with varying pseudo-thinking lengths."""
    all_data = []

    for num_lines in pseudo_thinking_lengths:
        print(f"Running with pseudo-thinking length: {num_lines} lines")
        for run in range(5):  # 5 runs for each length
            set_seed(42 + run)  # Different seed for each run
            batch_prompts = []
            for example in processed_data[:num_examples]:
                full_prompt = generate_pseudo_thinking_prompt(example['input_prompt_prefix'], num_lines)
                batch_prompts.append(full_prompt)

            results = model.generate(batch_prompts, sampling_params)
            for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
                generated_text = result.outputs[0].text
                final_answer = extract_boxed_answer(generated_text)
                ground_truth = example["solution"].strip().lower()
                extracted = final_answer.strip().lower() if final_answer else ""
                correct = (extracted == ground_truth)
                accuracy = 100.0 if correct else 0.0
                #Simple length count
                think_end = generated_text.find("</think>")
                if think_end != -1:
                  thinking_trace_len = think_end + 8
                  non_thinking_text = generated_text[think_end + 8:]
                  non_thinking_len = len(non_thinking_text)
                else:
                  thinking_trace_len, non_thinking_len =  len(generated_text), 0

                all_data.append({
                    'type': 'pseudo_thinking',
                    'lines': num_lines,
                    'run': run + 1,
                    'accuracy': accuracy,
                    'thinking_length': thinking_trace_len,
                    'non_thinking_length': non_thinking_len,
                    'problem_id': example['problem_id']
                })
            print(f"  Run {run + 1}: Completed")

    return all_data

# --- Main Experiment Execution ---
if __name__ == '__main__':
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    processed_data, tokenizer = load_and_preprocess_aime_dataset(model_name, split="train")
    if processed_data is None:
        raise ValueError("Dataset loading failed.")

    # vLLM setup
    model = LLM(model=model_name, max_model_len=16048, dtype="half", tensor_parallel_size=2) # Removed quantization
    temperature = 0.6
    top_p = 0.95
    max_new_tokens = 15000  # Adjust as needed
    num_examples = 100 # Number of examples

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=["</think>"])

    # Pseudo-thinking lengths to test
    pseudo_thinking_lengths = [1, 10, 100, 1000, 2000, 5000, 10000, 12000]

    # Run the experiment
    experiment_data = run_experiment(processed_data, model, tokenizer, num_examples, sampling_params, pseudo_thinking_lengths)

    # --- Data Aggregation and CSV Export ---
    df = pd.DataFrame(experiment_data)
    output_csv_path = "pseudo_thinking_experiment_results.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"Experiment results saved to: {output_csv_path}")

  # --- Plotting ---
    plt.figure(figsize=(15, 5))

    # Accuracy Plot (with error bars)
    plt.subplot(1, 3, 1)
    accuracy_summary = df.groupby('lines')['accuracy'].agg(['mean']).reset_index() # Removed 'std'
    plt.plot(accuracy_summary['lines'], accuracy_summary['mean'], marker='o', linestyle='-', label='Pseudo-Thinking') # Changed to plt.plot
    plt.xlabel('Number of Lines')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Pseudo-Thinking Length')
    plt.xscale('log')  # Use log scale for x-axis
    plt.grid(True)
    plt.ylim(0, 100)

    # Average Thinking Length Plot
    plt.subplot(1, 3, 2)
    thinking_length_means = df.groupby('lines')['thinking_length'].mean().reset_index()
    plt.plot(thinking_length_means['lines'], thinking_length_means['thinking_length'], marker='o', color='r')
    plt.xlabel('Number of Lines')
    plt.ylabel('Average Thinking Length')
    plt.title('Thinking Length vs. Pseudo-Thinking Length')
    plt.xscale('log')
    plt.grid(True)

    # Average Non-Thinking Length Plot
    plt.subplot(1, 3, 3)
    non_thinking_length_means = df.groupby('lines')['non_thinking_length'].mean().reset_index()
    plt.plot(non_thinking_length_means['lines'], non_thinking_length_means['non_thinking_length'], marker='o', color='g')
    plt.xlabel('Number of Lines')
    plt.ylabel('Average Non-Thinking Length')
    plt.title('Non-Thinking Length vs. Pseudo-Thinking Length')
    plt.xscale('log')
    plt.grid(True)

    plt.tight_layout()

    # Create output directory and save plots
    output_dir = "plot_results_pseudo_thinking_2"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "combined_plots.png"))
    plt.savefig(os.path.join(output_dir, "combined_plots.pdf"))

    # Individual plots
    plt.figure(figsize=(5, 5))
    plt.errorbar(accuracy_summary['lines'], accuracy_summary['mean'], yerr=accuracy_summary['std'], fmt='o-', capsize=5, label='Pseudo-Thinking')
    plt.xlabel('Number of Lines')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Pseudo-Thinking Length')
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.savefig(os.path.join(output_dir, "accuracy_plot.pdf"))

    plt.figure(figsize=(5, 5))
    plt.plot(thinking_length_means['lines'], thinking_length_means['thinking_length'], marker='o', color='r')
    plt.xlabel('Number of Lines')
    plt.ylabel('Average Thinking Length')
    plt.title('Thinking Length vs. Pseudo-Thinking Length')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "thinking_length_plot.png"))
    plt.savefig(os.path.join(output_dir, "thinking_length_plot.pdf"))

    plt.figure(figsize=(5, 5))
    plt.plot(non_thinking_length_means['lines'], non_thinking_length_means['non_thinking_length'], marker='o', color='g')
    plt.xlabel('Number of Lines')
    plt.ylabel('Average Non-Thinking Length')
    plt.title('Non-Thinking Length vs. Pseudo-Thinking Length')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "non_thinking_length_plot.png"))
    plt.savefig(os.path.join(output_dir, "non_thinking_length_plot.pdf"))

    plt.show()
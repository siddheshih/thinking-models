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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords and punkt tokenizer if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')


def set_seed(seed_value):
    """Set seed for reproducibility."""
    if seed_value is not None and not isinstance(seed_value, int):
        raise ValueError("seed_value must be an integer or None")
    seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def extract_boxed_answer(solution_text):
    """Extracts the answer inside \\boxed{}."""
    if not solution_text:
        return None
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
    dataset, tokenizer = None, None # Initialize to None in case of errors
    try:
        dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split=split)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token
        processed_data = []
        if not dataset: # Check if dataset is empty
            print("Warning: Loaded dataset is empty.")
            return [], tokenizer # Return empty list, but tokenizer if loaded
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
        return [], None # Return empty list and None tokenizer in case of error

def extract_thinking_trace(generated_text):
    """Extracts the thinking trace."""
    if not generated_text:
        return "", 0, 0 # Handle empty generated text
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
    if not full_prompt:
        return "" # Handle empty prompt
    if inject_len <= 0 or inject_len > len(tokenizer.encode(full_prompt)):
        return "" # Handle invalid inject_len
    tokens = tokenizer.encode(full_prompt) # Encode
    start_index = randint(0, len(tokens) - inject_len)
    injected_tokens = tokens[start_index:start_index + inject_len]
    return tokenizer.decode(injected_tokens) # Decode back into a string

def count_alt_wait_words(generated_text):
    """Counts occurrences of 'alternatively' and 'wait' in generated text (case-insensitive)."""
    if not generated_text:
        return 0 # Handle empty text
    text_lower = generated_text.lower()
    count_alternatively = text_lower.count('alternatively')
    count_wait = text_lower.count('wait')
    return count_alternatively + count_wait


def identify_keywords(text, tokenizer):
    """Identifies keywords in the text (excluding stopwords)."""
    if not text:
        return [] # Handle empty text
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    keywords = [
        word for word in words
        if word.lower() not in stop_words and word.isalnum()
    ]
    return keywords


def replace_keywords(text, fraction, tokenizer, keywords):
    """Replaces a fraction of identified keywords with random tokens."""
    if not text:
        return "" # Handle empty text
    if not keywords or fraction <= 0 or fraction > 1:
        return text # Handle empty keywords or invalid fraction

    num_to_replace = int(len(keywords) * fraction)
    if num_to_replace == 0: # Handle case where no keywords to replace due to fraction
        return text
    replace_indices = sample(range(len(keywords)), num_to_replace)
    tokens = tokenizer.tokenize(text)

    keyword_to_token_indices = {}
    temp_text = text
    for keyword in keywords:
        start = temp_text.find(keyword)
        while start != -1:
            end = start + len(keyword)
            keyword_tokens = tokenizer.tokenize(temp_text[start:end])
            num_keyword_tokens = len(keyword_tokens)
            for j in range(len(tokens)):
                if tokens[j:j+num_keyword_tokens] == keyword_tokens:
                    if keyword not in keyword_to_token_indices:
                        keyword_to_token_indices[keyword]=[]
                    keyword_to_token_indices[keyword].append((j, j + num_keyword_tokens))
            temp_text = temp_text[:start] + "#"*len(keyword) + temp_text[end:]
            start = temp_text.find(keyword)

    replaced_count = 0
    for i in replace_indices:
        if replaced_count < num_to_replace:
            keyword = keywords[i]
            if keyword in keyword_to_token_indices:
                token_indices = keyword_to_token_indices[keyword]
                if token_indices:
                    start, end = token_indices.pop(0)
                    for j in range(start, end):
                        random_token_id = randint(0, tokenizer.vocab_size - 1)
                        tokens[j] = tokenizer.decode([random_token_id])
                    replaced_count += 1
    return "".join(tokens)


def replace_numbers(text, fraction, tokenizer):
    """Replaces a fraction of numbers in the text with random numbers."""
    if not text:
        return "" # Handle empty text
    numbers = re.findall(r'\d+\.?\d*', text)
    if not numbers or fraction <= 0 or fraction > 1:
        return text # Handle no numbers or invalid fraction

    num_to_replace = int(len(numbers) * fraction)
    if num_to_replace == 0: # Handle case where no numbers to replace due to fraction
        return text

    replace_indices = sample(range(len(numbers)), num_to_replace)
    tokens = tokenizer.tokenize(text)

    number_to_token_indices = {}
    temp_text = text
    for number in numbers:
        start = temp_text.find(number)
        while start != -1:
            end = start + len(number)
            number_tokens = tokenizer.tokenize(temp_text[start:end])
            num_number_tokens = len(number_tokens)
            for j in range(len(tokens)):
                if tokens[j:j+num_number_tokens] == number_tokens:
                    if number not in number_to_token_indices:
                        number_to_token_indices[number]=[]
                    number_to_token_indices[number].append((j,j+num_number_tokens))
            temp_text = temp_text[:start] + "#"*len(number) + temp_text[end:]
            start = temp_text.find(number)

    replaced_count = 0
    for i in replace_indices:
        if replaced_count < num_to_replace:
            number = numbers[i]
            if number in number_to_token_indices:
                token_indices = number_to_token_indices[number]
                if token_indices:
                    start, end = token_indices.pop(0)
                    for j in range(start, end):
                        random_token_id = randint(0, tokenizer.vocab_size-1)
                        tokens[j] = tokenizer.decode([random_token_id])
                    replaced_count+=1
    return "".join(tokens)

def run_experiment_original(processed_data, model, tokenizer, num_examples, sampling_params):
    """Runs the original experiment (no augmentation)."""
    if not processed_data or num_examples <= 0:
        print("Warning: No processed data or num_examples is non-positive.")
        return [], [] # Handle empty data gracefully
    batch_prompts = [example["input_prompt"] for example in processed_data[:num_examples]]
    results = model.generate(batch_prompts, sampling_params)
    data = []
    generated_responses = [] # Store generated responses separately

    if not results: # Handle case where model generation fails (e.g., OOM)
        print("Warning: Model generation returned no results.")
        return [], []

    for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
        generated_text = result.outputs[0].text if result and result.outputs else "" # Handle None results
        final_answer = extract_boxed_answer(generated_text)
        think_trace, think_len, non_think_len = extract_thinking_trace(generated_text)
        ground_truth = example["solution"].strip().lower()
        extracted = final_answer.strip().lower() if final_answer else ""
        correct = (extracted == ground_truth)
        accuracy = 100.0 if correct else 0.0
        alt_wait_count = count_alt_wait_words(generated_text) # Count target words

        data.append({
            'type': 'original',
            'fraction': 0.0,
            'run': 0,
            'accuracy': accuracy,
            'thinking_length': think_len,
            'non_thinking_length': non_think_len,
            'problem_id': example['problem_id'],
            'generated_response': generated_text, # Save generated text
            'alt_wait_count': alt_wait_count
        })
        generated_responses.append(generated_text) # Append generated text to list

    return data, generated_responses


def run_experiment_random_injection(processed_data, model, tokenizer, num_examples, sampling_params, thinking_prompts, thinking_prompt_fraction, run_number):
    """Runs experiment with random injection."""
    if not processed_data or not thinking_prompts or num_examples <= 0:
        print("Warning: No processed data, thinking prompts, or num_examples is non-positive.")
        return [] # Handle empty data gracefully
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

    if not results: # Handle case where model generation fails
        print("Warning: Model generation returned no results.")
        return []


    for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
        generated_text = result.outputs[0].text if result and result.outputs else "" # Handle None results
        final_answer = extract_boxed_answer(generated_text)
        think_trace, think_len, non_think_len = extract_thinking_trace(generated_text)
        ground_truth = example["solution"].strip().lower()
        extracted = final_answer.strip().lower() if final_answer else ""
        correct = (extracted == ground_truth)
        accuracy = 100.0 if correct else 0.0
        alt_wait_count = count_alt_wait_words(generated_text) # Count target words


        data.append({
            'type': 'injection',
            'fraction': thinking_prompt_fraction,  # Use consistent naming
            'run': run_number,
            'accuracy': accuracy,
            'thinking_length': think_len,
            'non_thinking_length': non_think_len,
            'problem_id': example['problem_id'],
            'generated_response': generated_text, # Save generated text
            'alt_wait_count': alt_wait_count
        })
    return data


def run_experiment_targeted_replacement(processed_data, model, tokenizer, num_examples, sampling_params, thinking_prompts, trace_fraction, replacement_fraction, replace_type, run_number):
    """Runs experiment with targeted token replacement and returns data."""
    if not processed_data or not thinking_prompts or num_examples <= 0:
        print("Warning: No processed data, thinking prompts, or num_examples is non-positive.")
        return [] # Handle empty data
    batch_prompts = []
    for i, example in enumerate(processed_data[:num_examples]):
        prompt_to_inject = thinking_prompts[i]
        inject_len = int(len(prompt_to_inject) * trace_fraction)
        injected_prompt = prompt_to_inject
        if replace_type == "keywords":
            keywords = identify_keywords(injected_prompt, tokenizer)
            corrupted_prompt = replace_keywords(injected_prompt, replacement_fraction, tokenizer, keywords)
        elif replace_type == "numbers":
            corrupted_prompt = replace_numbers(injected_prompt, replacement_fraction, tokenizer)
        else:
            raise ValueError("Invalid replace_type.")

        input_prompt = (
            f"AIME Problem {example['problem_number']} ({example['year']}):\n{example['problem']}\n\n"
            f"Solution:\n<think>\n{corrupted_prompt}\n</think>\n" # Added closing </think> tag
        )
        batch_prompts.append(input_prompt)

    results = model.generate(batch_prompts, sampling_params)
    data = []
    if not results: # Handle case where model generation fails
        print("Warning: Model generation returned no results.")
        return []

    for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
        generated_text = result.outputs[0].text if result and result.outputs else "" # Handle None results
        final_answer = extract_boxed_answer(generated_text)
        think_trace, think_len, non_think_len = extract_thinking_trace(generated_text)
        ground_truth = example["solution"].strip().lower()
        extracted = final_answer.strip().lower() if final_answer else ""
        correct = (extracted == ground_truth)
        accuracy = 100.0 if correct else 0.0
        alt_wait_count = count_alt_wait_words(generated_text) # Count target words
        data.append({
            'type': replace_type,
            'fraction': replacement_fraction,
            'run': run_number,
            'accuracy': accuracy,
            'thinking_length': think_len,
            'non_thinking_length': non_think_len, # Corrected variable name here!
            'problem_id': example['problem_id'],
            'generated_response': generated_text,
            'alt_wait_count': alt_wait_count
        })

    return data


# --- Main Experiment Execution ---
if __name__ == '__main__':
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    processed_data, tokenizer = load_and_preprocess_aime_dataset(model_name, split="train")
    if not processed_data: # Check if processed_data is empty after loading failed or dataset was empty
        raise ValueError("Dataset loading failed or no data processed.")
    if not tokenizer: # Check if tokenizer is None
        raise ValueError("Tokenizer loading failed.")

    set_seed(42)
    model = LLM(model=model_name, max_model_len=32000, tensor_parallel_size=4)
    temperature = 0.6
    top_p = 0.95
    max_new_tokens = 16000
    num_examples = 200
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
            # --- Save CSV after each Random Injection run ---
            df = pd.DataFrame(all_data)
            output_csv_path = "experiment_results_with_responses.csv"
            df.to_csv(output_csv_path, index=False)
            print(f"    Results saved to: {output_csv_path}")


    # --- Experiment 4: Targeted Replacement (Keywords and Numbers) ---
    print("\nRunning Experiment 4: Targeted Replacement")
    trace_fraction = 1.0
    replacement_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    replace_types = ["keywords", "numbers"]
    num_runs = 3

    for replace_type in replace_types:
        print(f"  Replacement Type: {replace_type}")
        for replacement_fraction in replacement_fractions:
            print(f"    Replacement Fraction: {replacement_fraction}")
            for run in range(num_runs):
                set_seed(42 + run)  # Different seed for each run
                replacement_data = run_experiment_targeted_replacement(
                    processed_data, model, tokenizer, num_examples, sampling_params,
                    thinking_prompts_exp1, trace_fraction, replacement_fraction, replace_type, run + 1
                )
                all_data.extend(replacement_data)
                print(f"      Run {run + 1}: Completed")
                # --- Save CSV after each Targeted Replacement run ---
                df = pd.DataFrame(all_data)
                output_csv_path = "experiment_results_with_responses.csv" # Use the same CSV path to append
                df.to_csv(output_csv_path, index=False)
                print(f"      Results saved to: {output_csv_path}")


    # --- Plotting ---
    show_error_bars = True  # Set to True to show error bars, False to remove them

    for replace_type in replace_types:
        # Filter data for the current replacement type
        filtered_df = df[df['type'] == replace_type]

        plt.figure(figsize=(15, 5))

        # Accuracy Plot
        plt.subplot(1, 3, 1)
        if show_error_bars:
            accuracy_summary = filtered_df.groupby('fraction')['accuracy'].agg(['mean', 'std']).reset_index()
            plt.errorbar(accuracy_summary['fraction'], accuracy_summary['mean'], yerr=accuracy_summary['std'], fmt='o-', capsize=5)
        else:
            accuracy_means = filtered_df.groupby('fraction')['accuracy'].mean().reset_index()
            plt.plot(accuracy_means['fraction'], accuracy_means['accuracy'], marker='o')

        plt.xlabel('Replacement Fraction')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy vs. Replacement Fraction ({replace_type})')
        plt.grid(True)

        # Average Thinking Length Plot
        plt.subplot(1, 3, 2)
        thinking_length_means = filtered_df.groupby('fraction')['thinking_length'].mean().reset_index()
        plt.plot(thinking_length_means['fraction'], thinking_length_means['thinking_length'], marker='o', color='r')
        plt.xlabel('Replacement Fraction')
        plt.ylabel('Average Thinking Length')
        plt.title(f'Thinking Length vs. Replacement Fraction ({replace_type})')
        plt.grid(True)

        # Average Non-Thinking Length Plot
        plt.subplot(1, 3, 3)
        non_thinking_length_means = filtered_df.groupby('fraction')['non_thinking_length'].mean().reset_index()
        plt.plot(non_thinking_length_means['fraction'], non_thinking_length_means['non_thinking_length'], marker='o', color='g')
        plt.xlabel('Replacement Fraction')
        plt.ylabel('Average Non-Thinking Length')
        plt.title(f'Non-Thinking Length vs. Replacement Fraction ({replace_type})')
        plt.grid(True)

        plt.tight_layout()

        # --- Saving ---
        output_dir_targeted = f"final_plot_results_targeted_{replace_type}_2"
        if not os.path.exists(output_dir_targeted):
            os.makedirs(output_dir_targeted)
        plt.savefig(os.path.join(output_dir_targeted, f"combined_plots_targeted_{replace_type}.png"))


    # --- Plotting Alternative/Wait Word Count ---
    plt.figure(figsize=(5, 5))
    alt_wait_summary = df[df['type'] == 'injection'].groupby('fraction')['alt_wait_count'].mean().reset_index()
    plt.plot(alt_wait_summary['fraction'], alt_wait_summary['alt_wait_count'], marker='o', linestyle='-')
    plt.xlabel('Injection Fraction')
    plt.ylabel('Average Count of "alternatively" and "wait"')
    plt.title('Word Count vs. Injection Fraction')
    plt.grid(True)

    # --- Saving Word Count Plot ---
    output_dir_word_count = "plot_results_word_count_2"
    os.makedirs(output_dir_word_count, exist_ok=True)
    plt.savefig(os.path.join(output_dir_word_count, "alt_wait_word_count_plot.png"))
    plt.savefig(os.path.join(output_dir_word_count, "alt_wait_word_count_plot.pdf"))


    plt.show()
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from random import seed, randint, sample
import re
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd  # Import pandas

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')

# Download necessary NLTK data (only needed once)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab/english.pickle')
except LookupError:
    nltk.download('punkt_tab')


def set_seed(seed_value):
    """Set seed for reproducibility."""
    seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def extract_boxed_answer(solution_text):
    """Extracts the answer inside \\boxed{}."""
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


def identify_keywords(text, tokenizer):
    """Identifies keywords in the text (excluding stopwords)."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    keywords = [
        word for word in words
        if word.lower() not in stop_words and word.isalnum()
    ]
    return keywords


def replace_keywords(text, fraction, tokenizer, keywords):
    """Replaces a fraction of identified keywords with random tokens."""
    if not keywords:
        return text

    num_to_replace = int(len(keywords) * fraction)
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
    numbers = re.findall(r'\d+\.?\d*', text)
    if not numbers:
        return text

    num_to_replace = int(len(numbers) * fraction)
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
    """Runs the original experiment (no augmentation) and returns generated responses."""
    batch_prompts = [example["input_prompt"] for example in processed_data[:num_examples]]
    results = model.generate(batch_prompts, sampling_params)
    data = []
    generated_texts = []  # List to store generated texts

    for idx, (example, result) in enumerate(zip(processed_data[:num_examples], results)):
        generated_text = result.outputs[0].text
        generated_texts.append(generated_text) # Append to generated_texts list
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
            'problem_id': example['problem_id'],
            'generated_response': generated_text # Save generated text
        })

    return data, generated_texts # Return generated_texts


def run_experiment_targeted_replacement(processed_data, model, tokenizer, num_examples, sampling_params, thinking_prompts, trace_fraction, replacement_fraction, replace_type, run_number):
    """Runs experiment with targeted token replacement and returns data."""
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
            f"Solution:\n<think>\n{corrupted_prompt}"
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
            'type': replace_type,
            'fraction': replacement_fraction,
            'run': run_number,
            'accuracy': accuracy,
            'thinking_length': think_len,
            'non_thinking_length': non_thinking_len,
            'problem_id': example['problem_id'],
            'generated_response': generated_text # Save generated text
        })

    return data


# --- Main Experiment Execution ---
if __name__ == '__main__':
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    processed_data, tokenizer = load_and_preprocess_aime_dataset(model_name, split="train")
    if processed_data is None:
        raise ValueError("Dataset loading failed.")
    set_seed(42)
    model = LLM(model=model_name, max_model_len=32000, tensor_parallel_size=2)
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

    # --- Data Aggregation and CSV Export ---
    df = pd.DataFrame(all_data)
    output_csv_path = "experiment_results_with_responses.csv" # Changed output CSV file name
    df.to_csv(output_csv_path, index=False)
    print(f"Experiment results saved to: {output_csv_path}")


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
        plt.show()
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import gc
from scipy.stats import ttest_ind
import numpy as np
import re  # For parsing the solution steps
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt
from random import seed, randint
import os

class MultiStepForceTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, forced_tokens: dict):
        """
        forced_tokens: dict mapping generation step (int) to forced token id (int)
        """
        self.forced_tokens = forced_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_step = input_ids.shape[1]
        if current_step in self.forced_tokens:
            token_id = self.forced_tokens[current_step]
            forced_scores = torch.full_like(scores, -float("inf"))
            forced_scores[:, token_id] = 0  # Force this token.
            return forced_scores
        return scores


# Load the model and tokenizer.
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optional: If using PyTorch 2.0, compile the model for further speed improvements.
# Uncomment the following line if your environment supports it.
# model = torch.compile(model)

forced_token1 = "\n"
forced_token2 = "</think>"

# Encode the tokens. This approach works even if the raw token string isn't found directly.
forced_token1_ids = tokenizer.encode(forced_token1, add_special_tokens=False)
forced_token2_ids = tokenizer.encode(forced_token2, add_special_tokens=False)

if len(forced_token1_ids) != 1:
    raise ValueError("The forced token for newline is encoded as multiple tokens; adjust your approach.")
forced_token1_id = forced_token1_ids[0]

if len(forced_token2_ids) != 1:
    raise ValueError("The forced token for '</think>' is encoded as multiple tokens; adjust your approach.")
forced_token2_id = forced_token2_ids[0]

# Define the generation steps at which to force these tokens.
# For example, force forced_token1 at step 10 and forced_token2 at step 11.
force_step = 250
forced_tokens = {
    force_step: forced_token1_id,
    force_step + 1: forced_token2_id
}
logits_processor = LogitsProcessorList([MultiStepForceTokenLogitsProcessor(forced_tokens)])


# Define the prompt and move it to the same device.
prompt = "Solve the prblem: 2+100023. Enclose the answer is {}\n <think>"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate text using FP16 autocasting if running on GPU.
if device.type == "cuda":
    context_manager = torch.cuda.amp.autocast()
else:
    context_manager = torch.no_grad()

with context_manager:
    outputs = model.generate(
        input_ids,
        max_length=500,
        logits_processor=logits_processor,
        do_sample=True,  # Enables sampling; remove if you prefer greedy decoding.
        use_cache=True   # Ensures cached activations are used.
    )

# Decode and print the generated text.
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
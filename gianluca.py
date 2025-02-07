import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import torch.nn.functional as F


# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model pathsd
base_name = "meta-llama/Llama-3.2-1B"
instruct_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(base_name)
base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map=device)
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_name, device_map=device)

# Interpolation factor (alpha): 0.5 means equal weighting
alphas = np.arange(0, 1, 0.01)

for alpha in alphas:
    model = deepcopy(base_model).to(device)
    # Interpolate weights
    for param_base, param_instruct, param_model in zip(base_model.parameters(), instruct_model.parameters(), model.parameters()):
        param_model.data = ((1 - alpha) * param_base.data + alpha * param_instruct.data).to(device)

    # Storytelling query
    story_query = "Tell a 100-word story about a cat that finds a lost treasure in an abandoned lighthouse."

    # Tokenize the input
    inputs = tokenizer(story_query, return_tensors="pt").to(device)

    # Generate 100 stories for this alpha
    for story_idx in range(100):
        # Calculate next-word entropy
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[:, -1, :]  # Next word logits
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs), dim=-1).item()

        # Generate a story response
        output_ids = model.generate(
            **inputs,
            max_length=200,
            temperature=1.0,
        )

        # Decode the response
        prompt_length = inputs["input_ids"].shape[1]
        generated_story_ids = output_ids[0][prompt_length:]
        story = tokenizer.decode(generated_story_ids, skip_special_tokens=True)

        # Calculate unique words ratio
        words = story.split()
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words) if words else 0

        # Output the metrics for this story
        print(f"Alpha: {alpha:.2f}, Story {story_idx + 1}, Next-word entropy: {entropy:.2f}, Unique words ratio: {unique_ratio:.2f}")

        # Save the story to a file
        alpha_dir = os.path.join("./", f"alpha_{alpha:.2f}")
        os.makedirs(alpha_dir, exist_ok=True)
        story_file = os.path.join(alpha_dir, f"story_{story_idx + 1}.txt")
        with open(story_file, "w") as f:
            f.write(story)
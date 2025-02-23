import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import torch.nn.functional as F


# Model pathsd
base_name = "meta-llama/Llama-3.1-8B"
instruct_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(base_name, device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto")
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_name, device_map="auto")
tokenizer.pad_token = tokenizer.eos_tokenw
# Interpolation factor (alpha): 0.5 means equal weighting
alphas = np.arange(0, 1, 0.1)

# Add batch size parameter
BATCH_SIZE = 128  # Adjust based on your GPU memory

for alpha in alphas:
    model = deepcopy(base_model)

    # Interpolate weights
    for param_base, param_instruct, param_model in zip(base_model.parameters(), instruct_model.parameters(), model.parameters()):
        param_model.data = ((1 - alpha) * param_base.data  + alpha * param_instruct.data) 

    # Storytelling query
    story_query = "Tell a 100-word story about a cat that finds a lost treasure in an abandoned lighthouse."

    # Create batched inputs
    inputs = tokenizer(
        [story_query] * BATCH_SIZE, 
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Generate stories in batches
    for batch_idx in range(0, 100, BATCH_SIZE):
        # Calculate the actual batch size for the last batch
        current_batch_size = min(BATCH_SIZE, 100 - batch_idx)
        if current_batch_size != BATCH_SIZE:
            inputs = tokenizer(
                [story_query] * current_batch_size, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

        # Skip stories that already exist
        stories_to_generate = []
        story_indices = []
        for i in range(current_batch_size):
            story_idx = batch_idx + i
            alpha_dir = os.path.join("./", f"alpha_{alpha:.2f}")
            story_file = os.path.join(alpha_dir, f"llama_8b_story_{story_idx + 1}.txt")
            if not os.path.exists(story_file):
                stories_to_generate.append(i)
                story_indices.append(story_idx)

        if not stories_to_generate:
            continue

        # Calculate next-word entropy for the batch
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[:, -1, :]  # Next word logits
        probs = F.softmax(logits, dim=-1)
        entropies = -torch.sum(probs * torch.log2(probs), dim=-1)

        # Generate stories for the batch
        output_ids = model.generate(
            **inputs,
            max_length=200,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Process each story in the batch
        prompt_length = inputs["input_ids"].shape[1]
        for batch_pos, story_idx in zip(stories_to_generate, story_indices):
            generated_story_ids = output_ids[batch_pos][prompt_length:]
            story = tokenizer.decode(generated_story_ids, skip_special_tokens=True)

            # Calculate unique words ratio
            words = story.split()
            unique_words = set(words)
            unique_ratio = len(unique_words) / len(words) if words else 0

            # Output the metrics for this story
            print(f"Alpha: {alpha:.2f}, Story {story_idx + 1}, Next-word entropy: {entropies[batch_pos]:.2f}, Unique words ratio: {unique_ratio:.2f}")

            # Save the story to a file
            alpha_dir = os.path.join("./", f"alpha_{alpha:.2f}")
            os.makedirs(alpha_dir, exist_ok=True)
            story_file = os.path.join(alpha_dir, f"llama_8b_story_{story_idx + 1}.txt")
            with open(story_file, "w") as f:
                f.write(story)

        # Clear CUDA cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
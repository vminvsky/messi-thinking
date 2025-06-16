import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import torch.nn.functional as F
import json

import gc

# Get HF_HOME directory
hf_home = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
mixed_models_dir = os.path.join(hf_home, 'mixed_models')
os.makedirs(mixed_models_dir, exist_ok=True)

# Check if GPU is available and set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
base_name = "meta-llama/Llama-3.1-8B"
instruct_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Load tokenizer and models
save_only_tokenizers = False  # Set to False to save both models and tokenizers

base_tokenizer = AutoTokenizer.from_pretrained(base_name)
instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_name)
if save_only_tokenizers:
    base_model = None
    instruct_model = None
else:
    base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto")
    instruct_model = AutoModelForCausalLM.from_pretrained(instruct_name, device_map="auto")

# Create mixed models at different interpolation points
alphas = [0.3, 0.5, 0.7,0.9]  # Example interpolation points
# alphas = [0.7,0.9]  # Example interpolation points

# Add flag at the top

for alpha in alphas:
    print(f"Creating mixed model with alpha={alpha}")
    
    # Create directory for this mixed model
    model_dir = os.path.join(mixed_models_dir, f'llama-3.1-8b-mixed-{alpha:.2f}')
    
    if not save_only_tokenizers:
        # Create a new model by interpolating weights
        mixed_model = deepcopy(instruct_model)  # This will inherit the device_map from base_model
        
        for param_base, param_instruct, param_mixed in zip(
            base_model.parameters(), 
            instruct_model.parameters(), 
            mixed_model.parameters()
        ):
            param_mixed.data = (1 - alpha) * param_base.data + alpha * param_instruct.data
        
        os.makedirs(model_dir, exist_ok=True)
        mixed_model.save_pretrained(model_dir)
        
        # Save config file with interpolation information
        config = mixed_model.config
        config.interpolation = {
            'base_model': base_name,
            'instruct_model': instruct_name,
            'alpha': alpha
        }
        config.save_pretrained(model_dir)
        
        # Clear CUDA cache and delete the mixed model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del mixed_model
        gc.collect()
    
    # Save tokenizers regardless of the flag
    print(f"Saving tokenizers to {model_dir}")
    
    # Save tokenizers in the main directory and in separate folders
    base_tokenizer.save_pretrained(model_dir)  # Save base tokenizer files in main dir
    base_tokenizer.save_pretrained(os.path.join(model_dir, 'base_tokenizer'))
    instruct_tokenizer.save_pretrained(os.path.join(model_dir, 'instruct_tokenizer'))
    
    # Update the tokenizer_config.json to include information about both tokenizers
    tokenizer_config = {
        "base_tokenizer_name": base_name,
        "instruct_tokenizer_name": instruct_name,
        "base_tokenizer_path": "base_tokenizer",
        "instruct_tokenizer_path": "instruct_tokenizer"
    }
    
    with open(os.path.join(model_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"Saved tokenizers to {model_dir}")

# Update the test_generation function to use both tokenizers
def test_generation(model_path):
    print(f"\nTesting generation with model from {model_path}")
    
    # Load both tokenizers
    base_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'base_tokenizer'))
    instruct_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'instruct_tokenizer'))
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    try:
        story_query = "Tell a 100-word story about a cat that finds a lost treasure in an abandoned lighthouse."
        # Test with both tokenizers
        for tokenizer_name, tokenizer in [("base", base_tokenizer), ("instruct", instruct_tokenizer)]:
            print(f"\nTesting with {tokenizer_name} tokenizer:")
            inputs = tokenizer(story_query, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            story = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Generated Story:\n", story)
    
    finally:
        # Clean up after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model
        gc.collect()

from transformers import AutoModelForCausalLM, AutoTokenizer
from dynamic_decoder import DynamicTemperatureDecoder
from config import DynamicDecodingConfig, DecayType, SimilarityType
from datasets import load_from_disk
from tqdm import tqdm
import json
import os
from prompts import REASONING_SYSTEM_PROMPT

# Load dataset
ds = load_from_disk("data/1k_taco_train")

# Initialize model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

# Create config
config = DynamicDecodingConfig(
    similarity_threshold=0.98,
    verbose_generation=True,
    cache_embeddings=False,
)

# Initialize decoder
decoder = DynamicTemperatureDecoder(
    model=model,
    tokenizer=tokenizer,
    config=config,
)

OUTPUT_DIR = f"samples/1k_taco_train/{model_name.replace('/', '_')}"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, sample in enumerate(tqdm(ds, desc="Processing inputs")):
    # Parse sample metadata
    sample["solutions"] = json.loads(sample["solutions"])
    sample["input_output"] = json.loads(sample["input_output"])
    sample["raw_tags"] = eval(sample["raw_tags"])
    sample["tags"] = eval(sample["tags"])
    sample["skill_types"] = eval(sample["skill_types"])
    
    question = sample["question"]
    
    # Create question directory
    question_dir = os.path.join(OUTPUT_DIR, f"question_{idx}")
    os.makedirs(question_dir, exist_ok=True)
    
    # Clear reference thoughts for new question
    decoder.reference_thoughts = []
    decoder.reference_embeddings = None
    
    # Generate 8 samples
    for i in range(8):
        result = decoder.generate(REASONING_SYSTEM_PROMPT, question, max_new_tokens=4096)
        
        # Save sample with metadata
        sample_data = {
            "metadata": sample,
            "generation": {
                "text": result["generated_text"],
                "temperature_history": result["temperature_history"],
                "similarity_history": result["similarity_history"],
                "reference_thoughts": decoder.reference_thoughts.copy(),
            }
        }
        
        # Save to file
        sample_path = os.path.join(question_dir, f"sample_{i}.json")
        with open(sample_path, "w") as f:
            json.dump(sample_data, f, indent=2)
            
        # Add to reference thoughts for next generation
        decoder.add_reference_thoughts([result["generated_text"]])
        
    
    
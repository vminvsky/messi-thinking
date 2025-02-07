import os
import json

from vllm import LLM, SamplingParams
from datasets import load_dataset
from dynamic_scaling.prompt import SKY_T1_SYSTEM_PROMPT

# Parameters
K = 100  # tokens for base model generation per turn
P = 100  # tokens for instruction model generation per turn
NUM_SAMPLES = 1  # samples per dataset entry
OUTPUT_DIR = "generated_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset and filter by difficulty
_ds = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")

# Instantiate models
base_model = LLM(model="Qwen/Qwen2-1.5B", gpu_memory_utilization=0.4)
instruct_model = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", gpu_memory_utilization=0.4)

# Sampling parameters
base_sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=K)
instruct_sampling_params = SamplingParams(temperature=0.5, max_tokens=P)

# For each sample in the dataset
for idx, sample in enumerate(_ds):
    prompt = sample.get("question", sample.get("text", ""))
    if not prompt:
        continue
    for sample_num in range(NUM_SAMPLES):
        current_text = ""
        round_num = 0
        while True:
            if round_num % 2 == 0:
                # Base model generation turn
                input_text = SKY_T1_SYSTEM_PROMPT + "\n" + prompt + "\n" + current_text if current_text else prompt
                if round_num == 0:
                    input_text += " Solution: "
                
                print(f"\n\nBase model generation turn {round_num}")
                print(input_text)
                outputs = base_model.generate([input_text], sampling_params=base_sampling_params)
                generated = outputs[0].outputs[0].text
                
                print("\n\nGenerated:")
                print(generated)
                if not generated.strip():
                    break
                current_text += generated.replace("</think>", "[end_of_thought]").replace("<think>", "[begin_of_thought]")
            else:
                # Instruction model chat turn; complete the last message
                conversation = [
                    {"role": "system", "content": SKY_T1_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": current_text}
                ]
                print(f"\n\nInstruction model chat turn {round_num}")
                print(current_text)
                outputs = instruct_model.chat(conversation, sampling_params=instruct_sampling_params, use_tqdm=False, continue_final_message=True, add_generation_prompt=False)
                generated = outputs[0].outputs[0].text
                print("\n\nGenerated:")
                print(generated)
                if not generated.strip():
                    break
                current_text += generated.replace("</think>", "[end_of_thought]").replace("<think>", "[begin_of_thought]")
            round_num += 1
        
        # Save the generated output and metadata
        output_data = {
            "prompt": prompt,
            "generated_text": current_text,
            "metadata": sample,
            "sample_index": sample_num
        }
        output_filename = os.path.join(OUTPUT_DIR, f"sample_{idx}_{sample_num}.json")
        with open(output_filename, "w") as f:
            json.dump(output_data, f, indent=2) 
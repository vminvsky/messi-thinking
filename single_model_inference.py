import os
import json
import asyncio
import logging
import argparse

from datasets import load_dataset
from tqdm import tqdm
from dynamic_scaling.prompt import SKY_T1_FIXED, BASE_MODEL_SYSTEM_PROMPT, generate_prompt
from openai import AsyncOpenAI
from vllm import SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
MAX_TOKENS = 8192
NUM_SAMPLES = 10  # samples per dataset entry

# NOTE: CHANGE
# OUTPUT_DIR = "llama-3.1-8b"
OUTPUT_DIR = "taco_instruct_llama_8b_single_slerp_0.5/"
# OUTPUT_DIR = 'taco_instruct_llama_8b_single_slerp_0.90'
# OUTPUT_DIR = 

TEMPERATURE = 1.2 

# NOTE: CHANGE
use_slerp = True
# Models
merge_frac = "0.50"
base_model = "/scratch/gpfs/vv7118/models/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

if use_slerp:
    instruct_model = f"/scratch/gpfs/vv7118/models/mixed_models/llama-3.1-8b-mixed-slerp-{merge_frac}"
else:
    instruct_model = f"/scratch/gpfs/vv7118/models/mixed_models/llama-3.1-8b-mixed-{merge_frac}"

# instruct_model = 
instruct_model = "/scratch/gpfs/vv7118/models/mixed_models/llama-3.1-8b-mixed-0.70/"

# Use async inference with OpenAI client
USE_OPENAI = True


def get_prompt(sample):
    """Parse test cases and starter code from problem to create a prompt for the LLM."""
    test_case = json.loads(sample["input_output"])
    starter_code = sample["starter_code"]
    prompt_text = generate_prompt(test_case, sample["question"], starter_code)
    return [{"role": "system", "content": SKY_T1_FIXED}, {"role": "user", "content": prompt_text}]

def get_prompt_instruct(sample):
    return get_prompt(sample)
    
def get_prompt_base(sample):
    prompt = get_prompt(sample)
    return BASE_MODEL_SYSTEM_PROMPT + prompt[1]["content"] + "\nAssistant:\n[begin_of_thought]"

async def perform_base_inference(input_text):
    completion = await base_client.completions.create(
        model=base_model,
        prompt=input_text,
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.95)
    generated = completion.choices[0].text.replace("</think>", "[end_of_thought]").replace("<think>", "[begin_of_thought]")
    return generated

async def perform_instruct_inference(conversation):
    chat_completion = await instruct_client.chat.completions.create(
        model=instruct_model,
        messages=conversation,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.7)
    generated = chat_completion.choices[0].message.content.replace("</think>", "[end_of_thought]").replace("<think>", "[begin_of_thought]")
    return generated

async def process_sample(idx, sample, sample_num, prompt, output_filename, model_flag):
    if model_flag == "base":
        input_text = get_prompt_base(sample)
        generated = await perform_base_inference(input_text)
    else:
        conversation = get_prompt_instruct(sample)
        generated = await perform_instruct_inference(conversation)
    output_data = {
        "prompt": prompt,
        "generated_text": generated,
        "metadata": sample,
        "question_id": idx,
        "sample_index": sample_num,
        "generation_config": {
            "base_model": base_model,
            "instruct_model": instruct_model,
            "base_temperature": TEMPERATURE,
            "base_top_p": 0.95,
            "base_top_k": 0,
            "base_max_tokens": MAX_TOKENS,
            "instruct_temperature": TEMPERATURE,
            "instruct_top_p": 0.7,
            "instruct_top_k": 0,
            "instruct_max_tokens": MAX_TOKENS
        }
    }
    with open(output_filename, "w") as f:
        json.dump(output_data, f, indent=2)

async def limited_process_sample(semaphore, idx, sample, sample_num, prompt, output_filename, model_flag):
    async with semaphore:
        await process_sample(idx, sample, sample_num, prompt, output_filename, model_flag)

async def main(model_flag, begin_idx):
    semaphore = asyncio.Semaphore(100)
    tasks = []
    ds = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
    for idx, sample in tqdm(enumerate(ds), desc="Processing samples"):
        if idx > 1000:
            break

        if idx < begin_idx:
            continue
        for sample_num in range(NUM_SAMPLES):
            output_filename = os.path.join(OUTPUT_DIR, f"question_{idx}_sample_{sample_num}.json")
            if os.path.exists(output_filename):
                logger.info(f"Output file {output_filename} exists, skipping sample {sample_num} for question_{idx}")
                continue
            tasks.append(asyncio.create_task(limited_process_sample(semaphore, idx, sample, sample_num, prompt, output_filename, model_flag)))
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Waiting for tasks"):
        await task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "instruct"], required=True, help="Select model for inference (base or instruct)")
    parser.add_argument("--instruct_model", required=True, help="Path to instruct model")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens for generation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per dataset entry")
    parser.add_argument("--output_dir", required=True, help="Output directory for generated files")
    parser.add_argument("--use_slerp", dest="use_slerp", action="store_true", help="Enable slerp (default)")
    parser.add_argument("--no_slerp", dest="use_slerp", action="store_false", help="Disable slerp")
    parser.set_defaults(use_slerp=True)
    parser.add_argument("--begin_idx", type=int, default=0, help="Begin index for processing samples")
    parser.add_argument("--merge_frac", default="0.50", help="Merge fraction for instruct model")
    parser.add_argument("--base_model", default="models/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", help="Path to base model")
    parser.add_argument("--openai_api_key", default="token-abc123", help="API key for OpenAI client")
    parser.add_argument("--openai_base_url", default="http://localhost:8000/v1", help="Base URL for OpenAI API")
    parser.add_argument("--base_temperature", type=float, default=0.8, help="Temperature for base model")
    parser.add_argument("--base_top_p", type=float, default=0.95, help="Top-p for base model")
    parser.add_argument("--instruct_temperature", type=float, default=0.7, help="Temperature for instruct model")
    parser.add_argument("--instruct_top_p", type=float, default=0.7, help="Top-p for instruct model")
    parser.add_argument("--port", type=int, default=8000, help="Port for VLLM server")
    args = parser.parse_args()

    MAX_TOKENS = args.max_tokens
    NUM_SAMPLES = args.num_samples
    OUTPUT_DIR = args.output_dir
    use_slerp = args.use_slerp
    merge_frac = args.merge_frac
    base_model = args.base_model
    instruct_model = args.instruct_model
    base_temperature = args.base_temperature
    base_top_p = args.base_top_p
    instruct_temperature = args.instruct_temperature
    instruct_top_p = args.instruct_top_p
    port = args.port
    
    base_client = AsyncOpenAI(api_key="token-abc123", base_url=f"http://localhost:{port}/v1")
    instruct_client = AsyncOpenAI(api_key="token-abc123", base_url=f"http://localhost:{port}/v1")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    asyncio.run(main(args.model, args.begin_idx)) 
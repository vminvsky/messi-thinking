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
NUM_SAMPLES = 6  # samples per dataset entry

# Models
base_model = "meta-llama/Llama-3.1-8B"
instruct_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Use async inference with OpenAI client
USE_OPENAI = True


base_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)
instruct_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)

def get_prompt_instruct(sample):
    test_case = json.loads(sample["input_output"])
    starter_code = sample["starter_code"]
    prompt_text = generate_prompt(test_case, sample["question"], starter_code)
    return [
        {"role": "system", "content": SKY_T1_FIXED},
        {"role": "user", "content": prompt_text}
    ]
    
def get_prompt_base(sample):
    test_case = json.loads(sample["input_output"])
    starter_code = sample["starter_code"]
    prompt_text = generate_prompt(test_case, sample["question"], starter_code)
    return BASE_MODEL_SYSTEM_PROMPT.format(Question=prompt_text)

async def perform_base_inference(input_text):
    
    print(input_text)
    completion = await asyncio.to_thread(
        base_client.completions.create,
        model=base_model,
        prompt=input_text,
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.95
    )
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
        "generated_text": "[begin_of_thought]" + generated,
        "metadata": sample,
        "question_id": idx,
        "sample_index": sample_num,
        "generation_config": {
            "base_model": base_model,
            "instruct_model": instruct_model,
            "base_temperature": 0.8,
            "base_top_p": 0.95,
            "base_top_k": 0,
            "base_max_tokens": MAX_TOKENS,
            "instruct_temperature": 0.7,
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

async def main(model_flag):
    semaphore = asyncio.Semaphore(1)
    tasks = []
    ds = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
    for idx, sample in tqdm(enumerate(ds), desc="Processing samples"):
        prompt = get_prompt_base(sample) if model_flag == "base" else get_prompt_instruct(sample)
        if not prompt:
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
    args = parser.parse_args()
    MODEL_FLAG = args.model
    if MODEL_FLAG == "base":
        OUTPUT_DIR = "taco_base_llama_8b_single"
    else:
        OUTPUT_DIR = "taco_instruct_llama_8b_single"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    asyncio.run(main(MODEL_FLAG)) 
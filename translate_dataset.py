import os
import json
import argparse
import asyncio
from datasets import load_dataset, Dataset, load_from_disk
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv(override=True)

async def litellm_translate(text, target_lang, semaphore):
    async with semaphore:
        prompt = f"Translate the following text to {target_lang}:\n\n{text}"
        translation = await acompletion(model="gpt-4o", messages=[
            {"role": "system", "content": "You are a helpful translator. You return the translated text only."},
            {"role": "user", "content": prompt}
        ])
        return translation["choices"][0]["message"]["content"].strip()

async def translate_sample(sample, target_langs, semaphore):
    question_text = sample["question"]
    translations = {}
    tasks = {}
    for lang in target_langs:
        key = f"question_{lang}"
        if key in sample:
            print(f"Skipping {lang} because it already exists")
            continue
        tasks[lang] = asyncio.create_task(litellm_translate(question_text, lang, semaphore))
    if tasks:
        results = await asyncio.gather(*tasks.values())
        for lang, result in zip(tasks.keys(), results):
            translations[f"question_{lang}"] = result
    return translations

async def add_translations(sample, target_langs, semaphore):
    translations = await translate_sample(sample, target_langs, semaphore)
    sample.update(translations)
    return sample

async def main_async(languages, output_dir):
    target_langs = [lang.strip() for lang in languages.split(",")]
    dataset = load_dataset("BAAI/TACO", trust_remote_code=True)
    ds_train = dataset["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
    ds_test = dataset["test"].filter(lambda x: x["difficulty"] == "MEDIUM")
    semaphore = asyncio.Semaphore(50)
    updated_train = await asyncio.gather(*[add_translations(sample, target_langs, semaphore) for sample in ds_train])
    updated_test = await asyncio.gather(*[add_translations(sample, target_langs, semaphore) for sample in ds_test])
    new_train = Dataset.from_list(updated_train)
    new_test = Dataset.from_list(updated_test)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    new_train.save_to_disk(os.path.join(output_dir, "train"))
    new_test.save_to_disk(os.path.join(output_dir, "test"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", type=str, required=True, help="Comma separated list of target languages (e..g German,Russian)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the updated dataset")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    asyncio.run(main_async(args.languages, args.output_dir)) 

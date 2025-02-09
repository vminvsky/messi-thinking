import argparse
import json
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Convert JSON data for processing.")
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing input JSON files."
    )
    parser.add_argument("--output", type=str, help="Output JSON file.")
    args = parser.parse_args()

    all_data = []

    # Iterate through all files in the input directory
    for filename in tqdm(os.listdir(args.input_dir), desc="Processing files"):
        if filename.endswith(".json") and filename.startswith("converted_"):
            filepath = os.path.join(args.input_dir, filename)
            with open(filepath, "r") as f:
                sample = json.load(f)

            system_prompt = sample["prompt"][0]["content"]
            user_prompt = sample["prompt"][1]["content"]
            assistant_response = sample["converted_text"]

            # Accept this data
            if sample["correctness"]:
                # Create the conversation format
                conversations = [
                    {"from": "user", "value": user_prompt},
                    {"from": "assistant", "value": assistant_response},
                ]

                # Prepare the final structure
                cur_data = {
                    "system": system_prompt,
                    "conversations": conversations,
                }
                all_data.append(cur_data)

    # Save the converted data to the output file
    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=4)

    # Print number of unique questions based on unique user prompts
    unique_questions = len({data["conversations"][0]["value"] for data in all_data})
    print(f"Number of unique questions: {unique_questions}")

    print(
        f"Conversion completed. The data has been saved to {args.output} with {len(all_data)} data."
    )

if __name__ == "__main__":
    main()
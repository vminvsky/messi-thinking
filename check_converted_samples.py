import os
import json
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

from skythought_evals.tasks.taco.taco_handler import TACOTaskHandler

def process_file(filename, directory, handler):
    """Process a single file and return the results"""
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Skip if already processed
        if "correctness" in data and "correctness_reason" in data:
            return filename, "Already processed"
        
        problem = data.get("metadata", {})
        response = data.get("converted_text", "")
        result = handler.update_results(problem, response)
        
        data["correctness"] = result["correctness"]
        data["correctness_reason"] = result["reason"]
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        return filename, result['correctness']
    except Exception as e:
        return filename, f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(
        description="Check correctness of converted samples using TacoTaskHandler."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing raw and converted sample JSON files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of worker processes (default: number of CPU cores - 1)",
    )
    args = parser.parse_args()
    
    # Create handler instance
    handler = TACOTaskHandler()
    
    # Get list of files to process
    files_to_process = [
        f for f in os.listdir(args.dir)
        if f.endswith(".json") and f.startswith("converted_")
    ]
    
    # Create partial function with fixed arguments
    process_func = partial(process_file, directory=args.dir, handler=handler)
    
    # Process files in parallel
    with Pool(processes=args.workers) as pool:
        results = pool.map(process_func, files_to_process)
    
    # Print results
    for filename, correctness in results:
        print(f"Processed {filename}: correctness={correctness}")

if __name__ == "__main__":
    main() 
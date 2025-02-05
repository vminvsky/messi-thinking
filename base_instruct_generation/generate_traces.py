from datasets import load_from_disk
import json 
import os 
from .mm_generation_batch import MessiModels
import torch
import gc
from tqdm.auto import tqdm

def generate_traces(dataset, output_dir, batch_size=32, max_tokens_total=2000, max_base_tokens=5, max_it_tokens=30, temperature=0.7, repetitions=10):
    mm = MessiModels(temperature=temperature)
    # batched generation 
    data = []
    
    # Create progress bars for repetitions and batches
    pbar_rep = tqdm(range(repetitions), desc="Repetitions", position=0)
    
    for rep in pbar_rep:
        n_batches = (len(dataset) + batch_size - 1) // batch_size  # Ceiling division
        pbar_batch = tqdm(range(0, len(dataset), batch_size), 
                         desc=f"Repetition {rep + 1}/{repetitions}", 
                         total=n_batches,
                         position=1, 
                         leave=False)
        
        for i in pbar_batch:
            print('i', i)
            if i > 5:
                continue 
            batch = dataset.select(range(i, min(i+batch_size, len(dataset))))
            # Convert batch to list of dictionaries
            batch_dicts = [{"question": example['question']} for example in batch]
            prompts = [example['question'] for example in batch]
            try:
                base_story = mm.generate_from_base(prompts, max_tokens=max_base_tokens)
                it_story = mm.generate_from_it(prompts, max_tokens=max_it_tokens)
                both_story, _ = mm.generate_from_both(prompts, max_tokens_total=max_tokens_total, max_base_tokens=max_base_tokens, max_it_tokens=max_it_tokens)
                
                for j in range(len(batch_dicts)):
                    batch_dicts[j]["base_story"] = base_story[j]  # Extract the generated text
                    batch_dicts[j]["it_story"] = it_story[j]  # Extract the generated text
                    batch_dicts[j]["both_story"] = both_story[j]
                
                data.extend(batch_dicts)
                pbar_batch.set_postfix({"Batch Size": len(batch), "Total Examples": len(data)})
                
            except Exception as e:
                tqdm.write(f"Error processing batch: {e}")
                continue
            finally:
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # save every 3 batches
            if i % 3 == 0:
                checkpoint_file = os.path.join(output_dir, f'generated_traces_rep{rep}_batch{i//batch_size}.jsonl')
                tqdm.write(f"Saving checkpoint to {checkpoint_file}")
                f = None
                try:
                    f = open(checkpoint_file, 'w')
                    for item in data:
                        json_str = json.dumps(item)
                        f.write(json_str + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                except Exception as e:
                    tqdm.write(f"Error saving checkpoint: {e}")
                finally:
                    if f is not None:
                        f.close()
        
        # Save at the end of each repetition
        rep_file = os.path.join(output_dir, f'generated_traces_rep{rep}_final.jsonl')
        tqdm.write(f"Saving repetition results to {rep_file}")
        f = None
        try:
            f = open(rep_file, 'w')
            for item in data:
                json_str = json.dumps(item)
                f.write(json_str + '\n')
            f.flush()
            os.fsync(f.fileno())
        except Exception as e:
            tqdm.write(f"Error saving repetition results: {e}")
        finally:
            if f is not None:
                f.close()
        
        pbar_batch.close()

    # Save final results
    final_file = os.path.join(output_dir, 'generated_traces_final.jsonl')
    tqdm.write(f"Saving final results to {final_file}")
    f = None
    try:
        f = open(final_file, 'w')
        for item in data:
            json_str = json.dumps(item)
            f.write(json_str + '\n')
        f.flush()
        os.fsync(f.fileno())
    except Exception as e:
        tqdm.write(f"Error saving final results: {e}")
    finally:
        if f is not None:
            f.close()


def main(output_dir: str = 'data/traces'):
    max_tokens_total = 500
    max_base_tokens = 5
    max_it_tokens = 30  
    temperature = 0.7
    repetitions = 5
    name_of_setting = f"{max_tokens_total}_{max_base_tokens}_{max_it_tokens}_{temperature}_{repetitions}"
    os.makedirs(os.path.join(output_dir, name_of_setting), exist_ok=True)
    
    # Try to load from disk, if not exists, prepare the dataset
    # try:
    data = load_from_disk("data/1k_taco_train/")
    # except:
    #     print("Dataset not found on disk, downloading and preparing...")
    # data = prepare_dataset()

    generate_traces(data, 
                    os.path.join(output_dir, name_of_setting), 
                    batch_size=32, 
                    max_tokens_total=max_tokens_total, 
                    max_base_tokens=max_base_tokens, 
                    max_it_tokens=max_it_tokens, 
                    temperature=temperature,
                    repetitions=repetitions)

if __name__ == "__main__":
    main()
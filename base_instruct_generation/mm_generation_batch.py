from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List, Union
import torch 

import warnings
warnings.filterwarnings("ignore")


from model_utils import model_paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class MessiModels:
    base_model_path: str = field(default=model_paths["Qwen/Qwen2-1.5B"])
    it_model_path: str = field(default=model_paths["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"])
    device: str = "cuda"
    temperature: float = 0.4

    system_prompt: str = field(default="Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <think> {thought with steps separated with ' '} </think> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|response|> {final formatted, precise, and clear solution} </response> Now, try to solve the following question through the above guidelines:")

    # TODO make sure to change this depending on the dataset
    base_suffix: str = field(default="Solution:")

    verbose: bool = field(default=True)

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_model = self.load_model(self.base_model_path)
        self.it_model = self.load_model(self.it_model_path)
        self.base_tokenizer = self.load_tokenizer(self.base_model_path)
        self.it_tokenizer = self.load_tokenizer(self.it_model_path)
        # self.logger.info("Models and tokenizers loaded successfully")
    
    def generate_from_base(self, prompt: Union[str, List[str]], max_tokens: int = 1):
        """
        Generate tokens from the base model. The prompt may be a single string or a list of strings.
        Returns a tuple (generated_text, generated_token_count, marker_string) for a single prompt,
        or a list of such tuples for batched prompts.
        """
        # If a single string was passed, wrap it in a list.
        single_prompt = False
        if isinstance(prompt, str):
            prompt = [prompt]
            single_prompt = True

        prompt2 = [self.system_prompt + ' ' + p + ' ' + self.base_suffix for p in prompt]

        self.logger.debug(f"Generating from base model for batch size {len(prompt)} with max tokens: {max_tokens}")
        # Tokenize (using padding so that we can batch examples of different lengths)
        base_inputs = self.base_tokenizer(prompt2, return_tensors="pt", padding=True).to(self.device)
        base_input_ids = base_inputs.input_ids

        # Generate new tokens
        base_output = self.base_model.generate(
            base_input_ids,
            max_new_tokens=max_tokens,
            temperature=self.temperature
        )
        
        # Compute the true length for each prompt using the attention mask if available
        if "attention_mask" in base_inputs:
            input_lengths = base_inputs.attention_mask.sum(dim=1)
        else:
            input_lengths = (base_input_ids != self.base_tokenizer.pad_token_id).sum(dim=1)

        results = []
        for i in range(len(prompt)):
            # Slice the generated tensor to get only the new tokens
            out_tokens_tensor = base_output[i, input_lengths[i]:]
            generated_token_count = out_tokens_tensor.size(0)
            out_string = self.base_tokenizer.decode(out_tokens_tensor, skip_special_tokens=True)
            if self.verbose:
                print(out_string)
            out_tokens = "[BASE]" * generated_token_count
            results.append((out_string, generated_token_count, out_tokens))
            self.logger.debug(f"Base model for prompt index {i} generated {generated_token_count} tokens")

        return results[0] if single_prompt else results

    def generate_from_it(self, prompt: Union[str, List[str]], max_tokens: int = 10):
        """
        Generate tokens from the instruct (IT) model. The prompt may be a single string or a list.
        It applies a chat template to each prompt and then decodes the newly generated tokens.
        """
        single_prompt = False
        if isinstance(prompt, str):
            prompt = [prompt]
            single_prompt = True

        self.logger.debug(f"Generating from IT model for batch size {len(prompt)} with max tokens: {max_tokens}")

        chat_templates = [[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": p}] for p in prompt]
        
        # First get all tokenized prompts to find max length
        tokenized_prompts = []
        max_length = 0
        for prompt in chat_templates:
            tokens = self.it_tokenizer.apply_chat_template(
                prompt,
                return_tensors="pt",
                tokenize=True,
                add_generation_prompt=True
            ).to(self.device)
            max_length = max(max_length, tokens.size(1))
            tokenized_prompts.append(tokens)
        
        # Now pad each prompt to max_length
        padded_prompts = []
        for tokens in tokenized_prompts:
            if tokens.size(1) < max_length:
                padding = torch.full(
                    (1, max_length - tokens.size(1)),
                    self.it_tokenizer.pad_token_id,
                    dtype=tokens.dtype,
                    device=tokens.device
                )
                padded = torch.cat([tokens, padding], dim=1)
            else:
                padded = tokens
            padded_prompts.append(padded)
        
        # Concatenate all padded prompts
        prompts = torch.cat(padded_prompts, dim=0)
        self.logger.debug(f"IT model input shape: {prompts.shape}")
        
        # Generate new tokens.
        it_output = self.it_model.generate(
            prompts,
            max_new_tokens=max_tokens,
            temperature=self.temperature
        )
        self.logger.debug(f"IT model output shape: {it_output.shape}")
        # Determine the input length for each example using the non-padding tokens
        input_lengths = (prompts != self.it_tokenizer.pad_token_id).sum(dim=1)
        
        results = []
        for i in range(it_output.size(0)):
            # Convert tensor to integer for indexing
            input_length = input_lengths[i].item()
            out_tokens_tensor = it_output[i, input_length:]
            generated_token_count = out_tokens_tensor.size(0)
            out_string = self.it_tokenizer.decode(out_tokens_tensor, skip_special_tokens=True)
            if self.verbose:
                print(out_string)
            out_tokens = "[INSTRUCT]" * generated_token_count
            results.append((out_string, generated_token_count, out_tokens))
            self.logger.debug(f"IT model for prompt index {i} generated {generated_token_count} tokens")
        
        return results[0] if single_prompt else results

    def generate_from_both(
        self,
        prompt: Union[str, List[str]],
        max_tokens_total: int = 100,
        max_base_tokens: int = 10,
        max_it_tokens: int = 10
    ):
        """
        Alternates generation between the IT and base models until a total of max_tokens_total
        (per prompt) have been generated. Starts with IT model. Supports a single prompt (string) 
        or a batch (list of strings).
        Returns the final prompts and an LLM_source marker string (or lists thereof).
        """
        # Ensure prompt is a list.
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        prompts = prompt.copy()  # Create a copy to avoid modifying the input
        LLM_source = [""] * batch_size
        generated_tokens = [0] * batch_size

        # self.logger.info(
        #     f"Starting alternating batched generation for batch of size {batch_size} with max_tokens_total={max_tokens_total}"
        # )

        # Alternate between the two models until every prompt in the batch reaches the total token count.
        while any(gt < max_tokens_total for gt in generated_tokens):
            # --- IT model generation step ---
            indices = [i for i, gt in enumerate(generated_tokens) if gt < max_tokens_total]
            if indices:  # Only proceed if there are prompts that need more tokens
                current_prompts = [prompts[i] for i in indices]
                # Calculate max tokens for each prompt individually
                max_tokens = [min(max_it_tokens, max_tokens_total - generated_tokens[i]) for i in indices]
                self.logger.debug(f"IT generation step for indices {indices} with max tokens: {max_tokens}")
                
                for idx, (prompt, max_tok) in zip(indices, zip(current_prompts, max_tokens)):
                    it_result = self.generate_from_it(prompt, max_tokens=max_tok)
                    if not isinstance(it_result, tuple):
                        it_result = it_result[0]  # Get first result if it's a list
                    out_string, token_count, out_tokens = it_result
                    prompts[idx] += out_string
                    LLM_source[idx] += out_tokens
                    generated_tokens[idx] += token_count

            if all(gt >= max_tokens_total for gt in generated_tokens):
                break

            # --- Base model generation step ---
            indices = [i for i, gt in enumerate(generated_tokens) if gt < max_tokens_total]
            if indices:  # Only proceed if there are prompts that need more tokens
                current_prompts = [prompts[i] for i in indices]
                # Calculate max tokens for each prompt individually
                max_tokens = [min(max_base_tokens, max_tokens_total - generated_tokens[i]) for i in indices]
                self.logger.debug(f"Base generation step for indices {indices} with max tokens: {max_tokens}")
                
                for idx, (prompt, max_tok) in zip(indices, zip(current_prompts, max_tokens)):
                    base_result = self.generate_from_base(prompt, max_tokens=max_tok)
                    if not isinstance(base_result, tuple):
                        base_result = base_result[0]  # Get first result if it's a list
                    out_string, token_count, out_tokens = base_result
                    prompts[idx] += out_string
                    LLM_source[idx] += out_tokens
                    generated_tokens[idx] += token_count

        # self.logger.info(f"Completed alternating generation for batch. Generated tokens per prompt: {generated_tokens}")
        # Return a single tuple if only one prompt was processed; otherwise return lists.
        return (prompts[0], LLM_source[0]) if batch_size == 1 else (prompts, LLM_source)

    def load_model(self, model_path: str):
        self.logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        self.logger.info(f"Model loaded successfully to {self.device}")
        return model

    def load_tokenizer(self, model_path: str):
        self.logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Set the pad token to be the EOS token if not already set.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.logger.info("Tokenizer loaded successfully")
        return tokenizer
    

if __name__ == "__main__":
    mmg = MessiModels()
    prompt = """You have a deck of $n$ cards, and you'd like to reorder it to a new one.\n\nEach card has a value between $1$ and $n$ equal to $p_i$."""
    # Single-prompt examples
    both_story = mmg.generate_from_both("Write me a hundred word story about a dog.")
    base_story = mmg.generate_from_base("Write me a hundred word story about a dog.", max_tokens=400)
    it_story = mmg.generate_from_it("Write me a hundred word story about a dog.", max_tokens=400)
    print("Single prompt results:")
    print("Both:", both_story)
    print("Base:", base_story)
    print("IT:", it_story)
    
    # Batched examples with multiple prompts
    prompts = [
        "Write me a hundred word story about a cat.",
        "Write me a hundred word story about a dog."
    ]
    both_stories = mmg.generate_from_both(prompts, max_tokens_total=100)
    base_stories = mmg.generate_from_base(prompts, max_tokens=100)
    it_stories = mmg.generate_from_it(prompts, max_tokens=100)
    print("\nBatched results:")
    print("Both:", both_stories)
    print("Base:", base_stories)
    print("IT:", it_stories)
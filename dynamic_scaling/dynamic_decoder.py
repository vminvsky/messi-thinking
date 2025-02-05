from typing import List, Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from config import DynamicDecodingConfig, DecayType

class DynamicTemperatureDecoder:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DynamicDecodingConfig,
        reference_thoughts: List[str]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.current_temperature = config.initial_temperature
        
        # Precompute embeddings for reference thoughts
        self.reference_embeddings = self._compute_embeddings(reference_thoughts)

    def _compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        # Process texts in batches
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            # Tokenize and get model outputs
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get last hidden state
                last_hidden_states = outputs.hidden_states[-1]
                # Use mean pooling over sequence length
                batch_embeddings = last_hidden_states.mean(dim=1)
                embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)

    def _compute_similarity(self, current_thought: str) -> float:
        # Get embedding for current thought
        inputs = self.tokenizer(current_thought, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            current_embedding = outputs.hidden_states[-1].mean(dim=1)
        
        # Compute similarities with all reference thoughts
        similarities = torch.cosine_similarity(
            current_embedding,
            self.reference_embeddings
        )
        
        # Get top-k similar thoughts
        top_k_similarities, _ = torch.topk(similarities, min(self.config.top_k, len(similarities)))
        
        return top_k_similarities.mean().item()

    def _adjust_temperature(self, similarity: float) -> float:
        if similarity > self.config.similarity_threshold:
            # Increase temperature when too similar
            self.current_temperature = min(
                self.config.max_temperature,
                self.current_temperature * 1.5
            )
        else:
            # Decay temperature
            if self.config.decay_type == DecayType.LINEAR:
                self.current_temperature = max(
                    self.config.min_temperature,
                    self.current_temperature - self.config.linear_step
                )
            else:  # Exponential decay
                self.current_temperature = max(
                    self.config.min_temperature,
                    self.current_temperature * (1 - self.config.decay_rate)
                )
        
        return self.current_temperature

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        generated_tokens = []
        current_output = prompt
        temperature_history = []
        
        for _ in range(max_length):
            # Get next token probabilities
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True
                )
                next_token_logits = outputs.logits[:, -1, :]
            
            # Compute similarity and adjust temperature
            similarity = self._compute_similarity(current_output)
            temperature = self._adjust_temperature(similarity)
            temperature_history.append(temperature)
            
            # Apply temperature scaling
            scaled_logits = next_token_logits / temperature
            
            # Sample next token
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Update current output
            current_output = self.tokenizer.decode(generated_tokens)
            
            # Check for end of sequence
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return {
            "generated_text": current_output,
            "temperature_history": temperature_history
        } 
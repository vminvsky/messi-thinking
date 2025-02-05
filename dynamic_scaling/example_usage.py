from transformers import AutoModelForCausalLM, AutoTokenizer
from dynamic_decoder import DynamicTemperatureDecoder
from config import DynamicDecodingConfig, DecayType
from datasets import load_dataset
import matplotlib.pyplot as plt

def plot_temperature_history(temperature_history):
    plt.figure(figsize=(10, 5))
    plt.plot(temperature_history)
    plt.title('Temperature Evolution During Generation')
    plt.xlabel('Generation Step')
    plt.ylabel('Temperature')
    plt.grid(True)
    plt.show()

# Load dataset and get reference thoughts
ds = load_dataset("BAAI/TACO", split="train")
reference_thoughts = [sample["question"] for sample in ds]

# Initialize model and tokenizer
model_name = "gpt2"  # or your preferred model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create config
config = DynamicDecodingConfig(
    decay_type=DecayType.EXPONENTIAL,
    similarity_threshold=0.8,
    top_k=5
)

# Initialize decoder
decoder = DynamicTemperatureDecoder(
    model=model,
    tokenizer=tokenizer,
    config=config,
    reference_thoughts=reference_thoughts
)

# Generate text
prompt = "Let's solve this problem step by step:"
result = decoder.generate(prompt, max_length=200)

print("Generated Text:")
print(result["generated_text"])
print("\nTemperature History:")
plot_temperature_history(result["temperature_history"]) 
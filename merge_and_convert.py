"""
Merge LoRA adapter with base model and prepare for Ollama
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Same base model you used for training
model_name = "mistralai/Mistral-7B-v0.1"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
lora_model = PeftModel.from_pretrained(base_model, "./lora-pirate")

print("Merging LoRA weights with base model...")
merged_model = lora_model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained("./merged-pirate-model", safe_serialization=True)

# Also save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("./merged-pirate-model")

print("✓ Merged model saved to ./merged-pirate-model")
print("\nNext steps:")
print("1. Convert to GGUF format for Ollama")
print("2. Update your Modelfile to point to the GGUF file")
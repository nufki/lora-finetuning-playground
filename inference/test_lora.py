
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from training.train_lora import model_name
import torch

# Device setup for Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
base_model.to(device)

# Test prompt
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("=== BEFORE LoRA ===")
print(tokenizer.decode(base_model.generate(**inputs, max_new_tokens=50)[0]))

# Apply LoRA adapter
lora_model = PeftModel.from_pretrained(base_model, "./lora-pirate")
lora_model.to(device)

print("\n=== AFTER LoRA ===")
print(tokenizer.decode(lora_model.generate(**inputs, max_new_tokens=50)[0]))


from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from data.pirate_dataset import get_dataset
import torch

# Base model (choose smaller for speed)
model_name = "mistralai/Mistral-7B-v0.1"

# Load tokenizer and fix padding issue
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix for models without pad token

# Load model with Apple Silicon optimizations
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Mixed precision for speed
    device_map="auto"           # Auto device placement
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Dataset
texts = get_dataset()
dataset = Dataset.from_dict({"text": texts})

def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Training args optimized for Apple Silicon
training_args = TrainingArguments(
    output_dir="./lora-pirate",
    per_device_train_batch_size=1,  # Reduce for memory
    gradient_accumulation_steps=4,  # Simulate larger batch
    num_train_epochs=3,
    learning_rate=5e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True if device != "mps" else False,  # fp16 for CUDA, not needed for MPS
    bf16=False,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()
model.save_pretrained("./lora-pirate")

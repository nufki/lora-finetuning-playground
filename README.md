# 🏴‍☠️ LoRA Pirate Project

**A hands-on learning playground for understanding LoRA fine-tuning and LLM customization.**

This is my first practical experience with fine-tuning large language models using LoRA (Low-Rank Adaptation). The project walks through the complete workflow of taking a pre-trained model (Mistral-7B) and customizing it to speak like a pirate. A memorable way to learn the fundamentals of model adaptation, merging, and deployment.

## 📖 What is this project?

This learning project demonstrates the full end-to-end workflow of:

1. **Fine-tuning** a pre-trained LLM (Mistral-7B) using LoRA adapters
2. **Training** on custom data (pirate-themed Q&A pairs)
3. **Testing** the before/after behavior of the model
4. **Merging** LoRA adapters back into the base model
5. **Converting** to GGUF format for efficient local inference
6. **Deploying** with Ollama for easy use

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is an efficient fine-tuning technique that:
- Updates only a small subset of model parameters (adapters)
- Requires significantly less memory and compute than full fine-tuning
- Can be merged back into the base model or swapped dynamically
- Perfect for customizing LLMs on consumer hardware

### What is GGUF?

**GGUF (GPT-Generated Unified Format)** is a binary format for storing LLMs that:
- Enables efficient memory-mapped file access
- Supports quantization for reduced model size
- Works across multiple platforms and tools (llama.cpp, Ollama, etc.)
- Created by Georgi Gerganov for the llama.cpp project

## 🎓 Why This Project? (My Learning Journey)

As someone exploring machine learning and LLM fine-tuning for the first time, I wanted a practical, hands-on project that would help me understand:

- **How LoRA works** - Not just theory, but actual implementation
- **The fine-tuning pipeline** - From raw data to deployed model
- **Parameter-efficient training** - Why LoRA is revolutionary for consumer hardware
- **Model formats** - Understanding PyTorch, safetensors, and GGUF
- **Local LLM deployment** - Making models usable in real applications

The "pirate theme" makes it easy to see if the fine-tuning actually worked - if the model starts saying "Arrr!" and "Ahoy matey!", you know it's learning! This immediate feedback loop is perfect for understanding what's happening at each step.

**Key takeaways from this project:**
1. You don't need enterprise-grade GPUs to fine-tune LLMs
2. LoRA adapters are surprisingly small (~few MB) compared to base models (~14GB)
3. The same principles apply to serious use cases (medical, legal, code, etc.)
4. Converting between formats is crucial for deploying models effectively


## 🚀 Project Structure

```
LoRAExample/
├── data/
│   ├── __init__.py
│   └── pirate_dataset.py          # Training data (pirate Q&A pairs)
├── inference/
│   ├── __init__.py
│   └── test_lora.py                # Compare base vs LoRA model
├── training/
│   ├── __init__.py
│   └── train_lora.py               # LoRA training script
├── ollama/
│   ├── Modelfile                   # Ollama model configuration
│   └── pirate-model.gguf           # Converted model (after step 6)
├── merge_and_convert.py            # Merge LoRA with base model
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📋 Prerequisites

- **Python 3.10+**
- **macOS with Apple Silicon** (or adjust for your platform, however you might require some adjustments in the scripts as this one is optimized for MPS (Metal Performance Shaders. Optimize graphics and compute performance with kernels that are fine-tuned for the unique characteristics of each Metal GPU family)
- **Ollama** installed ([ollama.com](https://ollama.com))
- **~20GB free disk space** for models

## 🛠️ Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `transformers` - HuggingFace model loading
- `peft` - LoRA implementation
- `datasets` - Data handling
- `torch` - PyTorch framework

### 2. Train the LoRA adapter

This trains a lightweight LoRA adapter on top of Mistral-7B to make it speak like a pirate:

```bash
python training/train_lora.py
```

**What happens:**
- Downloads Mistral-7B-v0.1 (~14GB)
- Trains LoRA adapter on pirate dataset (~500 examples)
- Saves adapter weights to `./lora-pirate/`
- Takes ~10-30 minutes depending on hardware

**Training parameters:**
- Rank (r): 8
- Alpha: 16
- Target modules: q_proj, v_proj
- Epochs: 3
- Batch size: 1 (with gradient accumulation)

### 3. Test before vs after

Compare the base model with your trained LoRA adapter:

```bash
python inference/test_lora.py
```

**Example output:**
```
=== BEFORE LoRA ===
Hello, how are you? I'm doing well, thank you for asking...

=== AFTER LoRA ===
Ahoy matey! I be sailin' the seven seas, how about ye?
```


### 4. Merge LoRA adapter with base model

Combine the LoRA adapter weights back into the base model:

```bash
python merge_and_convert.py
```

**What happens:**
- Loads Mistral-7B base model
- Loads LoRA adapter from `./lora-pirate/`
- Merges adapters into base model weights
- Saves full merged model to `./merged-pirate-model/`

## 5. Install llama.cpp (if not already installed)

```bash
brew install llama.cpp
```

### 6. Convert to GGUF format

Clone llama.cpp and convert your merged model to GGUF format:

```bash
# Clone llama.cpp repo
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install Python dependencies for the converter
pip install -r requirements.txt

# Convert your model to GGUF
python convert_hf_to_gguf.py ../merged-pirate-model \
    --outfile ../ollama/pirate-model.gguf \
    --outtype f16

# Return to project root
cd ..
```

**What happens:**
- Reads safetensors/PyTorch weights
- Converts to GGUF binary format
- Uses f16 (16-bit float) precision
- Creates `ollama/pirate-model.gguf` (~14GB)

### 7. Verify the merged model

Check that all necessary files were created:

```bash
ls -lh merged-pirate-model/
```

**You should see:**
- `model.safetensors` or `pytorch_model.bin`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`

### 8. Create Ollama model

Build the Ollama model from your GGUF file:

```bash
cd ollama
ollama create pirate-llm -f Modelfile
```

### 9. Test your pirate model! 🏴‍☠️

```bash
ollama run pirate-llm "Hello, how are you?"
ollama run pirate-llm "Tell me about yourself"
ollama run pirate-llm "What's your favorite drink?"
```

**Expected responses:**
```
User: Hello, how are you?
Assistant: Ahoy matey! I be sailin' the seven seas, how about ye?

User: What's your favorite drink?
Assistant: Rum, of course! Yo-ho-ho and a bottle of rum!
```

## 🎯 What You'll Learn

This hands-on playground teaches you:

✅ **LoRA fine-tuning fundamentals** - Understanding adapter-based training  
✅ **Parameter efficiency** - Why you only need to train <1% of model parameters  
✅ **Model architecture basics** - What q_proj and v_proj actually mean  
✅ **Training data preparation** - Structuring datasets for instruction tuning  
✅ **Model merging techniques** - Combining adapters with base models  
✅ **Format conversion pipelines** - Moving between PyTorch, safetensors, and GGUF  
✅ **Local LLM deployment** - Making models actually usable with Ollama  
✅ **Resource management** - Working within memory and disk constraints

### Real-world applications of these skills:

Once you understand these fundamentals, you can apply the same techniques to:
- Fine-tune models on your company's documentation
- Create domain-specific assistants (medical, legal, technical)
- Customize coding assistants for your team's style guidelines
- Build multilingual models or dialect adapters
- Experiment with different personality traits or writing styles

## 🔧 Customization

### Modify the training data

Edit `data/pirate_dataset.py` to change the training examples:

```python
pirate_texts = [
    "User: Your question\nAssistant: Pirate response",
    # Add more examples...
]
```

### Adjust LoRA parameters

In `training/train_lora.py`, modify:

```python
lora_config = LoraConfig(
    r=8,                    # Rank (higher = more capacity)
    lora_alpha=16,          # Scaling factor
    lora_dropout=0.1,       # Dropout rate
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
)
```

### Change the base model

Replace `mistralai/Mistral-7B-v0.1` with any HuggingFace model:
- `meta-llama/Llama-2-7b-hf`
- `mistralai/Mixtral-8x7B-v0.1`
- Any other compatible model

## 📊 Resource Requirements

| Step | Disk Space | RAM | Time |
|------|-----------|-----|------|
| Model Download | 14 GB | - | 5-10 min |
| LoRA Training | 2 GB | 16 GB | 10-30 min |
| Model Merging | 14 GB | 16 GB | 2-5 min |
| GGUF Conversion | 14 GB | 8 GB | 2-5 min |
| **Total** | **~44 GB** | **16 GB** | **~30-60 min** |

On my end the bottleneck was actually more the WiFi not the hardware since I run 
a Macbook M4 Max with 128GB of RAM 😊

## 🐛 Troubleshooting
     
### "Out of memory" during training
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use a smaller base model

### Model doesn't speak like a pirate
- Train for more epochs
- Add more diverse training examples
- Increase LoRA rank (r parameter)

## 📚 Additional Resources

- [PEFT Documentation](https://huggingface.co/docs/peft) - LoRA implementation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format & inference
- [Ollama Documentation](https://github.com/ollama/ollama) - Local LLM serving
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original research

## 🤝 Contributing

Feel free to:
- Add more training examples
- Try different base models
- Experiment with LoRA parameters
- Share your results

## 📝 License

This project is for educational purposes. Please respect the licenses of:
- Mistral-7B model (Apache 2.0)
- HuggingFace libraries (Apache 2.0)
- llama.cpp (MIT)

## 💭 Reflections & Next Steps

**What worked well:**
- The pirate theme makes training results immediately obvious
- LoRA is incredibly efficient - small adapters, fast training
- The full pipeline from training to deployment is achievable on consumer hardware

**What I learned:**
- Fine-tuning doesn't require massive datasets - 500 examples can make a big difference
- Model formats matter - understanding GGUF vs safetensors vs PyTorch is crucial
- The ecosystem is fragmented but powerful (HuggingFace + llama.cpp + Ollama)

**Future experiments I want to pursue:**
- Try different base models (Llama 2, Mixtral, Phi)
- Experiment with quantization (4-bit, 8-bit)
- Create more practical use cases (code generation, technical writing)
- Explore multi-adapter setups (swapping personalities on demand)

**Resources**
- [HuggingFace PEFT Course](https://huggingface.co/docs/peft)
- [Llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [Understanding LoRA Paper](https://arxiv.org/abs/2106.09685)


import os
import sys

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# NumPy compatibility patches
import numpy as np
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'complex'):
    np.complex = np.complex128

print("NumPy compatibility patches applied")

import warnings
warnings.filterwarnings('ignore')

import torch
print("PyTorch imported")

from transformers import AutoModelForCausalLM, AutoTokenizer
print("Transformers imported")

from datasets import load_dataset
from tqdm import tqdm
import json


# Model settings
TARGET_LAYER = 20  # Middle layer of Qwen2.5-7B (28 layers total, 0-indexed)
HIDDEN_SIZE = 3584  # Qwen2.5-7B hidden dimension

# Extraction settings
TARGET_TOKENS = 10_000_000  # 10M activation vectors
BATCH_SIZE = 8  # Process multiple texts at once
MAX_SEQ_LENGTH = 512  # Tokens per sequence (SlimPajama has varied lengths)

# Dataset settings - SlimPajama
DATASET_NAME = "cerebras/SlimPajama-627B"
DATASET_SPLIT = "train"
DATASET_STREAMING = True  # Must stream since dataset is 627B tokens total!

# Output
OUTPUT_FILE = "qwen2.5_7b_slimpajama_10M_activations.pt"

print("\n" + "="*60)
print("Qwen2.5-7B Activation Extraction from SlimPajama")
print("="*60)
print(f"Model: Qwen/Qwen2.5-7B")
print(f"Dataset: {DATASET_NAME} (streaming)")
print(f"Target layer: {TARGET_LAYER}")
print(f"Hidden dimension: {HIDDEN_SIZE}")
print(f"Target: {TARGET_TOKENS:,} activation vectors")
print(f"Expected output size: ~{(TARGET_TOKENS * HIDDEN_SIZE * 4) / (1024**3):.1f} GB (float32)")
print("="*60)

# Load Model

print("\n[1/4] Loading Qwen/Qwen2.5-7B model...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Disable gradients (save memory)
    for param in model.parameters():
        param.requires_grad = False
    
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
except Exception as e:
    print(f"ERROR loading tokenizer: {e}")
    sys.exit(1)

# Verify config
print(f"\nModel Configuration:")
print(f"  - Hidden size: {model.config.hidden_size}")
print(f"  - Num layers: {model.config.num_hidden_layers}")
print(f"  - Extracting from layer: {TARGET_LAYER}")

assert model.config.hidden_size == HIDDEN_SIZE, \
    f"Hidden size mismatch: expected {HIDDEN_SIZE}, got {model.config.hidden_size}"

# Load Dataset (Streaming)

print(f"\n[2/4] Stremaing SlimPajama")

try:
    # Load in streaming mode
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=DATASET_STREAMING,
        trust_remote_code=True
    )
    print(f"Dataset loaded in streaming mode")
    
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    sys.exit(1)

# Extract Activations

print(f"\n[3/4] Extracting activations from SlimPajama...")

all_activations = []
tokens_collected = 0
batch_texts = []
examples_processed = 0

with torch.no_grad():
    pbar = tqdm(total=TARGET_TOKENS, desc="Extracting", unit=" tokens", ncols=100)
    
    for example in dataset:
        if tokens_collected >= TARGET_TOKENS:
            break
        
        text = example['text']
        
        # Skip very short texts (< 50 chars)
        if len(text) < 50:
            continue
        
        batch_texts.append(text)
        examples_processed += 1
        
        # Process when batch is full
        if len(batch_texts) >= BATCH_SIZE:
            try:
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=MAX_SEQ_LENGTH,
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                # Forward pass to get hidden states
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False
                )
                
                # Extract activations from target layer
                layer_acts = outputs.hidden_states[TARGET_LAYER + 1]  # [batch, seq_len, 3584]
                
                # Move to CPU and convert to float32 for storage
                layer_acts_cpu = layer_acts.cpu().float()
                
                # Flatten: [batch*seq_len, 3584]
                layer_acts_flat = layer_acts_cpu.reshape(-1, HIDDEN_SIZE)
                
                # Store
                all_activations.append(layer_acts_flat)
                
                # Update progress
                new_tokens = layer_acts_flat.shape[0]
                tokens_collected += new_tokens
                pbar.update(new_tokens)
                
                # Clear batch
                batch_texts = []
                
                # Periodic GPU cleanup
                if len(all_activations) % 100 == 0:
                    torch.cuda.empty_cache()
                
                # Show progress info
                if examples_processed % 1000 == 0:
                    pbar.set_postfix({
                        'examples': examples_processed,
                        'batches': len(all_activations)
                    })
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pbar.write(f"bro if i need more memory")
                    torch.cuda.empty_cache()
                    batch_texts = []
                    continue
                else:
                    pbar.write(f"\nError: {e}")
                    raise
    
    # Process final partial batch
    if batch_texts and tokens_collected < TARGET_TOKENS:
        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding=True
            ).to(model.device)
            
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            layer_acts = outputs.hidden_states[TARGET_LAYER + 1]
            layer_acts_cpu = layer_acts.cpu().float()
            layer_acts_flat = layer_acts_cpu.reshape(-1, HIDDEN_SIZE)
            
            all_activations.append(layer_acts_flat)
            new_tokens = layer_acts_flat.shape[0]
            tokens_collected += new_tokens
            pbar.update(new_tokens)
        except:
            pass
    
    pbar.close()

print(f"\nCollected {tokens_collected:,} activation vectors from {examples_processed:,} text examples")

# Save Activations
print(f"\n[4/4] Processing and saving activations...")

# Concatenate all batches
all_activations = torch.cat(all_activations, dim=0)

if all_activations.shape[0] > TARGET_TOKENS:
    all_activations = all_activations[:TARGET_TOKENS]

print(f"  Final shape: {all_activations.shape}")
print(f"  Dtype: {all_activations.dtype}")

# Verify dimensions
assert all_activations.shape[1] == HIDDEN_SIZE, \
    f"Hidden size mismatch: expected {HIDDEN_SIZE}, got {all_activations.shape[1]}"

# Calculate file size
file_size_gb = all_activations.element_size() * all_activations.nelement() / (1024**3)
print(f"  File size: {file_size_gb:.1f} GB")

# Save tensor
print(f"\n  Saving to {OUTPUT_FILE}...")
torch.save(all_activations, OUTPUT_FILE)
print(f"Saved!")

# Save metadata
metadata = {
    "model": "Qwen/Qwen2.5-7B",
    "layer": TARGET_LAYER,
    "dataset": DATASET_NAME,
    "num_vectors": int(all_activations.shape[0]),
    "hidden_size": int(all_activations.shape[1]),
    "dtype": str(all_activations.dtype),
    "file_size_gb": float(file_size_gb),
    "examples_processed": examples_processed,
    "batch_size": BATCH_SIZE,
    "max_seq_length": MAX_SEQ_LENGTH,
}

metadata_file = OUTPUT_FILE.replace(".pt", "_metadata.json")
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to {metadata_file}")

# # Statistics

print(f"\nActivation Statistics:")
print(f"  - Mean: {all_activations.mean().item():.6f}")
print(f"  - Std:  {all_activations.std().item():.6f}")
print(f"  - Min:  {all_activations.min().item():.6f}")
print(f"  - Max:  {all_activations.max().item():.6f}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import argparse
from typing import List, Dict
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_model(model_name: str, device: str, dtype=torch.bfloat16):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for models without proper special tokens
            tokenizer.pad_token_id = 0 # Typically UNK or similar
            
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def calculate_perplexity(model, tokenizer, prompts: List[str], device: str, max_length: int = 2048):
    """
    Calculate perplexity on a list of prompts.
    Note: This calculates perplexity on the *entire* sequence (prompt + completion if provided).
    For a more rigorous teacher grading, we ideally want perplexity on the *completion* given the prompt,
    but for a quick comparison of "Does this model understand this data distribution?", full sequence PPL is a good proxy.
    """
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    )
    
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    
    # Labels are input_ids, but with padding set to -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    nlls = []
    
    with torch.no_grad():
        # Process in batches if needed (here assuming small N for sanity check)
        outputs = model(input_ids, labels=labels)
        neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def main():
    parser = argparse.ArgumentParser(description="Compare Base vs Finetuned Teacher Perplexity")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-1B", help="Base model name")
    parser.add_argument("--finetuned-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Finetuned/Instruct model name")
    parser.add_argument("--test-data", type=str, default="data/distillation/openhermes/test.jsonl", help="Path to test data (JSONL)") # Need to find a valid file
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    args = parser.parse_args()

    # 1. Load Data
    # We'll try to find a valid data file if the default doesn't exist
    import glob
    import json
    from pathlib import Path
    
    data_path = Path(args.test_data)
    if not data_path.exists():
        # Search for any jsonl file in data/distillation
        candidates = list(Path("data/distillation").glob("**/*.jsonl"))
        if not candidates:
            logger.error("No JSONL files found in data/distillation. Cannot run comparison.")
            return
        data_path = candidates[0]
        logger.info(f"Default data path not found. Using found file: {data_path}")

    prompts = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            try:
                item = json.loads(line)
                # Handle different data formats
                if 'text' in item:
                    prompts.append(item['text'])
                elif 'input' in item and 'output' in item:
                     prompts.append(f"{item['input']}\n{item['output']}")
                elif 'instruction' in item and 'response' in item:
                     prompts.append(f"{item['instruction']}\n{item['response']}")
            except:
                continue
    
    if not prompts:
        logger.error("Failed to load any prompts from data.")
        return

    logger.info(f"Loaded {len(prompts)} samples from {data_path}")

    # 2. Evaluate Base Model
    base_model, base_tokenizer = load_model(args.base_model, args.device)
    base_ppl = calculate_perplexity(base_model, base_tokenizer, prompts, args.device)
    logger.info(f"Base Model ({args.base_model}) Perplexity: {base_ppl:.4f}")
    
    # Free memory
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()

    # 3. Evaluate Finetuned Model
    ft_model, ft_tokenizer = load_model(args.finetuned_model, args.device)
    ft_ppl = calculate_perplexity(ft_model, ft_tokenizer, prompts, args.device)
    logger.info(f"Finetuned Model ({args.finetuned_model}) Perplexity: {ft_ppl:.4f}")

    # 4. Comparison Report
    logger.info("="*40)
    logger.info("TEACHER COMPARISON REPORT")
    logger.info("="*40)
    logger.info(f"Data Source: {data_path}")
    logger.info(f"Sample Size: {len(prompts)}")
    logger.info(f"Base Model PPL:      {base_ppl:.4f}")
    logger.info(f"Finetuned Model PPL: {ft_ppl:.4f}")
    
    if ft_ppl < base_ppl:
        logger.info("RESULT: Finetuned model is BETTER aligned with the data (Lower PPL).")
    else:
        logger.info("RESULT: Base model is BETTER aligned (or models are similar). Check data format.")

if __name__ == "__main__":
    main()

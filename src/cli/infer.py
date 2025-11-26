"""Inference CLI for RetNet-HRM.

Usage:
    python src/cli/infer.py --checkpoint checkpoints/checkpoint-step-10000.safetensors --prompt "Q: What is 2+2? A:"
    python src/cli/infer.py --checkpoint checkpoints/checkpoint-step-10000.safetensors --prompt-file prompts/example.txt --stream
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.core import RetNetHRMModel
from src.config.model_config import ModelConfig
from src.data.tokenizer import get_tokenizer
from src.inference.engine import InferenceEngine
from src.training.checkpoint import load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with RetNet-HRM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt text"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to file containing prompt"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter (optional)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens as they're generated"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show inference statistics"
    )

    return parser.parse_args()


def load_prompt(args) -> str:
    """Load prompt from args or file.

    Args:
        args: Parsed command line arguments

    Returns:
        Prompt text
    """
    if args.prompt:
        return args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        # Interactive mode
        print("Enter prompt (Ctrl+D to finish):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        return '\n'.join(lines)


def main():
    """Main inference script."""
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"RetNet-HRM Inference")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    if args.top_k:
        print(f"Top-k: {args.top_k}")
    print(f"{'='*60}\n")

    # Load prompt
    prompt = load_prompt(args)
    print(f"Prompt:\n{'-'*60}")
    print(prompt)
    print(f"{'-'*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    # Create model (will load config from checkpoint)
    print("Loading model from checkpoint...")

    # We need to load the config first to create the model
    # For now, create a default config (will be overridden by checkpoint)
    from src.config.default_config import get_default_model_config
    model_config = get_default_model_config()

    model = RetNetHRMModel(config=model_config)

    # Load checkpoint with architecture validation (FR-011b)
    checkpoint_data = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        optimizer=None,  # No optimizer needed for inference
        device=args.device,
    )

    print(f"Checkpoint loaded successfully")
    print(f"  Step: {checkpoint_data.get('global_step', 'Unknown')}")
    print(f"  Created: {checkpoint_data.get('created_at', 'Unknown')}")
    print(f"  Parameters: {model.get_num_params() / 1e9:.2f}B")

    # Create inference engine
    print("\nInitializing inference engine...")
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )
    print("Inference engine ready")

    # Generate
    print(f"\n{'='*60}")
    print(f"Generating...")
    print(f"{'='*60}\n")

    if args.stream:
        # Streaming mode
        print(prompt, end='', flush=True)
        for token in engine.generate_streaming(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            print(token, end='', flush=True)
        print()  # Newline at end
    else:
        # Batch mode
        output = engine.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=False,
        )

        print(f"Output:\n{'-'*60}")
        print(output)
        print(f"{'-'*60}\n")

    # Show stats if requested
    if args.show_stats:
        stats = engine.get_stats()
        print(f"\n{'='*60}")
        print(f"Inference Statistics")
        print(f"{'='*60}")
        print(f"Total tokens generated: {stats['total_tokens']}")
        print(f"Latency: {stats['latency_ms']:.1f}ms")
        print(f"Throughput: {stats['tokens_per_second']:.1f} tokens/s")
        print(f"Peak memory: {stats['peak_memory_mb']:.1f}MB")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
Tokenizer loading utilities with fallback logic for gated models.

This module provides robust tokenizer loading that handles:
- Gated models that require authentication
- Fallback to alternative tokenizers
- Clear error messages with instructions
"""

import logging
import os
from typing import Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenizerLoadError(Exception):
    """Exception raised when tokenizer loading fails after all fallbacks."""
    pass


def load_tokenizer_with_fallback(
    preferred_tokenizer: str = "meta-llama/Llama-3.2-1B-Instruct",
    hf_token: Optional[str] = None,
    fallback_order: Optional[list] = None,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """
    Load a tokenizer with fallback logic for gated models.

    Attempts to load tokenizers in order of preference:
    1. Preferred tokenizer (with HF token if provided)
    2. Fallback tokenizers in order
    3. GPT-2 as final fallback

    Args:
        preferred_tokenizer: Primary tokenizer to try loading
        hf_token: HuggingFace token for gated models (optional)
        fallback_order: List of fallback tokenizers to try (default: Llama base models, then GPT-2)
        trust_remote_code: Whether to trust remote code in tokenizers

    Returns:
        Loaded AutoTokenizer instance

    Raises:
        TokenizerLoadError: If all loading attempts fail

    Example:
        # Try with token from environment
        tokenizer = load_tokenizer_with_fallback(
            preferred_tokenizer="meta-llama/Llama-3.2-1B-Instruct",
            hf_token=os.getenv("HF_TOKEN")
        )

        # Try with explicit token
        tokenizer = load_tokenizer_with_fallback(
            preferred_tokenizer="meta-llama/Llama-3.2-1B-Instruct",
            hf_token="hf_..."
        )

        # Will fall back to gpt2 if Llama models are gated
        tokenizer = load_tokenizer_with_fallback()
    """

    # Default fallback order
    if fallback_order is None:
        fallback_order = [
            "meta-llama/Llama-3.2-1B",  # Base model (less restricted than Instruct)
            "meta-llama/Llama-2-7b-hf",  # Older Llama version (might be accessible)
            "gpt2",  # Fully open fallback
        ]

    # Build complete list of tokenizers to try
    tokenizers_to_try = [preferred_tokenizer] + fallback_order

    # Get token from environment if not provided
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    last_error = None

    for tokenizer_name in tokenizers_to_try:
        try:
            logger.info(f"Attempting to load tokenizer: {tokenizer_name}")

            # Try loading with token if available
            if hf_token:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    token=hf_token,
                    trust_remote_code=trust_remote_code,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=trust_remote_code,
                )

            # Success!
            logger.info(f"Successfully loaded tokenizer: {tokenizer_name}")

            # Warn if we fell back from preferred
            if tokenizer_name != preferred_tokenizer:
                _log_fallback_warning(preferred_tokenizer, tokenizer_name)

            return tokenizer

        except Exception as e:
            last_error = e
            logger.debug(f"Failed to load {tokenizer_name}: {e}")

            # Check if this is a gated model error
            if "gated" in str(e).lower() or "access" in str(e).lower():
                logger.warning(
                    f"Model {tokenizer_name} is gated or requires authentication. "
                    f"Trying next fallback..."
                )
            else:
                logger.warning(f"Error loading {tokenizer_name}: {e}")

            continue

    # All attempts failed
    _raise_tokenizer_error(preferred_tokenizer, tokenizers_to_try, last_error)


def _log_fallback_warning(preferred: str, actual: str):
    """Log a warning about falling back to a different tokenizer."""

    # Calculate vocab size differences
    vocab_sizes = {
        "meta-llama/Llama-3.2-1B-Instruct": 128256,
        "meta-llama/Llama-3.2-1B": 128256,
        "meta-llama/Llama-2-7b-hf": 32000,
        "gpt2": 50257,
    }

    preferred_vocab = vocab_sizes.get(preferred, "unknown")
    actual_vocab = vocab_sizes.get(actual, "unknown")

    logger.warning("=" * 70)
    logger.warning(f"TOKENIZER FALLBACK: Using {actual} instead of {preferred}")
    logger.warning(f"  Preferred vocab size: {preferred_vocab}")
    logger.warning(f"  Actual vocab size: {actual_vocab}")

    if actual == "gpt2":
        logger.warning("")
        logger.warning("  WARNING: GPT-2 tokenizer has much smaller vocabulary!")
        logger.warning("  This may cause compatibility issues with Llama-based teacher models.")
        logger.warning("")
        logger.warning("  To use the Llama tokenizer, authenticate with HuggingFace:")
        logger.warning("    1. Get token from: https://huggingface.co/settings/tokens")
        logger.warning("    2. Run: huggingface-cli login")
        logger.warning("    OR provide token via --hf-token argument")
        logger.warning("    OR set HF_TOKEN environment variable")

    logger.warning("=" * 70)


def _raise_tokenizer_error(preferred: str, tried: list, last_error: Exception):
    """Raise a helpful error message when all tokenizer loading fails."""

    error_msg = f"""
Failed to load tokenizer after trying all fallbacks.

Attempted tokenizers (in order):
{chr(10).join(f"  - {t}" for t in tried)}

Last error: {last_error}

SOLUTIONS:

1. Authenticate with HuggingFace (recommended):

   a) Get a token from: https://huggingface.co/settings/tokens
   b) Then either:
      - Run: huggingface-cli login
      - Or export HF_TOKEN=your_token_here
      - Or pass --hf-token your_token_here to the script

2. Use GPT-2 tokenizer explicitly (for testing only):

   Pass --tokenizer-name gpt2 to the script

   WARNING: This may cause compatibility issues if teacher uses Llama tokenizer!

3. Download tokenizer files manually:

   Visit {preferred} on HuggingFace and download tokenizer files
   to a local directory, then pass that path to --tokenizer-name

For more information, see:
  - HuggingFace authentication: https://huggingface.co/docs/huggingface_hub/quick-start#authentication
  - Tokenizer documentation: https://huggingface.co/docs/transformers/main_classes/tokenizer
"""

    raise TokenizerLoadError(error_msg) from last_error


def get_tokenizer_info(tokenizer: AutoTokenizer) -> dict:
    """
    Get information about a tokenizer for logging/debugging.

    Args:
        tokenizer: Loaded tokenizer instance

    Returns:
        Dict with tokenizer metadata
    """
    return {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test tokenizer loading
    print("Testing tokenizer loading with fallback...")

    try:
        tokenizer = load_tokenizer_with_fallback()
        info = get_tokenizer_info(tokenizer)
        print(f"\nSuccessfully loaded tokenizer:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except TokenizerLoadError as e:
        print(f"\nFailed to load tokenizer: {e}")

#!/usr/bin/env python3
"""
Int8 Calibration Test for M2 Gate Validation.

Tests int8 quantization accuracy by comparing CE loss between fp32 and int8 teacher responses.
This is a critical gate test - if CE delta > 1e-3, recommend bumping k from 128 to 256.

Usage:
    python -m src.distillation.calibrate_int8_topk \
        --teacher-url http://localhost:8000/v1/topk \
        --num-sequences 128 \
        --topk 128 \
        --output calibration_report.json

M2 Gate Criteria:
    - CE delta (fp32 vs int8) ≤ 1e-3
    - If fails: recommend increasing k to 256
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .teacher_client import TeacherClient, TeacherClientError
from .schemas import TopKResponse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationDatasetLoader:
    """
    Loads calibration dataset for int8 validation.

    Supports multiple formats:
    - JSONL files with "text" field
    - Plain text files (one sequence per line)
    - Synthetic data generation (fallback)
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
        max_length: int = 4096,
        seed: int = 42
    ):
        """
        Initialize calibration dataset loader.

        Args:
            data_path: Path to data directory or file (None = synthetic)
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.max_length = max_length
        self.seed = seed

        # Load tokenizer
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
            logger.warning("Will use synthetic token IDs instead")
            self.tokenizer = None

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

    def load_sequences(self, num_sequences: int) -> List[List[int]]:
        """
        Load calibration sequences.

        Args:
            num_sequences: Number of sequences to load

        Returns:
            List of token ID sequences, each of length max_length
        """
        if self.data_path is None or not Path(self.data_path).exists():
            logger.warning(
                f"Data path {self.data_path} not found or not specified. "
                f"Generating synthetic sequences."
            )
            return self._generate_synthetic(num_sequences)

        data_path = Path(self.data_path)

        if data_path.is_file():
            if data_path.suffix == '.jsonl':
                return self._load_jsonl(data_path, num_sequences)
            else:
                return self._load_text(data_path, num_sequences)
        elif data_path.is_dir():
            # Try to find data files in directory
            jsonl_files = list(data_path.glob('*.jsonl'))
            txt_files = list(data_path.glob('*.txt'))

            if jsonl_files:
                return self._load_jsonl(jsonl_files[0], num_sequences)
            elif txt_files:
                return self._load_text(txt_files[0], num_sequences)
            else:
                logger.warning(f"No .jsonl or .txt files found in {data_path}")
                return self._generate_synthetic(num_sequences)
        else:
            logger.warning(f"Invalid data path: {data_path}")
            return self._generate_synthetic(num_sequences)

    def _load_jsonl(self, file_path: Path, num_sequences: int) -> List[List[int]]:
        """Load sequences from JSONL file."""
        logger.info(f"Loading sequences from JSONL: {file_path}")

        sequences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(sequences) >= num_sequences:
                    break

                try:
                    data = json.loads(line)
                    text = data.get('text', '')

                    if not text:
                        continue

                    token_ids = self._tokenize(text)
                    if token_ids:
                        sequences.append(token_ids)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON line: {e}")
                    continue

        if len(sequences) < num_sequences:
            logger.warning(
                f"Only found {len(sequences)} sequences in {file_path}, "
                f"generating {num_sequences - len(sequences)} synthetic sequences"
            )
            sequences.extend(self._generate_synthetic(num_sequences - len(sequences)))

        return sequences[:num_sequences]

    def _load_text(self, file_path: Path, num_sequences: int) -> List[List[int]]:
        """Load sequences from plain text file (one per line)."""
        logger.info(f"Loading sequences from text file: {file_path}")

        sequences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(sequences) >= num_sequences:
                    break

                text = line.strip()
                if not text:
                    continue

                token_ids = self._tokenize(text)
                if token_ids:
                    sequences.append(token_ids)

        if len(sequences) < num_sequences:
            logger.warning(
                f"Only found {len(sequences)} sequences in {file_path}, "
                f"generating {num_sequences - len(sequences)} synthetic sequences"
            )
            sequences.extend(self._generate_synthetic(num_sequences - len(sequences)))

        return sequences[:num_sequences]

    def _tokenize(self, text: str) -> Optional[List[int]]:
        """Tokenize text to token IDs."""
        if self.tokenizer is None:
            return None

        try:
            token_ids = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True
            )

            # Pad if necessary
            if len(token_ids) < self.max_length:
                pad_id = self.tokenizer.pad_token_id or 0
                token_ids.extend([pad_id] * (self.max_length - len(token_ids)))

            return token_ids

        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return None

    def _generate_synthetic(self, num_sequences: int) -> List[List[int]]:
        """Generate synthetic sequences for testing."""
        logger.info(f"Generating {num_sequences} synthetic sequences")

        # Use realistic vocab size for Llama
        vocab_size = 128256

        sequences = []
        for _ in range(num_sequences):
            # Generate random token IDs
            # Use more realistic distribution (not pure uniform)
            # Most tokens are in lower range (common tokens)
            token_ids = []
            for _ in range(self.max_length):
                # 70% common tokens (0-1000), 30% rare tokens
                if random.random() < 0.7:
                    token_id = random.randint(0, 1000)
                else:
                    token_id = random.randint(1001, vocab_size - 1)
                token_ids.append(token_id)

            sequences.append(token_ids)

        return sequences


class Int8Calibrator:
    """
    Int8 calibration validator for M2 gate.

    Computes CE loss difference between fp32 and int8 teacher responses
    to validate quantization accuracy.
    """

    def __init__(self, teacher_client: TeacherClient):
        """
        Initialize calibrator.

        Args:
            teacher_client: TeacherClient instance
        """
        self.client = teacher_client

    def run_calibration(
        self,
        sequences: List[List[int]],
        topk: int = 128,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run calibration test on sequences.

        Args:
            sequences: List of token ID sequences
            topk: Number of top logits to request
            temperature: Temperature for softmax

        Returns:
            Calibration report dict with CE losses and delta
        """
        logger.info(
            f"Running calibration on {len(sequences)} sequences "
            f"(topk={topk}, temperature={temperature})"
        )

        # Query teacher for fp32 baseline
        logger.info("Querying teacher for fp32 baseline...")
        response_fp32 = self.client.query_topk(
            input_ids=sequences,
            topk=topk,
            return_dtype="float32",
            temperature=temperature
        )

        # Query teacher for int8 quantized
        logger.info("Querying teacher for int8 quantized...")
        response_int8 = self.client.query_topk(
            input_ids=sequences,
            topk=topk,
            return_dtype="int8",
            temperature=temperature
        )

        # Compute CE losses
        ce_fp32 = self._compute_ce_loss(response_fp32, sequences)
        ce_int8 = self._compute_ce_loss(response_int8, sequences)

        # Compute delta
        delta = abs(ce_fp32 - ce_int8)

        # M2 gate threshold
        threshold = 1e-3
        status = "PASS" if delta <= threshold else "FAIL"

        # Generate recommendations
        recommendations = []
        if status == "FAIL":
            recommendations.append(
                f"CE delta ({delta:.6f}) exceeds threshold ({threshold}). "
                f"Recommend increasing topk from {topk} to {topk * 2}."
            )

        report = {
            "ce_fp32": float(ce_fp32),
            "ce_int8": float(ce_int8),
            "delta": float(delta),
            "threshold": threshold,
            "status": status,
            "num_sequences": len(sequences),
            "topk": topk,
            "temperature": temperature,
            "recommendations": recommendations
        }

        logger.info(f"Calibration complete: {status}")
        logger.info(f"  CE (fp32): {ce_fp32:.6f}")
        logger.info(f"  CE (int8): {ce_int8:.6f}")
        logger.info(f"  Delta: {delta:.6f}")

        return report

    def _compute_ce_loss(
        self,
        response: TopKResponse,
        target_sequences: List[List[int]]
    ) -> float:
        """
        Compute cross-entropy loss from sparse top-k teacher response.

        The teacher sends:
        - topk_values: Unnormalized logits for top-k tokens
        - other_mass: Probability mass of tokens NOT in top-k

        To compute CE, we need probabilities:
        1. Apply softmax to top-k logits: p_unnorm = softmax(topk_values)
        2. Rescale to account for tail: p_topk = p_unnorm * (1 - other_mass)
        3. For each target: CE = -log(P(target))

        Args:
            response: TopKResponse from teacher
            target_sequences: Ground truth token sequences

        Returns:
            Mean CE loss across all positions
        """
        # Convert to tensors
        indices = torch.tensor(response.indices)  # [B, L, K]
        scale = torch.tensor(response.scale)  # [B, L]
        other_mass = torch.tensor(response.other_mass)  # [B, L]
        targets = torch.tensor(target_sequences)  # [B, L]

        # Step 1: Dequantize values based on return_dtype
        if hasattr(response, 'return_dtype') and response.return_dtype == "int8":
            values_int8 = torch.tensor(response.values_int8, dtype=torch.int8)
            values = values_int8.float() * scale.unsqueeze(-1)  # [B, L, K]
        else:
            # For backward compatibility or float types
            values = torch.tensor(response.values_int8).float()  # [B, L, K]
            # Check if these look like quantized values
            if values.abs().max() <= 127:
                values = values * scale.unsqueeze(-1)

        # Step 2: Compute probabilities for top-k tokens
        # softmax gives probabilities that sum to 1.0
        # We need them to sum to (1 - other_mass) to leave room for tail
        p_topk_unnorm = F.softmax(values, dim=-1)  # [B, L, K]
        p_topk = p_topk_unnorm * (1 - other_mass).unsqueeze(-1)  # [B, L, K]

        # Step 3: Compute CE loss for each position
        B, L, K = indices.shape
        ce_losses = []

        for b in range(B):
            for l in range(L - 1):  # L-1 because we predict next token
                target_token = targets[b, l + 1].item()
                topk_indices = indices[b, l].tolist()

                if target_token in topk_indices:
                    # Target is in top-k
                    k_idx = topk_indices.index(target_token)
                    prob = p_topk[b, l, k_idx].item()
                else:
                    # Target is in tail
                    prob = other_mass[b, l].item()

                # Compute CE with epsilon for numerical stability
                ce_loss = -np.log(prob + 1e-10)
                ce_losses.append(ce_loss)

        return float(np.mean(ce_losses))


def generate_report(
    calibration_result: Dict[str, Any],
    output_file: Optional[Path] = None
) -> str:
    """
    Generate human-readable calibration report.

    Args:
        calibration_result: Calibration report dict
        output_file: Optional path to save JSON report

    Returns:
        Human-readable report string
    """
    # Save JSON report if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(calibration_result, f, indent=2)

        logger.info(f"Saved JSON report to {output_file}")

    # Generate human-readable summary
    status_symbol = "✓" if calibration_result["status"] == "PASS" else "✗"

    report_lines = [
        "=" * 70,
        "INT8 CALIBRATION TEST REPORT (M2 Gate)",
        "=" * 70,
        "",
        f"Status: {status_symbol} {calibration_result['status']}",
        "",
        "Metrics:",
        f"  CE Loss (fp32):  {calibration_result['ce_fp32']:.6f}",
        f"  CE Loss (int8):  {calibration_result['ce_int8']:.6f}",
        f"  Delta:           {calibration_result['delta']:.6f}",
        f"  Threshold:       {calibration_result['threshold']:.6f}",
        "",
        "Configuration:",
        f"  Sequences:    {calibration_result['num_sequences']}",
        f"  Top-k:        {calibration_result['topk']}",
        f"  Temperature:  {calibration_result['temperature']}",
    ]

    if calibration_result["recommendations"]:
        report_lines.extend([
            "",
            "Recommendations:",
        ])
        for rec in calibration_result["recommendations"]:
            report_lines.append(f"  - {rec}")

    report_lines.extend([
        "",
        "=" * 70
    ])

    return "\n".join(report_lines)


def main():
    """Main entry point for calibration test."""
    parser = argparse.ArgumentParser(
        description="Int8 calibration test for M2 gate validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--teacher-url',
        type=str,
        default='http://localhost:8000/v1/topk',
        help='Teacher server endpoint URL (default: http://localhost:8000/v1/topk)'
    )

    parser.add_argument(
        '--num-sequences',
        type=int,
        default=128,
        help='Number of calibration sequences (default: 128)'
    )

    parser.add_argument(
        '--topk',
        type=int,
        default=128,
        help='Number of top logits to request (default: 128)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for softmax (default: 1.0)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to calibration data (JSONL/text file or directory). If not specified, uses synthetic data.'
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        default='meta-llama/Llama-3.2-1B',
        help='HuggingFace tokenizer name (default: meta-llama/Llama-3.2-1B)'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=4096,
        help='Maximum sequence length (default: 4096)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='calibration_report.json',
        help='Output JSON report path (default: calibration_report.json)'
    )

    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Request timeout in seconds (default: 60.0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    try:
        # Initialize dataset loader
        logger.info("Initializing calibration dataset loader...")
        loader = CalibrationDatasetLoader(
            data_path=Path(args.data_path) if args.data_path else None,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            seed=args.seed
        )

        # Load sequences
        logger.info(f"Loading {args.num_sequences} calibration sequences...")
        sequences = loader.load_sequences(args.num_sequences)
        logger.info(f"Loaded {len(sequences)} sequences of length {len(sequences[0])}")

        # Initialize teacher client
        logger.info(f"Connecting to teacher server at {args.teacher_url}...")
        client = TeacherClient(
            endpoint_url=args.teacher_url,
            timeout=args.timeout
        )

        # Check health
        if not client.health_check():
            logger.warning("Teacher health check failed, but continuing anyway...")

        # Initialize calibrator
        calibrator = Int8Calibrator(client)

        # Run calibration
        logger.info("Running calibration test...")
        result = calibrator.run_calibration(
            sequences=sequences,
            topk=args.topk,
            temperature=args.temperature
        )

        # Generate report
        report = generate_report(
            result,
            output_file=Path(args.output)
        )

        # Print to stdout
        print(report)

        # Exit with appropriate code
        sys.exit(0 if result["status"] == "PASS" else 1)

    except TeacherClientError as e:
        logger.error(f"Teacher client error: {e}")
        sys.exit(2)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(3)


if __name__ == '__main__':
    main()

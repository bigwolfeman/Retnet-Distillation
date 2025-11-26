"""
Evaluation runner and report generator.

Orchestrates multiple evaluation metrics (perplexity, NIAH) and generates
comprehensive reports.

Tasks: T058, T060
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from .perplexity import PerplexityEvaluator, PerplexityConfig
from .niah import NIAHEvaluator, NIAHConfig


logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates multiple evaluation metrics and generates reports.

    Runs multiple evaluations (perplexity, NIAH, custom metrics) and generates
    comprehensive reports with all results.

    Features:
    - Run multiple evaluations in sequence
    - Generate JSON and human-readable reports
    - Track evaluation history
    - Support custom evaluation functions

    Example:
        >>> runner = EvaluationRunner(model, tokenizer)
        >>> results = runner.run_all(
        ...     val_dataloader=val_loader,
        ...     perplexity_config=PerplexityConfig(),
        ...     niah_config=NIAHConfig(context_length=4096),
        ...     output_dir=Path("eval_results"),
        ... )
        >>> print(f"Perplexity: {results['perplexity']['perplexity']:.2f}")
        >>> print(f"NIAH accuracy: {results['niah']['accuracy']:.2%}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
    ):
        """Initialize evaluation runner.

        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on (default: CUDA if available)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize evaluators
        self.perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device)
        self.niah_evaluator = NIAHEvaluator(model, tokenizer, device)

        # Custom evaluators
        self.custom_evaluators: Dict[str, Callable] = {}

        logger.info(f"Initialized EvaluationRunner:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model: {type(self.model).__name__}")

    def register_custom_evaluator(
        self,
        name: str,
        evaluator_fn: Callable[[nn.Module, PreTrainedTokenizer], Dict[str, Any]],
    ):
        """Register a custom evaluation function.

        Args:
            name: Name of the custom evaluator
            evaluator_fn: Function that takes (model, tokenizer) and returns results dict

        Example:
            >>> def custom_eval(model, tokenizer):
            ...     # Custom evaluation logic
            ...     return {'metric': 0.95}
            >>> runner.register_custom_evaluator('custom', custom_eval)
        """
        self.custom_evaluators[name] = evaluator_fn
        logger.info(f"Registered custom evaluator: {name}")

    def run_perplexity(
        self,
        dataloader: DataLoader,
        config: Optional[PerplexityConfig] = None,
    ) -> Dict[str, Any]:
        """Run perplexity evaluation.

        Args:
            dataloader: DataLoader with evaluation data
            config: Perplexity configuration

        Returns:
            Perplexity results dictionary
        """
        logger.info("Running perplexity evaluation...")
        start_time = time.time()

        results = self.perplexity_evaluator.evaluate(dataloader, config)

        elapsed = time.time() - start_time
        results['elapsed_time'] = elapsed

        logger.info(f"Perplexity evaluation complete in {elapsed:.2f}s")
        return results

    def run_niah(
        self,
        config: Optional[NIAHConfig] = None,
    ) -> Dict[str, Any]:
        """Run NIAH evaluation.

        Args:
            config: NIAH configuration

        Returns:
            NIAH results dictionary
        """
        logger.info("Running NIAH evaluation...")
        start_time = time.time()

        results = self.niah_evaluator.evaluate(config)

        elapsed = time.time() - start_time
        results['elapsed_time'] = elapsed

        logger.info(f"NIAH evaluation complete in {elapsed:.2f}s")
        return results

    def run_all(
        self,
        val_dataloader: Optional[DataLoader] = None,
        perplexity_config: Optional[PerplexityConfig] = None,
        niah_config: Optional[NIAHConfig] = None,
        run_perplexity: bool = True,
        run_niah: bool = True,
        run_custom: bool = True,
        output_dir: Optional[Path] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run all evaluations and generate report.

        Args:
            val_dataloader: DataLoader for perplexity evaluation
            perplexity_config: Perplexity configuration
            niah_config: NIAH configuration
            run_perplexity: Whether to run perplexity evaluation
            run_niah: Whether to run NIAH evaluation
            run_custom: Whether to run custom evaluators
            output_dir: Directory to save reports (optional)
            step: Training step number (for report naming)

        Returns:
            Dictionary with all evaluation results

        Example:
            >>> results = runner.run_all(
            ...     val_dataloader=val_loader,
            ...     output_dir=Path("eval_results"),
            ...     step=5000,
            ... )
        """
        logger.info("=" * 80)
        logger.info("Running comprehensive evaluation...")
        logger.info("=" * 80)

        start_time = time.time()
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
        }

        # Run perplexity evaluation
        if run_perplexity and val_dataloader is not None:
            try:
                perplexity_results = self.run_perplexity(val_dataloader, perplexity_config)
                all_results['perplexity'] = perplexity_results
                # Bug #3 fix: Proper memory cleanup - delete tensors BEFORE empty_cache()
                # empty_cache() only releases cached memory, it doesn't free allocated tensors
                # We must explicitly delete tensor references first
                if torch.cuda.is_available():
                    # Delete result tensors if they exist
                    if 'perplexity' in all_results and isinstance(all_results['perplexity'], dict):
                        for key in list(all_results['perplexity'].keys()):
                            if torch.is_tensor(all_results['perplexity'][key]):
                                del all_results['perplexity'][key]
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                    # Now clear CUDA cache
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache after perplexity evaluation")
            except Exception as e:
                logger.error(f"Perplexity evaluation failed: {e}")
                all_results['perplexity'] = {'error': str(e)}
                # Clean up on error too
                if torch.cuda.is_available():
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

        # Run NIAH evaluation
        if run_niah:
            try:
                niah_results = self.run_niah(niah_config)
                all_results['niah'] = niah_results
                # Bug #3 fix: Proper memory cleanup - delete tensors BEFORE empty_cache()
                if torch.cuda.is_available():
                    # Delete result tensors if they exist
                    if 'niah' in all_results and isinstance(all_results['niah'], dict):
                        for key in list(all_results['niah'].keys()):
                            if torch.is_tensor(all_results['niah'][key]):
                                del all_results['niah'][key]
                    # Force garbage collection
                    import gc
                    gc.collect()
                    # Now clear CUDA cache
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache after NIAH evaluation")
            except Exception as e:
                logger.error(f"NIAH evaluation failed: {e}")
                all_results['niah'] = {'error': str(e)}
                # Clean up on error too
                if torch.cuda.is_available():
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

        # Run custom evaluators
        if run_custom:
            for name, evaluator_fn in self.custom_evaluators.items():
                try:
                    logger.info(f"Running custom evaluator: {name}")
                    custom_results = evaluator_fn(self.model, self.tokenizer)
                    all_results[name] = custom_results
                except Exception as e:
                    logger.error(f"Custom evaluator '{name}' failed: {e}")
                    all_results[name] = {'error': str(e)}

        # Total elapsed time
        total_elapsed = time.time() - start_time
        all_results['total_elapsed_time'] = total_elapsed

        # Bug #3 fix: Final memory cleanup after all evaluations
        if torch.cuda.is_available():
            # Delete any remaining tensor references in results
            import gc
            for key in list(all_results.keys()):
                if isinstance(all_results[key], dict):
                    for subkey in list(all_results[key].keys()):
                        if torch.is_tensor(all_results[key][subkey]):
                            del all_results[key][subkey]
            # Force garbage collection
            gc.collect()
            # Now clear cache
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"Final CUDA memory cleanup complete. Allocated: {mem_allocated:.2f} GB")

        # Ensure model is back in training mode
        self.model.train()

        logger.info("=" * 80)
        logger.info(f"Evaluation complete in {total_elapsed:.2f}s")
        logger.info("=" * 80)

        # Generate and save reports
        if output_dir:
            self.save_reports(all_results, output_dir, step)

        return all_results

    def save_reports(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        step: Optional[int] = None,
    ):
        """Save evaluation reports to disk.

        Generates:
        - JSON report with full results
        - Human-readable text report with summary

        Args:
            results: Evaluation results dictionary
            output_dir: Directory to save reports
            step: Training step number (for naming)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename prefix
        if step is not None:
            prefix = f"eval_{step:08d}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"eval_{timestamp}"

        # Save JSON report
        json_path = output_dir / f"{prefix}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON report to {json_path}")

        # Save human-readable report
        txt_path = output_dir / f"{prefix}.txt"
        report_text = self.generate_report_text(results)
        with open(txt_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved text report to {txt_path}")

    def generate_report_text(
        self,
        results: Dict[str, Any],
    ) -> str:
        """Generate human-readable report text.

        Args:
            results: Evaluation results dictionary

        Returns:
            Formatted report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Metadata
        lines.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
        lines.append(f"Step: {results.get('step', 'N/A')}")
        lines.append(f"Total time: {results.get('total_elapsed_time', 0):.2f}s")
        lines.append("")

        # Perplexity results
        if 'perplexity' in results:
            lines.append("-" * 80)
            lines.append("PERPLEXITY")
            lines.append("-" * 80)

            ppl_results = results['perplexity']
            if 'error' in ppl_results:
                lines.append(f"ERROR: {ppl_results['error']}")
            else:
                lines.append(f"Perplexity:      {ppl_results.get('perplexity', float('inf')):.2f}")
                lines.append(f"Loss (nats):     {ppl_results.get('loss', float('inf')):.4f}")
                lines.append(f"Bits/token:      {ppl_results.get('bits_per_token', float('inf')):.4f}")
                lines.append(f"Total tokens:    {ppl_results.get('total_tokens', 0):,}")
                lines.append(f"Num samples:     {ppl_results.get('num_samples', 0):,}")
                lines.append(f"Elapsed time:    {ppl_results.get('elapsed_time', 0):.2f}s")
            lines.append("")

        # NIAH results
        if 'niah' in results:
            lines.append("-" * 80)
            lines.append("NEEDLE-IN-A-HAYSTACK (NIAH)")
            lines.append("-" * 80)

            niah_results = results['niah']
            if 'error' in niah_results:
                lines.append(f"ERROR: {niah_results['error']}")
            else:
                lines.append(f"Overall accuracy: {niah_results.get('accuracy', 0):.2%}")
                lines.append(f"Total samples:    {niah_results.get('total_samples', 0)}")
                lines.append(f"Total correct:    {niah_results.get('total_correct', 0)}")
                lines.append(f"Elapsed time:     {niah_results.get('elapsed_time', 0):.2f}s")
                lines.append("")

                # Per-position accuracy
                position_accuracies = niah_results.get('position_accuracies', {})
                if position_accuracies:
                    lines.append("Position-wise accuracy:")
                    for pos, acc in sorted(position_accuracies.items()):
                        lines.append(f"  Position {float(pos):.1%}: {acc:.2%}")
            lines.append("")

        # Custom evaluators
        custom_keys = [k for k in results.keys()
                      if k not in ['timestamp', 'step', 'total_elapsed_time', 'perplexity', 'niah']]
        if custom_keys:
            lines.append("-" * 80)
            lines.append("CUSTOM EVALUATORS")
            lines.append("-" * 80)

            for key in custom_keys:
                custom_results = results[key]
                lines.append(f"\n{key.upper()}:")
                if 'error' in custom_results:
                    lines.append(f"  ERROR: {custom_results['error']}")
                else:
                    for k, v in custom_results.items():
                        lines.append(f"  {k}: {v}")
            lines.append("")

        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return '\n'.join(lines)

    def print_summary(
        self,
        results: Dict[str, Any],
    ):
        """Print evaluation summary to console.

        Args:
            results: Evaluation results dictionary
        """
        print(self.generate_report_text(results))


def run_evaluation(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    val_dataloader: Optional[DataLoader] = None,
    perplexity_config: Optional[PerplexityConfig] = None,
    niah_config: Optional[NIAHConfig] = None,
    output_dir: Optional[Path] = None,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """Convenience function to run full evaluation suite.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        val_dataloader: DataLoader for perplexity evaluation
        perplexity_config: Perplexity configuration
        niah_config: NIAH configuration
        output_dir: Directory to save reports
        step: Training step number

    Returns:
        Dictionary with all evaluation results

    Example:
        >>> from distillation.evaluation.runner import run_evaluation
        >>> from distillation.evaluation.perplexity import PerplexityConfig
        >>> from distillation.evaluation.niah import NIAHConfig
        >>>
        >>> results = run_evaluation(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     val_dataloader=val_loader,
        ...     perplexity_config=PerplexityConfig(max_samples=1000),
        ...     niah_config=NIAHConfig(context_length=4096, num_samples=50),
        ...     output_dir=Path("eval_results"),
        ...     step=5000,
        ... )
    """
    runner = EvaluationRunner(model, tokenizer)
    results = runner.run_all(
        val_dataloader=val_dataloader,
        perplexity_config=perplexity_config,
        niah_config=niah_config,
        output_dir=output_dir,
        step=step,
    )

    # Print summary
    runner.print_summary(results)

    return results

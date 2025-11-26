#!/usr/bin/env python3
"""
Curriculum-aware dataset downloader for knowledge distillation training.

This script intelligently downloads datasets based on a predefined curriculum,
checks existing data to avoid duplication, and provides progress tracking.

Target Curriculum (25B tokens total):
- 35% General instruction & chat: 8.75B tokens (~8-10M samples)
- 25% Code: 6.25B tokens (~6-8M samples)
- 20% Math/logic: 5.0B tokens (~5-6M samples)
- 10% Factual QA: 2.5B tokens (~2-3M samples)
- 10% General web (FineWeb): 2.5B tokens (already have 5M samples)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm


# Dataset source definitions with HuggingFace paths and target samples
INSTRUCTION_SOURCES = {
    'openhermes': {
        'path': 'teknium/OpenHermes-2.5',
        'target_samples': 2500000,  # Increased to help reach 8M target
        'license': 'apache-2.0',
        'description': 'High-quality instruction dataset from GPT-4',
        'text_field': 'conversations',
        'format': 'conversational'
    },
    'ultrachat': {
        'path': 'HuggingFaceH4/ultrachat_200k',
        'target_samples': 200000,
        'license': 'mit',
        'description': 'Multi-turn conversations from ChatGPT',
        'text_field': 'messages',
        'format': 'conversational'
    },
    'open_orca': {
        'path': 'Open-Orca/OpenOrca',
        'target_samples': 3000000,  # Increased to help reach 8M target
        'license': 'apache-2.0',
        'description': 'OpenAI-style instruction tuning',
        'text_field': 'system_prompt',
        'format': 'instruction'
    },
    'sharegpt': {
        'path': 'anon8231489123/ShareGPT_Vicuna_unfiltered',
        'target_samples': 1000000,  # Increased to help reach 8M target
        'license': 'cc-by-nc-4.0',
        'description': 'ShareGPT conversation dataset',
        'text_field': 'conversations',
        'format': 'conversational'
    },
    'alpaca_gpt4': {
        'path': 'vicgalle/alpaca-gpt4',
        'target_samples': 52000,
        'license': 'apache-2.0',
        'description': 'GPT-4 generated Alpaca dataset',
        'text_field': 'instruction',
        'format': 'instruction'
    },
    'dolly': {
        'path': 'databricks/databricks-dolly-15k',
        'target_samples': 15000,
        'license': 'cc-by-sa-3.0',
        'description': 'Human-generated instruction dataset',
        'text_field': 'instruction',
        'format': 'instruction'
    }
}

CODE_SOURCES = {
    'starcoder_python': {
        'path': 'bigcode/starcoderdata',
        'subset': 'python',
        'target_samples': 3000000,  # Increased to help reach 6M target
        'license': 'bigcode-openrail-m',
        'description': 'Python code from GitHub',
        'text_field': 'content',
        'format': 'code'
    },
    'the_stack_python': {
        'path': 'bigcode/the-stack-v2',
        'subset': 'python',
        'target_samples': 2000000,  # Increased to help reach 6M target
        'license': 'bigcode-openrail-m',
        'description': 'The Stack v2 Python subset',
        'text_field': 'content',
        'format': 'code'
    },
    'code_alpaca': {
        'path': 'sahil2801/CodeAlpaca-20k',
        'target_samples': 100000,  # Increased
        'license': 'apache-2.0',
        'description': 'Code instruction dataset',
        'text_field': 'instruction',
        'format': 'instruction'
    },
    'code_contests': {
        'path': 'deepmind/code_contests',
        'target_samples': 200000,  # Increased
        'license': 'apache-2.0',
        'description': 'Programming contest problems',
        'text_field': 'description',
        'format': 'problem'
    },
    'apps': {
        'path': 'codeparrot/apps',
        'target_samples': 500000,  # Increased to help reach 6M target
        'license': 'mit',
        'description': 'Python programming problems',
        'text_field': 'question',
        'format': 'problem'
    },
    'code_search_net': {
        'path': 'code_search_net',
        'subset': 'python',
        'target_samples': 500000,  # Increased to help reach 6M target
        'license': 'mit',
        'description': 'Code with docstrings',
        'text_field': 'func_code_string',
        'format': 'code'
    }
}

MATH_SOURCES = {
    'numina_cot': {
        'path': 'AI-MO/NuminaMath-CoT',
        'target_samples': 3000000,  # Increased to help reach 5M target
        'license': 'apache-2.0',
        'description': 'Math problems with chain-of-thought',
        'text_field': 'problem',
        'format': 'problem'
    },
    'gsm8k': {
        'path': 'openai/gsm8k',
        'subset': 'main',
        'target_samples': 100000,  # Increased (even though original is smaller, will stream)
        'license': 'mit',
        'description': 'Grade school math problems',
        'text_field': 'question',
        'format': 'problem'
    },
    'math': {
        'path': 'lighteval/MATH',
        'target_samples': 100000,  # Increased
        'license': 'mit',
        'description': 'Competition math problems',
        'text_field': 'problem',
        'format': 'problem'
    },
    'metamath': {
        'path': 'meta-math/MetaMathQA',
        'target_samples': 1000000,  # Increased to help reach 5M target
        'license': 'mit',
        'description': 'Augmented math QA dataset',
        'text_field': 'query',
        'format': 'problem'
    },
    'orca_math': {
        'path': 'microsoft/orca-math-word-problems-200k',
        'target_samples': 600000,  # Increased to help reach 5M target
        'license': 'mit',
        'description': 'Math word problems',
        'text_field': 'question',
        'format': 'problem'
    },
    'mathinstruct': {
        'path': 'TIGER-Lab/MathInstruct',
        'target_samples': 500000,  # Increased to help reach 5M target
        'license': 'cc-by-4.0',
        'description': 'Math instruction tuning',
        'text_field': 'instruction',
        'format': 'instruction'
    }
}

FACTUAL_QA_SOURCES = {
    'natural_questions': {
        'path': 'google-research-datasets/natural_questions',
        'target_samples': 800000,  # Increased to help reach 2.5M target
        'license': 'apache-2.0',
        'description': 'Google natural questions',
        'text_field': 'question',
        'format': 'qa'
    },
    'hotpot_qa': {
        'path': 'hotpot_qa',
        'subset': 'distractor',
        'target_samples': 500000,  # Increased to help reach 2.5M target
        'license': 'cc-by-sa-4.0',
        'description': 'Multi-hop reasoning QA',
        'text_field': 'question',
        'format': 'qa'
    },
    'squad_v2': {
        'path': 'rajpurkar/squad_v2',
        'target_samples': 400000,  # Increased to help reach 2.5M target
        'license': 'cc-by-sa-4.0',
        'description': 'Reading comprehension QA',
        'text_field': 'question',
        'format': 'qa'
    },
    'trivia_qa': {
        'path': 'mandarjoshi/trivia_qa',
        'subset': 'rc.wikipedia',
        'target_samples': 700000,  # Increased to help reach 2.5M target
        'license': 'apache-2.0',
        'description': 'Trivia questions with evidence',
        'text_field': 'question',
        'format': 'qa'
    },
    'web_questions': {
        'path': 'web_questions',
        'target_samples': 100000,  # Increased to help reach 2.5M target
        'license': 'cc-by-4.0',
        'description': 'Web-based questions',
        'text_field': 'question',
        'format': 'qa'
    }
}


class CurriculumDownloader:
    """Smart dataset downloader with curriculum awareness."""

    def __init__(self, data_dir: str = 'data/distillation'):
        """Initialize the downloader.

        Args:
            data_dir: Base directory for distillation data
        """
        self.data_dir = Path(data_dir)
        self.categories = {
            'instruction_chat': {
                'target_samples': 8000000,
                'target_tokens': 8750000000,
                'sources': INSTRUCTION_SOURCES,
                'percentage': 35
            },
            'code': {
                'target_samples': 6000000,
                'target_tokens': 6250000000,
                'sources': CODE_SOURCES,
                'percentage': 25
            },
            'math_reasoning': {
                'target_samples': 5000000,
                'target_tokens': 5000000000,
                'sources': MATH_SOURCES,
                'percentage': 20
            },
            'factual_qa': {
                'target_samples': 2500000,
                'target_tokens': 2500000000,
                'sources': FACTUAL_QA_SOURCES,
                'percentage': 10
            }
        }

        # Average tokens per sample (rough estimates)
        self.avg_tokens_per_sample = {
            'instruction_chat': 1000,
            'code': 1000,
            'math_reasoning': 1000,
            'factual_qa': 1000
        }

    def scan_existing(self) -> Dict[str, Dict]:
        """Scan existing downloads and count samples.

        Returns:
            Dictionary with existing sample counts per category
        """
        print("Scanning existing data...\n")

        for category, info in self.categories.items():
            category_dir = self.data_dir / category

            if not category_dir.exists():
                info['existing_samples'] = 0
                info['existing_sources'] = {}
                info['gap_samples'] = info['target_samples']
                continue

            # Count samples per source file
            existing_sources = {}
            total_samples = 0

            for jsonl_file in category_dir.glob('*.jsonl'):
                source_name = jsonl_file.stem

                # Count lines in JSONL
                try:
                    with open(jsonl_file, 'r') as f:
                        count = sum(1 for _ in f)
                    existing_sources[source_name] = count
                    total_samples += count
                except Exception as e:
                    print(f"  Warning: Could not read {jsonl_file}: {e}")

            info['existing_samples'] = total_samples
            info['existing_sources'] = existing_sources
            info['gap_samples'] = max(0, info['target_samples'] - total_samples)

            # Calculate estimated tokens
            info['existing_tokens'] = total_samples * self.avg_tokens_per_sample[category]
            info['gap_tokens'] = info['gap_samples'] * self.avg_tokens_per_sample[category]

        return self.categories

    def print_status(self):
        """Print current status of all categories."""
        print("=" * 80)
        print("CURRICULUM DATASET STATUS")
        print("=" * 80)

        total_existing = 0
        total_target = 0
        total_gap = 0

        for category, info in self.categories.items():
            existing = info['existing_samples']
            target = info['target_samples']
            gap = info['gap_samples']
            pct_complete = (existing / target * 100) if target > 0 else 0

            total_existing += existing
            total_target += target
            total_gap += gap

            status = "✓" if gap == 0 else "⚠"

            print(f"\n{status} {category.upper().replace('_', ' ')}")
            print(f"  Target:   {target:>12,} samples ({info['percentage']:>2}% of curriculum)")
            print(f"  Existing: {existing:>12,} samples ({pct_complete:>5.1f}% complete)")
            print(f"  Gap:      {gap:>12,} samples")

            # Show existing sources
            if info['existing_sources']:
                print(f"  Sources downloaded:")
                for source, count in sorted(info['existing_sources'].items()):
                    print(f"    - {source}: {count:,} samples")

        # Overall summary
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        overall_pct = (total_existing / total_target * 100) if total_target > 0 else 0
        print(f"Total existing: {total_existing:>12,} / {total_target:,} samples ({overall_pct:.1f}%)")
        print(f"Total gap:      {total_gap:>12,} samples")

        # Storage estimate
        avg_bytes_per_sample = 1500  # Rough estimate
        gap_gb = (total_gap * avg_bytes_per_sample) / (1024 ** 3)
        print(f"Estimated storage needed: {gap_gb:.1f} GB")
        print("=" * 80)

    def calculate_download_plan(self, category: str) -> List[Tuple[str, Dict, int]]:
        """Calculate how many samples to download from each source.

        Args:
            category: Category to calculate plan for

        Returns:
            List of (source_name, source_info, samples_needed) tuples
        """
        info = self.categories[category]
        gap = info['gap_samples']

        if gap <= 0:
            return []

        # Get sources that aren't fully downloaded yet
        sources = info['sources']
        existing = info['existing_sources']

        plan = []
        for source_name, source_info in sources.items():
            target = source_info['target_samples']
            current = existing.get(source_name, 0)
            needed = max(0, target - current)

            if needed > 0:
                plan.append((source_name, source_info, needed))

        return plan

    def format_sample(self, raw_sample: Dict, source_info: Dict,
                     source_name: str, idx: int, category: str) -> Optional[Dict]:
        """Format a raw dataset sample into standard format.

        Args:
            raw_sample: Raw sample from HuggingFace dataset
            source_info: Source configuration
            source_name: Name of the source dataset
            idx: Sample index
            category: Category name

        Returns:
            Formatted sample or None if formatting fails
        """
        try:
            # Extract text based on format
            text_format = source_info.get('format', 'text')

            if text_format == 'conversational':
                # Handle conversational format (messages/conversations)
                text_field = source_info.get('text_field', 'conversations')
                if text_field in raw_sample:
                    convs = raw_sample[text_field]
                    if isinstance(convs, list):
                        text = '\n'.join([f"{msg.get('from', 'unknown')}: {msg.get('value', '')}"
                                        for msg in convs])
                    else:
                        text = str(convs)
                else:
                    return None

            elif text_format == 'instruction':
                # Handle instruction format
                instruction = raw_sample.get('instruction', '')
                input_text = raw_sample.get('input', '')
                output = raw_sample.get('output', '')

                if input_text:
                    text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
                else:
                    text = f"Instruction: {instruction}\nOutput: {output}"

            elif text_format == 'problem':
                # Handle problem format (math, code)
                question = raw_sample.get('question', raw_sample.get('problem', ''))
                answer = raw_sample.get('answer', raw_sample.get('solution', ''))
                text = f"Problem: {question}\nSolution: {answer}"

            elif text_format == 'qa':
                # Handle QA format
                question = raw_sample.get('question', '')

                # Safe handling of answers field
                answers_data = raw_sample.get('answers', {})
                if isinstance(answers_data, dict):
                    text_list = answers_data.get('text', [])
                    answer = text_list[0] if text_list else ''
                else:
                    answer = raw_sample.get('answer', '')

                context = raw_sample.get('context', '')

                if context:
                    text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
                else:
                    text = f"Question: {question}\nAnswer: {answer}"

            elif text_format == 'code':
                # Handle raw code
                text_field = source_info.get('text_field', 'content')
                text = raw_sample.get(text_field, '')
            else:
                # Default: try to find any text field
                text_field = source_info.get('text_field', 'text')
                text = raw_sample.get(text_field, str(raw_sample))

            # Skip if text is too short
            if len(text.strip()) < 50:
                return None

            # Create standardized format
            sample = {
                'text': text,
                'id': f"{source_name}_{idx:08d}",
                'source': source_name,
                'category': category,
                'metadata': {
                    'num_chars': len(text),
                    'format': text_format
                }
            }

            return sample

        except Exception as e:
            print(f"    Warning: Failed to format sample {idx}: {e}")
            return None

    def download_source(self, category: str, source_name: str,
                       source_info: Dict, samples_needed: int,
                       dry_run: bool = False) -> int:
        """Download samples from a specific source.

        Args:
            category: Category name
            source_name: Source dataset name
            source_info: Source configuration
            samples_needed: Number of samples to download
            dry_run: If True, don't actually download

        Returns:
            Number of samples actually downloaded
        """
        output_dir = self.data_dir / category
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{source_name}.jsonl"
        stats_file = output_dir / f"{source_name}.stats.json"

        if dry_run:
            print(f"  [DRY RUN] Would download {samples_needed:,} samples from {source_info['path']}")
            print(f"            License: {source_info['license']}")
            print(f"            Output: {output_file}")
            return samples_needed

        print(f"\nDownloading {source_name}...")
        print(f"  Source: {source_info['path']}")
        print(f"  License: {source_info['license']}")
        print(f"  Target: {samples_needed:,} samples")

        try:
            # Load dataset
            dataset_path = source_info['path']
            subset = source_info.get('subset')

            if subset:
                dataset = load_dataset(dataset_path, subset, split='train', streaming=True)
            else:
                # Try to load train split first, fall back to other splits
                try:
                    dataset = load_dataset(dataset_path, split='train', streaming=True)
                except Exception:
                    # Try loading without split specification
                    dataset = load_dataset(dataset_path, streaming=True)
                    if isinstance(dataset, dict):
                        # Use first available split
                        dataset = dataset[list(dataset.keys())[0]]

            # Download and format samples
            downloaded = 0
            skipped = 0
            total_chars = 0
            min_length = float('inf')
            max_length = 0

            # Check if file exists and append mode needed
            mode = 'a' if output_file.exists() else 'w'
            existing_count = 0
            if mode == 'a':
                with open(output_file, 'r') as f:
                    existing_count = sum(1 for _ in f)
                print(f"  Appending to existing file ({existing_count:,} samples)")

            with open(output_file, mode) as f:
                # Use tqdm for progress bar
                pbar = tqdm(total=samples_needed, desc=f"  {source_name}", unit="samples")

                for idx, raw_sample in enumerate(dataset):
                    if downloaded >= samples_needed:
                        break

                    # Format sample
                    sample = self.format_sample(raw_sample, source_info, source_name,
                                              existing_count + idx, category)

                    if sample is None:
                        skipped += 1
                        continue

                    # Write to file
                    f.write(json.dumps(sample) + '\n')

                    # Update stats
                    downloaded += 1
                    text_len = len(sample['text'])
                    total_chars += text_len
                    min_length = min(min_length, text_len)
                    max_length = max(max_length, text_len)

                    pbar.update(1)

                pbar.close()

            # Save stats
            stats = {
                'samples': downloaded + existing_count,
                'new_samples': downloaded,
                'total_chars': total_chars,
                'min_length': min_length if min_length != float('inf') else 0,
                'max_length': max_length,
                'skipped': skipped,
                'avg_length': total_chars / downloaded if downloaded > 0 else 0,
                'size_mb': output_file.stat().st_size / (1024 * 1024),
                'output_file': str(output_file),
                'license': source_info['license']
            }

            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"  ✓ Downloaded {downloaded:,} samples ({skipped:,} skipped)")
            print(f"    Avg length: {stats['avg_length']:.0f} chars")
            print(f"    File size: {stats['size_mb']:.1f} MB")

            return downloaded

        except Exception as e:
            print(f"  ✗ Error downloading {source_name}: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def download_category(self, category: str, dry_run: bool = False) -> int:
        """Download missing data for a category.

        Args:
            category: Category to download
            dry_run: If True, only show what would be downloaded

        Returns:
            Total number of samples downloaded
        """
        if category not in self.categories:
            print(f"Error: Unknown category '{category}'")
            return 0

        info = self.categories[category]
        gap = info['gap_samples']

        if gap <= 0:
            print(f"\n✓ {category}: Already complete ({info['existing_samples']:,} samples)")
            return 0

        print(f"\n{'=' * 80}")
        print(f"DOWNLOADING: {category.upper().replace('_', ' ')}")
        print(f"{'=' * 80}")
        print(f"Gap: {gap:,} samples needed")

        # Get download plan
        plan = self.calculate_download_plan(category)

        if not plan:
            print("No sources available to download from.")
            return 0

        print(f"\nDownload plan ({len(plan)} sources):")
        for source_name, source_info, needed in plan:
            print(f"  - {source_name}: {needed:,} samples")

        if dry_run:
            print("\n[DRY RUN] Not actually downloading.")
            return sum(needed for _, _, needed in plan)

        # Download from each source
        total_downloaded = 0
        for source_name, source_info, needed in plan:
            downloaded = self.download_source(category, source_name, source_info,
                                             needed, dry_run=dry_run)
            total_downloaded += downloaded

        print(f"\n✓ Downloaded {total_downloaded:,} total samples for {category}")
        return total_downloaded

    def download_all(self, dry_run: bool = False):
        """Download missing data for all categories.

        Args:
            dry_run: If True, only show what would be downloaded
        """
        for category in self.categories.keys():
            self.download_category(category, dry_run=dry_run)

        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)

        # Re-scan and show final status
        self.scan_existing()
        self.print_status()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download curriculum datasets for knowledge distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current status and what would be downloaded
  python download_curriculum_datasets.py --dry-run

  # Download all missing data
  python download_curriculum_datasets.py

  # Download only code datasets
  python download_curriculum_datasets.py --category code

  # Force re-download (ignoring existing data)
  python download_curriculum_datasets.py --force --category instruction_chat
        """
    )

    parser.add_argument(
        '--data-dir',
        default='data/distillation',
        help='Base directory for distillation data (default: data/distillation)'
    )

    parser.add_argument(
        '--category',
        choices=['instruction_chat', 'code', 'math_reasoning', 'factual_qa'],
        help='Download only this category (default: all)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download (ignore existing data)'
    )

    parser.add_argument(
        '--status-only',
        action='store_true',
        help='Only show status, do not download anything'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = CurriculumDownloader(data_dir=args.data_dir)

    # If force mode, clear existing data tracking
    if args.force:
        print("Force mode: Will re-download all data\n")
        for category in downloader.categories.values():
            category['existing_samples'] = 0
            category['existing_sources'] = {}
            category['gap_samples'] = category['target_samples']
    else:
        # Scan existing data
        downloader.scan_existing()

    # Show status
    downloader.print_status()

    # If status-only mode, exit here
    if args.status_only:
        return

    # Download data
    if args.category:
        downloader.download_category(args.category, dry_run=args.dry_run)
    else:
        downloader.download_all(dry_run=args.dry_run)


if __name__ == '__main__':
    main()

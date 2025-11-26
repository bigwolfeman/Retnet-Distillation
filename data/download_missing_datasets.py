#!/usr/bin/env python3
"""
Quick downloader for missing datasets to balance the POC dataset.
Downloads modest amounts to fill gaps without taking forever.
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_metamath(output_dir: Path, max_samples: int = 100000):
    """Download MetaMath dataset to fix the empty file."""
    print(f"\n{'='*60}")
    print("Downloading MetaMath dataset")
    print(f"{'='*60}")

    output_file = output_dir / "math_reasoning" / "metamath.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)

        downloaded = 0
        with open(output_file, 'w') as f:
            for idx, sample in enumerate(tqdm(dataset, total=max_samples, desc="MetaMath")):
                if downloaded >= max_samples:
                    break

                # Format: Problem + Solution
                problem = sample.get('query', sample.get('question', ''))
                solution = sample.get('response', sample.get('answer', ''))

                if len(problem) < 20 or len(solution) < 20:
                    continue

                formatted = {
                    'text': f"Problem: {problem}\n\nSolution: {solution}",
                    'id': f"metamath_{idx:08d}",
                    'source': 'metamath',
                    'category': 'math_reasoning'
                }

                f.write(json.dumps(formatted) + '\n')
                downloaded += 1

        print(f"✓ Downloaded {downloaded:,} MetaMath samples")
        print(f"  Output: {output_file}")
        return downloaded

    except Exception as e:
        print(f"✗ Error downloading MetaMath: {e}")
        import traceback
        traceback.print_exc()
        return 0


def download_code_search_net(output_dir: Path, max_samples: int = 100000):
    """Download CodeSearchNet Python for code diversity."""
    print(f"\n{'='*60}")
    print("Downloading CodeSearchNet (Python)")
    print(f"{'='*60}")

    output_file = output_dir / "code" / "code_search_net.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset("code_search_net", "python", split="train", streaming=True)

        downloaded = 0
        with open(output_file, 'w') as f:
            for idx, sample in enumerate(tqdm(dataset, total=max_samples, desc="CodeSearchNet")):
                if downloaded >= max_samples:
                    break

                code = sample.get('func_code_string', sample.get('whole_func_string', ''))
                docstring = sample.get('func_documentation_string', '')

                if len(code) < 50:
                    continue

                # Include docstring if available
                if docstring:
                    text = f"# {docstring}\n\n{code}"
                else:
                    text = code

                formatted = {
                    'text': text,
                    'id': f"codesearchnet_{idx:08d}",
                    'source': 'code_search_net',
                    'category': 'code'
                }

                f.write(json.dumps(formatted) + '\n')
                downloaded += 1

        print(f"✓ Downloaded {downloaded:,} CodeSearchNet samples")
        print(f"  Output: {output_file}")
        return downloaded

    except Exception as e:
        print(f"✗ Error downloading CodeSearchNet: {e}")
        import traceback
        traceback.print_exc()
        return 0


def download_apps(output_dir: Path, max_samples: int = 50000):
    """Download APPS programming problems."""
    print(f"\n{'='*60}")
    print("Downloading APPS (programming problems)")
    print(f"{'='*60}")

    output_file = output_dir / "code" / "apps.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset("codeparrot/apps", split="train", streaming=True)

        downloaded = 0
        with open(output_file, 'w') as f:
            for idx, sample in enumerate(tqdm(dataset, total=max_samples, desc="APPS")):
                if downloaded >= max_samples:
                    break

                problem = sample.get('question', sample.get('problem', ''))
                solutions = sample.get('solutions', '')

                if len(problem) < 50:
                    continue

                # Format problem with first solution if available
                if solutions and isinstance(solutions, str) and len(solutions) > 0:
                    text = f"Problem: {problem}\n\nSolution:\n{solutions}"
                else:
                    text = f"Problem: {problem}"

                formatted = {
                    'text': text,
                    'id': f"apps_{idx:08d}",
                    'source': 'apps',
                    'category': 'code'
                }

                f.write(json.dumps(formatted) + '\n')
                downloaded += 1

        print(f"✓ Downloaded {downloaded:,} APPS samples")
        print(f"  Output: {output_file}")
        return downloaded

    except Exception as e:
        print(f"✗ Error downloading APPS: {e}")
        import traceback
        traceback.print_exc()
        return 0


def download_mbpp(output_dir: Path, max_samples: int = 50000):
    """Download MBPP (Mostly Basic Python Problems)."""
    print(f"\n{'='*60}")
    print("Downloading MBPP (Python problems)")
    print(f"{'='*60}")

    output_file = output_dir / "code" / "mbpp.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # MBPP doesn't support streaming, load full dataset
        dataset = load_dataset("google-research-datasets/mbpp", "full", split="train")

        downloaded = 0
        with open(output_file, 'w') as f:
            for idx, sample in enumerate(tqdm(dataset, desc="MBPP")):
                if downloaded >= max_samples:
                    break

                problem = sample.get('text', sample.get('prompt', ''))
                code = sample.get('code', '')
                test_list = sample.get('test_list', [])

                if len(problem) < 20:
                    continue

                # Format with problem, solution, and tests
                text = f"Problem: {problem}\n\nSolution:\n{code}"
                if test_list:
                    text += f"\n\nTests:\n" + "\n".join(test_list[:3])

                formatted = {
                    'text': text,
                    'id': f"mbpp_{idx:08d}",
                    'source': 'mbpp',
                    'category': 'code'
                }

                f.write(json.dumps(formatted) + '\n')
                downloaded += 1

        print(f"✓ Downloaded {downloaded:,} MBPP samples")
        print(f"  Output: {output_file}")
        return downloaded

    except Exception as e:
        print(f"✗ Error downloading MBPP: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Download all missing datasets."""
    output_dir = Path("distillation")

    print("="*60)
    print("POC Dataset Gap Filler")
    print("="*60)
    print("\nDownloading missing datasets to balance the POC:")
    print("  1. MetaMath (100K samples) - fix empty file")
    print("  2. CodeSearchNet Python (100K samples) - add code diversity")
    print("  3. APPS (50K samples) - programming problems")
    print("  4. MBPP (50K samples) - Python problems")
    print("\nEstimated addition: ~300-500M tokens")
    print("="*60)

    total = 0

    # Download each dataset
    total += download_metamath(output_dir, max_samples=100000)
    total += download_code_search_net(output_dir, max_samples=100000)
    total += download_apps(output_dir, max_samples=50000)
    total += download_mbpp(output_dir, max_samples=50000)

    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE!")
    print(f"{'='*60}")
    print(f"Total new samples: {total:,}")
    print(f"Estimated tokens: ~{total * 1000:,} ({total * 1000 / 1e9:.2f}B)")
    print(f"\nNext step: Run tokenizer on new datasets:")
    print(f"  python data/preprocess_to_parquet.py --input data/distillation --output data/distillation_preprocessed --recursive --skip-existing")


if __name__ == '__main__':
    main()

"""Index building CLI for RetNet-HRM.

This script builds retrieval indexes for code chunks. It supports two types:
1. Workspace index: HNSW-based, for local project code (2-8GB repos)
2. Global index: FAISS-based, for general knowledge (large code corpora)

Usage:
    # Build workspace index for a local project
    python src/cli/index.py \
        --type workspace \
        --source-dir /path/to/project \
        --encoder-checkpoint checkpoints/dual_encoder.pt \
        --output indexes/my_project

    # Build global index for a large corpus
    python src/cli/index.py \
        --type global \
        --source-dir /path/to/corpus \
        --encoder-checkpoint checkpoints/dual_encoder.pt \
        --output indexes/global_knowledge

    # Resume building (add to existing index)
    python src/cli/index.py \
        --type workspace \
        --source-dir /path/to/project \
        --encoder-checkpoint checkpoints/dual_encoder.pt \
        --output indexes/my_project \
        --load-existing

Features:
- Scans source directory for code files (Python, JavaScript, TypeScript, etc.)
- Extracts chunks using simple function/class-based chunking
- Encodes chunks with dual encoder
- Builds FAISS (global) or HNSW (workspace) index
- Saves index + chunks to disk
- Supports incremental updates
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple
import torch
import numpy as np
from tqdm import tqdm
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval_index.code_chunk import CodeChunk
from src.retrieval_index.index import RetrievalIndex
from src.retrieval_index.dual_encoder import DualEncoder
from src.data.tokenizer import get_tokenizer


# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.hpp': 'cpp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.sh': 'bash',
    '.md': 'markdown',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json',
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build retrieval index for code chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Index type
    parser.add_argument(
        "--type",
        type=str,
        choices=["workspace", "global"],
        required=True,
        help="Index type: workspace (HNSW, local) or global (FAISS, large corpus)"
    )

    # Source directory
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory to scan for code files"
    )

    # Encoder checkpoint
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        required=True,
        help="Path to dual encoder checkpoint (.pt file)"
    )

    # Output path
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for index (without extension)"
    )

    # Optional parameters
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing index and add new chunks (incremental update)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Maximum chunk size in bytes (default: 1024, max: 2048)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for encoding (cuda/cpu)"
    )

    parser.add_argument(
        "--exclude-patterns",
        type=str,
        nargs="+",
        default=[],
        help="File/directory patterns to exclude (e.g., 'test_*' '*/node_modules/*')"
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )

    # FAISS-specific parameters
    parser.add_argument(
        "--faiss-centroids",
        type=int,
        default=1024,
        help="Number of IVF centroids for FAISS (default: 1024)"
    )

    parser.add_argument(
        "--faiss-subquantizers",
        type=int,
        default=64,
        help="Number of PQ subquantizers for FAISS (default: 64)"
    )

    # HNSW-specific parameters
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW M parameter (default: 32)"
    )

    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=200,
        help="HNSW ef_construction parameter (default: 200)"
    )

    return parser.parse_args()


def should_exclude(path: Path, exclude_patterns: List[str]) -> bool:
    """Check if path matches any exclude pattern."""
    path_str = str(path)
    for pattern in exclude_patterns:
        if pattern in path_str:
            return True
    return False


def scan_directory(
    source_dir: str,
    exclude_patterns: List[str],
    max_files: int = None
) -> List[Tuple[Path, str]]:
    """
    Scan directory for code files.

    Args:
        source_dir: Directory to scan
        exclude_patterns: Patterns to exclude
        max_files: Maximum number of files to process

    Returns:
        List of (file_path, language) tuples
    """
    print(f"\n{'='*60}")
    print(f"Scanning directory: {source_dir}")
    print(f"{'='*60}")

    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    files = []

    # Default exclude patterns
    default_excludes = [
        '__pycache__',
        '.git',
        'node_modules',
        '.venv',
        'venv',
        'build',
        'dist',
        '.egg-info',
        '__pycache__',
    ]
    all_excludes = default_excludes + exclude_patterns

    # Walk directory
    for root, dirs, filenames in os.walk(source_path):
        # Filter out excluded directories (modify in-place)
        dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d, all_excludes)]

        for filename in filenames:
            file_path = Path(root) / filename

            # Check if excluded
            if should_exclude(file_path, all_excludes):
                continue

            # Check extension
            ext = file_path.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                language = SUPPORTED_EXTENSIONS[ext]
                files.append((file_path, language))

                # Stop if max_files reached
                if max_files and len(files) >= max_files:
                    break

        if max_files and len(files) >= max_files:
            break

    print(f"Found {len(files)} code files")

    # Print language distribution
    lang_counts = {}
    for _, lang in files:
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print("\nLanguage distribution:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")

    return files


def simple_chunk_file(
    file_path: Path,
    language: str,
    chunk_size: int,
    source_dir: Path
) -> List[CodeChunk]:
    """
    Simple chunking strategy: split by functions/classes or by lines.

    For now, this is a simple line-based chunking. In production, you'd want
    to use tree-sitter or AST parsing for better structure-aware chunking.

    Args:
        file_path: Path to file
        language: Programming language
        chunk_size: Maximum chunk size in bytes
        source_dir: Base source directory (for relative paths)

    Returns:
        List of CodeChunk objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")
        return []

    if not content.strip():
        return []

    chunks = []
    lines = content.split('\n')

    # Get relative path
    try:
        rel_path = file_path.relative_to(source_dir)
    except ValueError:
        rel_path = file_path

    # Simple strategy: group lines into chunks
    current_chunk = []
    current_size = 0
    start_line = 1

    for i, line in enumerate(lines, 1):
        line_bytes = len(line.encode('utf-8')) + 1  # +1 for newline

        # Check if adding this line would exceed chunk_size
        if current_size + line_bytes > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = '\n'.join(current_chunk)

            # Generate chunk ID
            chunk_id = f"{rel_path}:{start_line}-{i-1}"

            # Create chunk (without embedding - will be added later)
            chunk = CodeChunk(
                chunk_id=chunk_id,
                source_type="file",  # Could be 'function', 'class' with better parsing
                text=chunk_text,
                language=language,
                file_path=str(rel_path),
                start_line=start_line,
                end_line=i-1,
            )

            # Validate chunk size
            try:
                chunk.validate()
                chunks.append(chunk)
            except AssertionError as e:
                # Skip chunks that are too large
                print(f"Warning: Skipping oversized chunk in {file_path}:{start_line}-{i-1}")

            # Start new chunk
            current_chunk = [line]
            current_size = line_bytes
            start_line = i
        else:
            current_chunk.append(line)
            current_size += line_bytes

    # Add remaining chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunk_id = f"{rel_path}:{start_line}-{len(lines)}"

        chunk = CodeChunk(
            chunk_id=chunk_id,
            source_type="file",
            text=chunk_text,
            language=language,
            file_path=str(rel_path),
            start_line=start_line,
            end_line=len(lines),
        )

        try:
            chunk.validate()
            chunks.append(chunk)
        except AssertionError:
            # Skip if too large
            pass

    return chunks


def extract_chunks(
    files: List[Tuple[Path, str]],
    chunk_size: int,
    source_dir: Path
) -> List[CodeChunk]:
    """
    Extract chunks from files.

    Args:
        files: List of (file_path, language) tuples
        chunk_size: Maximum chunk size in bytes
        source_dir: Base source directory

    Returns:
        List of CodeChunk objects
    """
    print(f"\n{'='*60}")
    print(f"Extracting chunks (max size: {chunk_size} bytes)")
    print(f"{'='*60}")

    all_chunks = []

    for file_path, language in tqdm(files, desc="Extracting chunks"):
        file_chunks = simple_chunk_file(file_path, language, chunk_size, source_dir)
        all_chunks.extend(file_chunks)

    print(f"\nExtracted {len(all_chunks)} chunks")

    # Print statistics
    chunk_sizes = [len(c.text.encode('utf-8')) for c in all_chunks]
    if chunk_sizes:
        print(f"Chunk size stats:")
        print(f"  Min: {min(chunk_sizes)} bytes")
        print(f"  Max: {max(chunk_sizes)} bytes")
        print(f"  Avg: {sum(chunk_sizes) / len(chunk_sizes):.1f} bytes")

    return all_chunks


def load_encoder(checkpoint_path: str, device: str) -> Tuple[DualEncoder, any]:
    """
    Load dual encoder from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (encoder, tokenizer)
    """
    print(f"\n{'='*60}")
    print(f"Loading dual encoder")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        vocab_size = config.get('vocab_size', 50000)
        d_model = config.get('d_model', 768)
    else:
        # Default values
        vocab_size = 50000
        d_model = 768
        print("Warning: No model config in checkpoint, using defaults")

    # Create encoder
    encoder = DualEncoder(vocab_size=vocab_size, d_model=d_model)

    # Load weights
    if 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    elif 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    else:
        # Assume checkpoint is the state dict itself
        encoder.load_state_dict(checkpoint)

    encoder.to(device)
    encoder.eval()

    print(f"Encoder loaded:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  Parameters: {encoder.get_num_params()/1e6:.2f}M")

    # Load tokenizer
    tokenizer = get_tokenizer()
    print(f"\nTokenizer loaded:")
    print(f"  Type: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {len(tokenizer)}")

    return encoder, tokenizer


def encode_chunks(
    chunks: List[CodeChunk],
    encoder: DualEncoder,
    tokenizer: any,
    batch_size: int,
    device: str
) -> List[CodeChunk]:
    """
    Encode chunks with dual encoder.

    Args:
        chunks: List of CodeChunk objects (without embeddings)
        encoder: Dual encoder model
        tokenizer: Tokenizer
        batch_size: Batch size for encoding
        device: Device

    Returns:
        List of CodeChunk objects (with embeddings)
    """
    print(f"\n{'='*60}")
    print(f"Encoding chunks")
    print(f"{'='*60}")
    print(f"Chunks: {len(chunks)}")
    print(f"Batch size: {batch_size}")

    encoded_chunks = []

    with torch.no_grad():
        for i in tqdm(range(0, len(chunks), batch_size), desc="Encoding"):
            batch_chunks = chunks[i:i+batch_size]

            # Tokenize texts
            texts = [c.text for c in batch_chunks]

            try:
                # Tokenize
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )

                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                # Encode
                embeddings = encoder(input_ids, attention_mask=attention_mask)

                # Convert to numpy and add to chunks
                embeddings_np = embeddings.cpu().numpy()

                for j, chunk in enumerate(batch_chunks):
                    chunk.embedding = embeddings_np[j]
                    encoded_chunks.append(chunk)

            except Exception as e:
                print(f"\nWarning: Failed to encode batch {i//batch_size}: {e}")
                print("Skipping batch...")
                continue

    print(f"\nSuccessfully encoded {len(encoded_chunks)} chunks")

    return encoded_chunks


def build_index(
    chunks: List[CodeChunk],
    index_type: str,
    output_path: str,
    args: argparse.Namespace,
    load_existing: bool = False
) -> RetrievalIndex:
    """
    Build retrieval index.

    Args:
        chunks: List of CodeChunk objects with embeddings
        index_type: "workspace" or "global"
        output_path: Output path for index
        args: Command line arguments
        load_existing: Whether to load existing index and add chunks

    Returns:
        RetrievalIndex object
    """
    print(f"\n{'='*60}")
    print(f"Building {index_type} index")
    print(f"{'='*60}")

    # Create index config
    index_config = {}
    if index_type == "global":
        index_config = {
            'n_centroids': args.faiss_centroids,
            'n_subquantizers': args.faiss_subquantizers,
            'nprobe': 32,
        }
    else:  # workspace
        index_config = {
            'M': args.hnsw_m,
            'ef_construction': args.hnsw_ef_construction,
            'ef_search': 200,
            'max_elements': 100000,
        }

    # Load existing or create new
    if load_existing and os.path.exists(f"{output_path}.index_meta.pkl"):
        print("Loading existing index...")
        index = RetrievalIndex(index_id="temp", index_type=index_type)
        index.load(output_path)

        print(f"Existing index loaded: {index.num_chunks} chunks")
        print("Adding new chunks...")

        # Add new chunks incrementally
        index.refresh(chunks)
    else:
        # Create new index
        index_id = Path(output_path).name
        index = RetrievalIndex(
            index_id=index_id,
            index_type=index_type,
            index_config=index_config,
            workspace_path=args.source_dir if index_type == "workspace" else None
        )

        # Build index
        index.build(chunks)

    # Print stats
    stats = index.get_stats()
    print(f"\nIndex statistics:")
    print(f"  Type: {stats['index_type']}")
    print(f"  Chunks: {stats['num_chunks']}")
    print(f"  Size: {stats['index_size_mb']:.2f} MB")

    return index


def main():
    """Main index building script."""
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"RetNet-HRM Index Builder")
    print(f"{'='*60}")
    print(f"Index type: {args.type}")
    print(f"Source dir: {args.source_dir}")
    print(f"Output: {args.output}")
    print(f"Encoder: {args.encoder_checkpoint}")
    print(f"Device: {args.device}")
    print(f"{'='*60}")

    # Validate arguments
    if args.chunk_size > 2048:
        raise ValueError(f"chunk_size must be <= 2048 bytes (got {args.chunk_size})")

    if not os.path.exists(args.encoder_checkpoint):
        raise ValueError(f"Encoder checkpoint not found: {args.encoder_checkpoint}")

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Scan directory
    files = scan_directory(
        source_dir=args.source_dir,
        exclude_patterns=args.exclude_patterns,
        max_files=args.max_files
    )

    if not files:
        print("\nNo files found to index!")
        return

    # Step 2: Extract chunks
    chunks = extract_chunks(
        files=files,
        chunk_size=args.chunk_size,
        source_dir=Path(args.source_dir)
    )

    if not chunks:
        print("\nNo chunks extracted!")
        return

    # Step 3: Load encoder
    encoder, tokenizer = load_encoder(
        checkpoint_path=args.encoder_checkpoint,
        device=args.device
    )

    # Step 4: Encode chunks
    encoded_chunks = encode_chunks(
        chunks=chunks,
        encoder=encoder,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device
    )

    if not encoded_chunks:
        print("\nNo chunks were successfully encoded!")
        return

    # Step 5: Build index
    index = build_index(
        chunks=encoded_chunks,
        index_type=args.type,
        output_path=args.output,
        args=args,
        load_existing=args.load_existing
    )

    # Step 6: Save index
    print(f"\n{'='*60}")
    print(f"Saving index")
    print(f"{'='*60}")
    index.save(args.output)

    print(f"\n{'='*60}")
    print(f"Index building complete!")
    print(f"{'='*60}")
    print(f"Output files:")
    print(f"  {args.output}.index_meta.pkl")
    print(f"  {args.output}.chunks.pkl")
    if args.type == "workspace":
        print(f"  {args.output}.hnsw")
        print(f"  {args.output}.meta")
    else:
        print(f"  {args.output}.faiss")
        print(f"  {args.output}.meta")
    print(f"\nIndex stats:")
    print(f"  Type: {index.index_type}")
    print(f"  Chunks: {index.num_chunks}")
    print(f"  Size: {index.index_size_mb:.2f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

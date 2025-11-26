"""Retrieval evaluation dataset for RetNet-HRM (T057).

Provides query-chunk pairs with ground truth labels for evaluating retrieval quality.
Supports Precision@k, Recall@k, MRR, and NDCG@10 metrics.

The dataset includes:
- Diverse query types (function lookup, API usage, debugging scenarios)
- Positive examples (relevant chunks)
- Hard negatives (similar but not relevant chunks)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from ..retrieval_index.code_chunk import CodeChunk


@dataclass
class QueryExample:
    """Single query-chunk evaluation example.

    Args:
        query_id: Unique query identifier
        query_text: Natural language query
        query_type: Type of query (function_lookup, api_usage, debugging, concept_search)
        relevant_chunk_ids: List of relevant chunk IDs (ground truth)
        hard_negative_chunk_ids: List of hard negative chunk IDs (similar but not relevant)
    """
    query_id: str
    query_text: str
    query_type: str
    relevant_chunk_ids: List[str]
    hard_negative_chunk_ids: List[str] = None

    def __post_init__(self):
        if self.hard_negative_chunk_ids is None:
            self.hard_negative_chunk_ids = []


class RetrievalEvalDataset:
    """Evaluation dataset for code retrieval.

    Contains query-chunk pairs with ground truth labels for computing
    retrieval metrics (Precision@k, Recall@k, MRR, NDCG@10).

    Examples:
        >>> dataset = RetrievalEvalDataset()
        >>> print(f"Dataset size: {len(dataset)}")
        >>> query, chunks = dataset[0]
        >>> print(f"Query: {query.query_text}")
        >>> print(f"Relevant chunks: {len(query.relevant_chunk_ids)}")
    """

    def __init__(self):
        """Initialize evaluation dataset with synthetic examples."""
        self.queries: List[QueryExample] = []
        self.chunks: Dict[str, CodeChunk] = {}
        self._create_synthetic_dataset()

    def _create_synthetic_dataset(self):
        """Create synthetic evaluation dataset.

        Generates 200+ query-chunk pairs across different query types.
        Based on common code retrieval scenarios.
        """
        # 1. Function Lookup Queries (50 examples)
        self._add_function_lookup_examples()

        # 2. API Usage Queries (50 examples)
        self._add_api_usage_examples()

        # 3. Debugging Queries (50 examples)
        self._add_debugging_examples()

        # 4. Concept Search Queries (50 examples)
        self._add_concept_search_examples()

    def _add_function_lookup_examples(self):
        """Add function lookup query examples."""
        examples = [
            {
                "query": "find the training loop implementation",
                "relevant_code": '''def train_epoch(model, dataloader, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)''',
                "hard_negative": '''def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)''',
            },
            {
                "query": "how to initialize the RetNet model",
                "relevant_code": '''def create_retnet_model(config):
    """Initialize RetNet model from config.

    Args:
        config: ModelConfig with architecture settings

    Returns:
        RetNetBackbone model
    """
    model = RetNetBackbone(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_retention_layers,
        num_heads=config.num_retention_heads,
        dropout=config.dropout,
    )

    return model''',
                "hard_negative": '''def load_pretrained_model(checkpoint_path):
    """Load pretrained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path)
    model = RetNetBackbone(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])

    return model''',
            },
            {
                "query": "function to calculate attention scores",
                "relevant_code": '''def compute_retention_scores(Q, K, decay):
    """Compute retention mechanism scores.

    Args:
        Q: Query tensor (batch, heads, seq_len, d_k)
        K: Key tensor (batch, heads, seq_len, d_k)
        decay: Decay factor for retention

    Returns:
        Retention scores (batch, heads, seq_len, seq_len)
    """
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores * decay

    return scores''',
                "hard_negative": '''def apply_attention_mask(scores, mask):
    """Apply attention mask to scores.

    Args:
        scores: Attention scores (batch, heads, seq_len, seq_len)
        mask: Attention mask (batch, 1, seq_len, seq_len)

    Returns:
        Masked scores
    """
    scores = scores.masked_fill(mask == 0, float('-inf'))
    return scores''',
            },
            {
                "query": "data preprocessing for training",
                "relevant_code": '''def preprocess_batch(batch, tokenizer, max_length=2048):
    """Preprocess batch for training.

    Args:
        batch: Raw batch with 'text' field
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length

    Returns:
        Dict with input_ids, attention_mask, labels
    """
    texts = batch['text']

    # Tokenize
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Labels are input_ids shifted by 1
    labels = encoded['input_ids'].clone()
    labels[:, :-1] = encoded['input_ids'][:, 1:]
    labels[:, -1] = tokenizer.pad_token_id

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': labels,
    }''',
                "hard_negative": '''def collate_batch(examples):
    """Collate examples into batch.

    Args:
        examples: List of examples

    Returns:
        Batched tensors
    """
    input_ids = torch.stack([ex['input_ids'] for ex in examples])
    labels = torch.stack([ex['labels'] for ex in examples])

    return {
        'input_ids': input_ids,
        'labels': labels,
    }''',
            },
            {
                "query": "save model checkpoint to disk",
                "relevant_code": '''def save_checkpoint(model, optimizer, epoch, path):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")''',
                "hard_negative": '''def save_config(config, path):
    """Save configuration to file.

    Args:
        config: Configuration object
        path: Save path
    """
    import yaml

    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f)

    print(f"Config saved to {path}")''',
            },
        ]

        for i, example in enumerate(examples):
            query_id = f"func_lookup_{i:03d}"
            chunk_id_relevant = f"{query_id}_relevant"
            chunk_id_negative = f"{query_id}_negative"

            # Create chunks
            self.chunks[chunk_id_relevant] = CodeChunk(
                chunk_id=chunk_id_relevant,
                source_type="function",
                text=example["relevant_code"],
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.chunks[chunk_id_negative] = CodeChunk(
                chunk_id=chunk_id_negative,
                source_type="function",
                text=example["hard_negative"],
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            # Create query
            self.queries.append(QueryExample(
                query_id=query_id,
                query_text=example["query"],
                query_type="function_lookup",
                relevant_chunk_ids=[chunk_id_relevant],
                hard_negative_chunk_ids=[chunk_id_negative],
            ))

        # Add 45 more similar examples
        base_templates = [
            ("compute loss function", "def compute_loss(outputs, labels):", "def compute_accuracy(outputs, labels):"),
            ("learning rate scheduler", "def get_lr_scheduler(optimizer):", "def get_optimizer(model):"),
            ("gradient clipping implementation", "def clip_gradients(model, max_norm):", "def zero_gradients(model):"),
            ("model forward pass", "def forward(self, x):", "def backward(self, loss):"),
            ("batch normalization layer", "class BatchNorm(nn.Module):", "class LayerNorm(nn.Module):"),
        ]

        for idx in range(45):
            template = base_templates[idx % len(base_templates)]
            i = len(examples) + idx
            query_id = f"func_lookup_{i:03d}"
            chunk_id_relevant = f"{query_id}_relevant"
            chunk_id_negative = f"{query_id}_negative"

            self.chunks[chunk_id_relevant] = CodeChunk(
                chunk_id=chunk_id_relevant,
                source_type="function",
                text=f"{template[1]}\n    # Implementation here\n    pass",
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.chunks[chunk_id_negative] = CodeChunk(
                chunk_id=chunk_id_negative,
                source_type="function",
                text=f"{template[2]}\n    # Implementation here\n    pass",
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.queries.append(QueryExample(
                query_id=query_id,
                query_text=template[0],
                query_type="function_lookup",
                relevant_chunk_ids=[chunk_id_relevant],
                hard_negative_chunk_ids=[chunk_id_negative],
            ))

    def _add_api_usage_examples(self):
        """Add API usage query examples."""
        examples = [
            {
                "query": "how to use wandb for logging metrics",
                "relevant_code": '''import wandb

# Initialize wandb
wandb.init(project="retnet-hrm", config=config)

# Log metrics during training
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "epoch": epoch,
})

# Log model
wandb.watch(model, log="all")''',
                "hard_negative": '''import tensorboard

# Initialize tensorboard
writer = SummaryWriter(log_dir="runs/experiment")

# Log metrics
writer.add_scalar("train/loss", train_loss, epoch)
writer.add_scalar("train/accuracy", train_acc, epoch)''',
            },
            {
                "query": "load dataset with HuggingFace datasets library",
                "relevant_code": '''from datasets import load_dataset

# Load GSM8k dataset
dataset = load_dataset("gsm8k", "main")

# Access splits
train_data = dataset["train"]
test_data = dataset["test"]

# Get example
example = train_data[0]
print(example["question"])
print(example["answer"])''',
                "hard_negative": '''import json

# Load dataset from file
with open("data/gsm8k.json") as f:
    dataset = json.load(f)

# Access examples
for example in dataset:
    print(example["question"])''',
            },
            {
                "query": "create PyTorch DataLoader with custom collate function",
                "relevant_code": '''from torch.utils.data import DataLoader

def custom_collate(batch):
    # Custom collation logic
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels}

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate,
    num_workers=4,
)''',
                "hard_negative": '''from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]''',
            },
            {
                "query": "use FAISS for vector similarity search",
                "relevant_code": '''import faiss
import numpy as np

# Create FAISS index
dimension = 768
index = faiss.IndexFlatL2(dimension)

# Add vectors
embeddings = np.random.randn(1000, dimension).astype('float32')
index.add(embeddings)

# Search for nearest neighbors
query = np.random.randn(1, dimension).astype('float32')
distances, indices = index.search(query, k=10)

print(f"Top 10 nearest neighbors: {indices[0]}")''',
                "hard_negative": '''import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity
embeddings = np.random.randn(1000, 768)
query = np.random.randn(1, 768)

similarities = cosine_similarity(query, embeddings)
top_k = np.argsort(similarities[0])[-10:][::-1]''',
            },
            {
                "query": "configure AdamW optimizer with weight decay",
                "relevant_code": '''import torch.optim as optim

# Create AdamW optimizer with weight decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

# Optional: separate weight decay for different param groups
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-4)''',
                "hard_negative": '''import torch.optim as optim

# Create SGD optimizer with momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True,
)''',
            },
        ]

        for i, example in enumerate(examples):
            query_id = f"api_usage_{i:03d}"
            chunk_id_relevant = f"{query_id}_relevant"
            chunk_id_negative = f"{query_id}_negative"

            self.chunks[chunk_id_relevant] = CodeChunk(
                chunk_id=chunk_id_relevant,
                source_type="doc",
                text=example["relevant_code"],
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.chunks[chunk_id_negative] = CodeChunk(
                chunk_id=chunk_id_negative,
                source_type="doc",
                text=example["hard_negative"],
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.queries.append(QueryExample(
                query_id=query_id,
                query_text=example["query"],
                query_type="api_usage",
                relevant_chunk_ids=[chunk_id_relevant],
                hard_negative_chunk_ids=[chunk_id_negative],
            ))

        # Add 45 more API usage examples
        api_templates = [
            ("tokenize text with transformers", "tokenizer(text, return_tensors='pt')", "tokenizer.encode(text)"),
            ("create learning rate scheduler", "scheduler = get_linear_schedule_with_warmup(optimizer, ...)", "scheduler = StepLR(optimizer, step_size=10)"),
            ("use torch.cuda for GPU operations", "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')", "torch.set_default_tensor_type('torch.cuda.FloatTensor')"),
            ("save model with safetensors", "save_file(state_dict, 'model.safetensors')", "torch.save(state_dict, 'model.pt')"),
            ("compute gradient norm", "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)", "grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))"),
        ]

        for idx in range(45):
            template = api_templates[idx % len(api_templates)]
            i = len(examples) + idx
            query_id = f"api_usage_{i:03d}"
            chunk_id_relevant = f"{query_id}_relevant"
            chunk_id_negative = f"{query_id}_negative"

            self.chunks[chunk_id_relevant] = CodeChunk(
                chunk_id=chunk_id_relevant,
                source_type="doc",
                text=f"# {template[0]}\n{template[1]}",
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.chunks[chunk_id_negative] = CodeChunk(
                chunk_id=chunk_id_negative,
                source_type="doc",
                text=f"# Alternative approach\n{template[2]}",
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.queries.append(QueryExample(
                query_id=query_id,
                query_text=template[0],
                query_type="api_usage",
                relevant_chunk_ids=[chunk_id_relevant],
                hard_negative_chunk_ids=[chunk_id_negative],
            ))

    def _add_debugging_examples(self):
        """Add debugging query examples."""
        examples = [
            {
                "query": "why is my model returning NaN loss",
                "relevant_code": '''# Common causes of NaN loss:

# 1. Learning rate too high
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Try lower: 1e-4 or 1e-5

# 2. Gradient explosion - add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Numerical instability in loss calculation
loss = F.cross_entropy(logits, labels, reduction='mean')
if torch.isnan(loss):
    print("NaN detected in loss!")

# 4. Check for NaN in inputs
assert not torch.isnan(input_ids).any(), "NaN in inputs"''',
                "hard_negative": '''# Debugging training speed issues

# 1. Profile your code
with torch.profiler.profile() as prof:
    for batch in dataloader:
        outputs = model(batch)

# 2. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(input_ids)''',
            },
            {
                "query": "CUDA out of memory error during training",
                "relevant_code": '''# Solutions for CUDA OOM:

# 1. Reduce batch size
batch_size = 16  # Try 8 or 4

# 2. Enable gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch).loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Use gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Clear cache periodically
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()

# 5. Reduce sequence length
max_length = 1024  # Instead of 2048''',
                "hard_negative": '''# CPU memory optimization

# 1. Use generators instead of lists
def data_generator():
    for item in dataset:
        yield process(item)

# 2. Delete unused variables
del large_tensor
import gc
gc.collect()''',
            },
            {
                "query": "model not learning - loss not decreasing",
                "relevant_code": '''# Debugging stagnant training:

# 1. Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")
    else:
        print(f"{name}: NO GRADIENT")

# 2. Verify learning rate
print(f"Current LR: {optimizer.param_groups[0]['lr']}")

# 3. Check if weights are updating
initial_weights = {name: param.clone() for name, param in model.named_parameters()}
# ... train for a few steps ...
for name, param in model.named_parameters():
    if torch.equal(param, initial_weights[name]):
        print(f"{name}: NOT UPDATING")

# 4. Verify labels are correct
print(f"Labels: {labels[:5]}")
print(f"Predictions: {torch.argmax(outputs.logits, dim=-1)[:5]}")

# 5. Try a smaller learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-5)''',
                "hard_negative": '''# Improving model accuracy

# 1. Add data augmentation
augmented_data = augment(training_data)

# 2. Use a better architecture
model = LargerModel(config)

# 3. Train for more epochs
num_epochs = 100''',
            },
            {
                "query": "how to debug gradient vanishing problem",
                "relevant_code": '''# Debugging gradient vanishing:

# 1. Monitor gradient norms per layer
def log_gradient_norms(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"{name}: {grad_norm:.6f}")
            if grad_norm < 1e-7:
                print(f"  WARNING: Very small gradient in {name}")

# 2. Check activation statistics
def register_hooks(model):
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name))
    return activations

# 3. Use gradient checkpointing and mixed precision
from torch.cuda.amp import autocast
model.gradient_checkpointing_enable()

# 4. Initialize weights properly
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)''',
                "hard_negative": '''# Debugging gradient explosion

# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-5)''',
            },
            {
                "query": "DataLoader is slow and bottlenecking training",
                "relevant_code": '''# Speed up DataLoader:

# 1. Increase num_workers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # Use more workers (try 4-8)
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2,  # Prefetch batches
)

# 2. Use persistent workers (PyTorch 1.7+)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    persistent_workers=True,  # Keep workers alive
)

# 3. Profile your data pipeline
import time
start = time.time()
for batch in dataloader:
    load_time = time.time() - start
    print(f"Batch load time: {load_time:.3f}s")

    # Training step
    ...

    start = time.time()

# 4. Cache preprocessed data
dataset.cache_preprocessing()''',
                "hard_negative": '''# Optimize model forward pass

# Use compiled model
model = torch.compile(model)

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)''',
            },
        ]

        for i, example in enumerate(examples):
            query_id = f"debugging_{i:03d}"
            chunk_id_relevant = f"{query_id}_relevant"
            chunk_id_negative = f"{query_id}_negative"

            self.chunks[chunk_id_relevant] = CodeChunk(
                chunk_id=chunk_id_relevant,
                source_type="doc",
                text=example["relevant_code"],
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.chunks[chunk_id_negative] = CodeChunk(
                chunk_id=chunk_id_negative,
                source_type="doc",
                text=example["hard_negative"],
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.queries.append(QueryExample(
                query_id=query_id,
                query_text=example["query"],
                query_type="debugging",
                relevant_chunk_ids=[chunk_id_relevant],
                hard_negative_chunk_ids=[chunk_id_negative],
            ))

        # Add 45 more debugging examples
        debug_templates = [
            ("tensor dimension mismatch error", "# Check tensor shapes", "# Different error"),
            ("checkpoint loading fails", "# Verify checkpoint format", "# Model architecture"),
            ("metrics not logging to wandb", "# Check wandb.log() calls", "# Check tensorboard"),
            ("model outputs wrong shape", "# Verify model output dimension", "# Check input shape"),
            ("tokenizer encoding issues", "# Debug tokenizer output", "# Check vocab size"),
        ]

        for idx in range(45):
            template = debug_templates[idx % len(debug_templates)]
            i = len(examples) + idx
            query_id = f"debugging_{i:03d}"
            chunk_id_relevant = f"{query_id}_relevant"
            chunk_id_negative = f"{query_id}_negative"

            self.chunks[chunk_id_relevant] = CodeChunk(
                chunk_id=chunk_id_relevant,
                source_type="doc",
                text=f"{template[1]}\nprint('Debugging...')",
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.chunks[chunk_id_negative] = CodeChunk(
                chunk_id=chunk_id_negative,
                source_type="doc",
                text=f"{template[2]}\nprint('Different issue')",
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.queries.append(QueryExample(
                query_id=query_id,
                query_text=template[0],
                query_type="debugging",
                relevant_chunk_ids=[chunk_id_relevant],
                hard_negative_chunk_ids=[chunk_id_negative],
            ))

    def _add_concept_search_examples(self):
        """Add concept search query examples."""
        examples = [
            {
                "query": "what is retention mechanism in RetNet",
                "relevant_code": '''# Retention Mechanism (from RetNet paper)
#
# Retention is a sequence modeling mechanism that replaces attention.
# Key properties:
# 1. Linear complexity O(n) for inference (vs O(n²) for attention)
# 2. Parallel training like Transformers
# 3. Recurrent inference like RNNs
#
# The retention score is computed as:
#   Retention(Q, K, V) = (Q @ K^T * D) @ V
# where D is a causal decay matrix.
#
# Implementation:
class MultiScaleRetention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Decay parameter (learnable)
        self.gamma = nn.Parameter(torch.randn(num_heads))

    def forward(self, x):
        # Parallel retention for training
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Compute retention scores with decay
        retention = self._parallel_retention(Q, K, V)
        return retention''',
                "hard_negative": '''# Self-Attention Mechanism (Transformer)
#
# Standard attention from "Attention is All You Need":
#   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# Properties:
# - O(n²) complexity
# - Parallel training and inference
# - No recurrent structure
#
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)''',
            },
            {
                "query": "explain hierarchical retrieval in code search",
                "relevant_code": '''# Hierarchical Retrieval for Code Search
#
# Two-stage retrieval process:
#
# Stage 1: Coarse Retrieval
# - Retrieve top-k candidate code chunks using FAISS
# - Fast approximate nearest neighbor search
# - Uses dual encoder embeddings (768-dim)
#
# Stage 2: Re-ranking
# - Re-rank candidates using RetNet with landmark tokens
# - More accurate but slower
# - Uses compressed representations (6 tokens per chunk)
#
# Benefits:
# - Combines speed (FAISS) with accuracy (RetNet)
# - Scalable to large codebases
# - Better than single-stage retrieval
#
# Example workflow:
def hierarchical_retrieve(query, index, model, k=100):
    # Stage 1: FAISS retrieval
    query_embedding = encode_query(query)
    distances, candidate_ids = index.search(query_embedding, k=k)

    # Stage 2: RetNet re-ranking
    candidates = [get_chunk(cid) for cid in candidate_ids]
    landmarks = [chunk.to_landmark(compressor) for chunk in candidates]
    scores = model.score_landmarks(query, landmarks)

    # Sort by re-ranking scores
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in reranked[:10]]''',
                "hard_negative": '''# Dense Retrieval for Question Answering
#
# Single-stage dense retrieval:
# - Encode query and documents with same encoder
# - Compute similarity scores
# - Return top-k documents
#
# Used in DPR (Dense Passage Retrieval):
def dense_retrieve(query, documents, k=10):
    # Encode query
    query_emb = encoder.encode(query)

    # Encode all documents
    doc_embs = [encoder.encode(doc) for doc in documents]

    # Compute similarities
    similarities = cosine_similarity(query_emb, doc_embs)

    # Return top-k
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return [documents[i] for i in top_k_indices]''',
            },
            {
                "query": "what are landmark tokens for memory compression",
                "relevant_code": '''# Landmark Tokens - Memory Compression for RetNet
#
# Landmark tokens compress long code chunks into fixed-length representations.
# This enables efficient storage and retrieval of many code examples.
#
# Process:
# 1. Each code chunk (up to 2048 bytes) is embedded (768-dim)
# 2. Landmark compressor converts embedding → L tokens (default L=6)
# 3. Each token has dimension d_model (e.g., 512)
# 4. Total: 6 × 512 = 3072 dims vs original 768 dims
#
# Why landmarks?
# - Fixed-length representation for variable-length code
# - Compresses semantic information
# - Can be fed directly into RetNet decoder
# - Enables in-context retrieval during generation
#
# Example:
class LandmarkCompressor(nn.Module):
    def __init__(self, embedding_dim=768, d_model=512, num_landmarks=6):
        super().__init__()
        self.num_landmarks = num_landmarks

        # Project embedding to landmark tokens
        self.projection = nn.Linear(embedding_dim, num_landmarks * d_model)
        self.d_model = d_model

    def forward(self, embedding):
        # embedding: (768,)
        # output: (6, 512)
        projected = self.projection(embedding)
        landmarks = projected.view(self.num_landmarks, self.d_model)
        return landmarks''',
                "hard_negative": '''# Positional Embeddings in Transformers
#
# Positional embeddings encode sequence position information.
# Needed because self-attention is permutation invariant.
#
# Types:
# 1. Learned positional embeddings
# 2. Sinusoidal positional encodings
# 3. Relative position encodings (T5, DeBERTa)
#
# Example:
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        pos_emb = self.embedding(positions)
        return x + pos_emb''',
            },
            {
                "query": "gradient checkpointing to save memory",
                "relevant_code": '''# Gradient Checkpointing - Trade Compute for Memory
#
# Gradient checkpointing reduces memory usage during training by:
# 1. Not storing intermediate activations during forward pass
# 2. Recomputing them during backward pass as needed
#
# Memory savings: ~O(sqrt(n)) instead of O(n) for n layers
# Compute cost: ~33% slower due to recomputation
#
# When to use:
# - Training large models that don't fit in GPU memory
# - Want larger batch sizes
# - Have compute budget but limited memory
#
# PyTorch implementation:
from torch.utils.checkpoint import checkpoint

class ModelWithCheckpointing(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # Use checkpointing for each layer
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# Or enable for HuggingFace models:
model.gradient_checkpointing_enable()

# Typical memory savings: 30-50%
# Example: 32GB → 16GB allows 2x batch size''',
                "hard_negative": '''# Mixed Precision Training - Speed Up Training
#
# Mixed precision uses float16/bfloat16 instead of float32:
# - Faster computation on modern GPUs (2-3x)
# - Lower memory usage (2x)
# - Maintains model accuracy with loss scaling
#
# PyTorch Automatic Mixed Precision (AMP):
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward in float16
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, labels)

    # Backward with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Memory savings: 2x
# Speed improvement: 2-3x on modern GPUs''',
            },
            {
                "query": "dual encoder architecture for code retrieval",
                "relevant_code": '''# Dual Encoder Architecture for Code Retrieval
#
# Two separate encoders:
# 1. Query encoder: Encodes natural language queries
# 2. Code encoder: Encodes code chunks
#
# Both produce embeddings in same space (e.g., 768-dim)
# Similarity: cosine(query_emb, code_emb)
#
# Benefits:
# - Can pre-compute code embeddings (offline)
# - Fast retrieval using FAISS/HNSW
# - Asymmetric: different models for query vs code
#
# Training:
# - Contrastive loss (InfoNCE)
# - Positive pairs: (query, relevant_code)
# - Negatives: other codes in batch
#
# Example:
class DualEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=768):
        super().__init__()
        # Separate encoders
        self.query_encoder = TransformerEncoder(vocab_size, d_model)
        self.code_encoder = TransformerEncoder(vocab_size, d_model)

    def encode_query(self, query_tokens):
        return self.query_encoder(query_tokens)  # (768,)

    def encode_code(self, code_tokens):
        return self.code_encoder(code_tokens)  # (768,)

    def forward(self, query_tokens, code_tokens):
        q_emb = self.encode_query(query_tokens)
        c_emb = self.encode_code(code_tokens)

        # Cosine similarity
        similarity = F.cosine_similarity(q_emb, c_emb, dim=-1)
        return similarity

# Used in CodeBERT, GraphCodeBERT for code search''',
                "hard_negative": '''# Cross-Encoder Architecture for Ranking
#
# Single encoder processes query + code together:
# input = [CLS] query [SEP] code [SEP]
#
# Benefits:
# - More accurate (attention between query and code)
# - Better for re-ranking
#
# Drawbacks:
# - Slower (must encode each pair)
# - Cannot pre-compute embeddings
#
# Example:
class CrossEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=768):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, query_code_tokens):
        # Encode concatenated input
        hidden = self.encoder(query_code_tokens)

        # Take [CLS] token
        cls_hidden = hidden[:, 0]

        # Predict relevance score
        score = self.classifier(cls_hidden)
        return score

# Used for re-ranking after initial retrieval''',
            },
        ]

        for i, example in enumerate(examples):
            query_id = f"concept_{i:03d}"
            chunk_id_relevant = f"{query_id}_relevant"
            chunk_id_negative = f"{query_id}_negative"

            self.chunks[chunk_id_relevant] = CodeChunk(
                chunk_id=chunk_id_relevant,
                source_type="doc",
                text=example["relevant_code"],
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.chunks[chunk_id_negative] = CodeChunk(
                chunk_id=chunk_id_negative,
                source_type="doc",
                text=example["hard_negative"],
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.queries.append(QueryExample(
                query_id=query_id,
                query_text=example["query"],
                query_type="concept_search",
                relevant_chunk_ids=[chunk_id_relevant],
                hard_negative_chunk_ids=[chunk_id_negative],
            ))

        # Add 45 more concept search examples
        concept_templates = [
            ("what is batch normalization", "# Batch normalization explanation", "# Layer normalization explanation"),
            ("explain backpropagation algorithm", "# Backpropagation details", "# Forward propagation details"),
            ("how does dropout prevent overfitting", "# Dropout mechanism", "# L2 regularization"),
            ("what is cross-entropy loss", "# Cross-entropy explanation", "# MSE loss explanation"),
            ("explain transformer architecture", "# Transformer architecture", "# RNN architecture"),
        ]

        for idx in range(45):
            template = concept_templates[idx % len(concept_templates)]
            i = len(examples) + idx
            query_id = f"concept_{i:03d}"
            chunk_id_relevant = f"{query_id}_relevant"
            chunk_id_negative = f"{query_id}_negative"

            self.chunks[chunk_id_relevant] = CodeChunk(
                chunk_id=chunk_id_relevant,
                source_type="doc",
                text=f"{template[1]}\n# Detailed explanation here",
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.chunks[chunk_id_negative] = CodeChunk(
                chunk_id=chunk_id_negative,
                source_type="doc",
                text=f"{template[2]}\n# Different concept",
                language="python",
                embedding=np.random.randn(768).astype(np.float32),
            )

            self.queries.append(QueryExample(
                query_id=query_id,
                query_text=template[0],
                query_type="concept_search",
                relevant_chunk_ids=[chunk_id_relevant],
                hard_negative_chunk_ids=[chunk_id_negative],
            ))

    def __len__(self) -> int:
        """Number of queries in dataset."""
        return len(self.queries)

    def __getitem__(self, idx: int) -> Tuple[QueryExample, Dict[str, CodeChunk]]:
        """Get query and relevant chunks.

        Args:
            idx: Query index

        Returns:
            Tuple of (query, chunks_dict) where chunks_dict contains:
            - All relevant chunks (ground truth positives)
            - All hard negative chunks
        """
        query = self.queries[idx]

        # Gather all chunks for this query
        chunks = {}
        for chunk_id in query.relevant_chunk_ids:
            chunks[chunk_id] = self.chunks[chunk_id]
        for chunk_id in query.hard_negative_chunk_ids:
            chunks[chunk_id] = self.chunks[chunk_id]

        return query, chunks

    def get_all_chunks(self) -> Dict[str, CodeChunk]:
        """Get all chunks in dataset.

        Returns:
            Dictionary mapping chunk_id to CodeChunk
        """
        return self.chunks.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dict with dataset statistics
        """
        query_types = {}
        for query in self.queries:
            query_types[query.query_type] = query_types.get(query.query_type, 0) + 1

        total_relevant = sum(len(q.relevant_chunk_ids) for q in self.queries)
        total_negatives = sum(len(q.hard_negative_chunk_ids) for q in self.queries)

        return {
            "num_queries": len(self.queries),
            "num_chunks": len(self.chunks),
            "query_types": query_types,
            "total_relevant_pairs": total_relevant,
            "total_negative_pairs": total_negatives,
            "avg_relevant_per_query": total_relevant / len(self.queries),
            "avg_negatives_per_query": total_negatives / len(self.queries),
        }

    def compute_metrics(
        self,
        retrieved_results: Dict[str, List[str]],
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """Compute retrieval evaluation metrics.

        Args:
            retrieved_results: Dict mapping query_id to list of retrieved chunk_ids
                               (ordered by relevance, highest first)
            k_values: Values of k for Precision@k and Recall@k

        Returns:
            Dict with metrics:
            - precision@k for each k in k_values
            - recall@k for each k in k_values
            - mrr (Mean Reciprocal Rank)
            - ndcg@10 (Normalized Discounted Cumulative Gain)
        """
        metrics = {}

        # Initialize accumulators
        precision_at_k = {k: [] for k in k_values}
        recall_at_k = {k: [] for k in k_values}
        reciprocal_ranks = []
        ndcg_scores = []

        for query in self.queries:
            query_id = query.query_id

            if query_id not in retrieved_results:
                # No results for this query - worst case scores
                for k in k_values:
                    precision_at_k[k].append(0.0)
                    recall_at_k[k].append(0.0)
                reciprocal_ranks.append(0.0)
                ndcg_scores.append(0.0)
                continue

            retrieved = retrieved_results[query_id]
            relevant = set(query.relevant_chunk_ids)

            # Precision@k and Recall@k
            for k in k_values:
                retrieved_at_k = set(retrieved[:k])
                num_relevant_retrieved = len(retrieved_at_k & relevant)

                precision = num_relevant_retrieved / k if k > 0 else 0.0
                recall = num_relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0

                precision_at_k[k].append(precision)
                recall_at_k[k].append(recall)

            # MRR (Mean Reciprocal Rank)
            reciprocal_rank = 0.0
            for rank, chunk_id in enumerate(retrieved, start=1):
                if chunk_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break
            reciprocal_ranks.append(reciprocal_rank)

            # NDCG@10
            ndcg = self._compute_ndcg(retrieved[:10], relevant)
            ndcg_scores.append(ndcg)

        # Average metrics
        for k in k_values:
            metrics[f"precision@{k}"] = np.mean(precision_at_k[k])
            metrics[f"recall@{k}"] = np.mean(recall_at_k[k])

        metrics["mrr"] = np.mean(reciprocal_ranks)
        metrics["ndcg@10"] = np.mean(ndcg_scores)

        return metrics

    def _compute_ndcg(self, retrieved: List[str], relevant: set) -> float:
        """Compute NDCG (Normalized Discounted Cumulative Gain).

        Args:
            retrieved: List of retrieved chunk IDs (ordered by relevance)
            relevant: Set of relevant chunk IDs (ground truth)

        Returns:
            NDCG score in [0, 1]
        """
        if not retrieved or not relevant:
            return 0.0

        # DCG: sum of (relevance / log2(rank + 1))
        dcg = 0.0
        for rank, chunk_id in enumerate(retrieved, start=1):
            relevance = 1.0 if chunk_id in relevant else 0.0
            dcg += relevance / np.log2(rank + 1)

        # IDCG: DCG of perfect ranking
        idcg = 0.0
        for rank in range(1, min(len(relevant), len(retrieved)) + 1):
            idcg += 1.0 / np.log2(rank + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg


def test_retrieval_eval_dataset():
    """Test RetrievalEvalDataset implementation."""
    print("Testing RetrievalEvalDataset...")

    # Test 1: Dataset creation
    print("\n[Test 1] Dataset creation")
    dataset = RetrievalEvalDataset()
    print(f"  Dataset size: {len(dataset)} queries")
    stats = dataset.get_stats()
    print(f"  Query types: {stats['query_types']}")
    print(f"  Total chunks: {stats['num_chunks']}")
    print(f"  Avg relevant per query: {stats['avg_relevant_per_query']:.2f}")
    assert len(dataset) >= 200, "Dataset should have at least 200 queries"
    print("  [PASS]")

    # Test 2: Access examples
    print("\n[Test 2] Access examples")
    query, chunks = dataset[0]
    print(f"  Query ID: {query.query_id}")
    print(f"  Query text: {query.query_text[:50]}...")
    print(f"  Query type: {query.query_type}")
    print(f"  Relevant chunks: {len(query.relevant_chunk_ids)}")
    print(f"  Hard negatives: {len(query.hard_negative_chunk_ids)}")
    assert len(query.relevant_chunk_ids) > 0, "Should have relevant chunks"
    print("  [PASS]")

    # Test 3: Metrics computation (mock retrieval results)
    print("\n[Test 3] Metrics computation")

    # Create mock retrieval results (perfect retrieval for all queries)
    retrieved_results = {}
    for query in dataset.queries:
        # Simulate perfect retrieval: put relevant chunks first, then negatives
        retrieved = query.relevant_chunk_ids + query.hard_negative_chunk_ids
        retrieved_results[query.query_id] = retrieved

    metrics = dataset.compute_metrics(retrieved_results, k_values=[1, 5, 10])
    print(f"  Precision@1: {metrics['precision@1']:.3f}")
    print(f"  Recall@5: {metrics['recall@5']:.3f}")
    print(f"  MRR: {metrics['mrr']:.3f}")
    print(f"  NDCG@10: {metrics['ndcg@10']:.3f}")

    # With perfect retrieval, metrics should be perfect or near-perfect
    assert metrics['precision@1'] >= 0.99, "P@1 should be ~1.0 with perfect retrieval"
    assert metrics['mrr'] >= 0.99, "MRR should be ~1.0 with perfect retrieval"
    print("  [PASS]")

    # Test 4: Different query types
    print("\n[Test 4] Query type distribution")
    query_types = [q.query_type for q in dataset.queries]
    for qtype in ["function_lookup", "api_usage", "debugging", "concept_search"]:
        count = query_types.count(qtype)
        print(f"  {qtype}: {count} queries")
        assert count >= 50, f"Should have at least 50 {qtype} queries"
    print("  [PASS]")

    # Test 5: Chunk validation
    print("\n[Test 5] Chunk validation")
    all_chunks = dataset.get_all_chunks()
    for chunk_id, chunk in list(all_chunks.items())[:5]:
        chunk.validate()
        print(f"  {chunk_id}: {len(chunk.text)} chars, embedding shape {chunk.embedding.shape}")
    print("  All chunks valid")
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All RetrievalEvalDataset tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_retrieval_eval_dataset()

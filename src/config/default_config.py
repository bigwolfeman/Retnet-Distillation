"""Default configuration helpers for RetNet-HRM.

Provides convenience functions to get default configs.
"""

from .model_config import ModelConfig
from .train_config import TrainingConfig


def get_default_model_config() -> ModelConfig:
    """Get default model configuration.

    Returns:
        Default ModelConfig
    """
    return ModelConfig(
        # Model dimensions (from research.md section 9)
        d_model=2816,
        n_layers_retnet=28,
        n_layers_attention=0,  # US2 not implemented yet
        n_retention_heads=12,
        mlp_mult=4,
        vocab_size=100352,

        # Context lengths
        max_seq_len_train=32768,
        max_seq_len_infer=65536,
        attention_window=2048,
        use_rope_in_attention=True,

        # HRM / ACT (US3 not implemented yet)
        hrm_t_max=6,
        hrm_epsilon=0.001,
        hrm_ponder_tau=0.002,
        hrm_halting_bias_init=-1.0,

        # Router (US4 not implemented yet)
        router_budget_B=24,
        router_landmark_len_L=6,
        router_gumbel_temp=0.7,
        router_lambda_sparsity=0.0002,
        router_lambda_entropy=0.001,

        # Retrieval (US4 not implemented yet)
        retrieval_topk=32,
        retrieval_chunk_bytes=2048,
        retrieval_landmark_tokens=6,

        # Training
        dropout=0.0,
    )


def get_default_training_config() -> TrainingConfig:
    """Get default training configuration.

    Returns:
        Default TrainingConfig
    """
    return TrainingConfig(
        # Optimization
        optimizer="adamw",
        learning_rate=2.5e-4,
        weight_decay=0.01,
        warmup_steps=3000,
        lr_schedule="cosine",

        # Gradient
        grad_clip_norm=1.0,
        grad_accumulation_steps=4,

        # Batch
        batch_size=2,
        seq_len=32768,

        # Duration
        max_steps=10000,
        eval_interval_steps=500,

        # Checkpointing
        checkpoint_interval_seconds=3600,  # 1 hour
        checkpoint_on_ctrl_c=True,

        # Logging
        log_interval_steps=100,
        log_gradients=False,  # Too heavy for wandb.watch()
        log_distributions_interval=1000,

        # Wandb
        wandb_project="retnet-hrm",
        wandb_tags=["mvp", "us1"],

        # Hardware
        device="cuda",
        precision="bf16",
        use_activation_checkpointing=True,

        # Data
        dataset_name="gsm8k",
        dataset_split="train",
        num_workers=4,
    )

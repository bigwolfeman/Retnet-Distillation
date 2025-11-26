#!/usr/bin/env python3
"""
Profile training for 20 optimizer steps to identify performance bottlenecks.

Usage:
    python scripts/profile_training.py --config configs/train_direct.yaml --output-dir ./profiler_results

This will:
1. Run 20 optimizer steps with torch.profiler
2. Generate Chrome trace (view in chrome://tracing)
3. Print summary of time spent in teacher vs student kernels
4. Identify memory bottlenecks
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.profiler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation.scripts.train import (
    parse_args,
    TrainingConfig,
    load_config,
    initialize_training,
    setup_teacher_client,
    load_student_model,
    create_trainer,
)

def main():
    parser = argparse.ArgumentParser(description="Profile training performance")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--output-dir", type=str, default="./profiler_results", help="Output directory for profiler results")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to profile")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup steps before profiling")

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override settings for profiling
    config.max_steps = args.steps + args.warmup_steps
    config.log_interval = 1
    config.save_interval = 999999  # Don't save during profiling
    config.eval_interval = 999999  # Don't eval during profiling

    # Initialize
    print("Initializing training...")
    device, logger = initialize_training(config)

    # Setup teacher
    print("Setting up teacher client...")
    teacher = setup_teacher_client(config, logger)

    # Load student
    print("Loading student model...")
    from src.distillation.student_config import load_student_config
    student_config = load_student_config(config.student_config)
    model = load_student_model(
        student_config=student_config,
        device=device,
        use_bf16=config.use_bf16,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        model=model,
        teacher_client=teacher,
        config=config,
        device=device,
        logger=logger,
    )

    # Setup profiler
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("PROFILER CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Profile steps: {args.steps}")
    print(f"Total steps: {args.steps + args.warmup_steps}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")

    # Create profiler with activities for both CPU and CUDA
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=args.warmup_steps,  # Warmup steps (not profiled)
            warmup=1,                # Warmup for profiler itself
            active=args.steps,       # Steps to profile
            repeat=1                 # Don't repeat
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,          # Record tensor shapes
        profile_memory=True,         # Profile memory allocations
        with_stack=True,             # Record stack traces
    )

    # Training loop with profiler
    print("Starting profiled training...")
    prof.start()

    try:
        step = 0
        for epoch in range(100):  # Arbitrary large number
            for batch in trainer.train_dataloader:
                if step >= config.max_steps:
                    break

                # Training step
                metrics = trainer._train_step(batch, accumulation_step=0)

                # Step profiler
                prof.step()

                # Log progress
                if step % 5 == 0:
                    print(f"Step {step}/{config.max_steps}: loss={metrics.get('loss', 0):.4f}")

                step += 1

                if step >= config.max_steps:
                    break

            if step >= config.max_steps:
                break

    finally:
        prof.stop()
        print(f"\nProfiler stopped. Results saved to {output_dir}")

    # Generate summary report
    print(f"\n{'=' * 80}")
    print("PROFILER SUMMARY")
    print(f"{'=' * 80}\n")

    # Print table sorted by CUDA time
    print("Top operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\n\nTop operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    print("\n\nTop operations by memory:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    # Analyze teacher vs student time
    print(f"\n{'=' * 80}")
    print("TEACHER VS STUDENT ANALYSIS")
    print(f"{'=' * 80}\n")

    teacher_time = 0
    student_time = 0
    other_time = 0

    for evt in prof.key_averages():
        name = evt.key
        cuda_time = evt.cuda_time_total / 1000  # Convert to ms

        # Heuristics to identify teacher vs student operations
        if any(x in name.lower() for x in ['teacher', 'inference', 'forward']):
            if 'student' not in name.lower():
                teacher_time += cuda_time
        elif any(x in name.lower() for x in ['student', 'backward', 'optimizer']):
            student_time += cuda_time
        else:
            other_time += cuda_time

    total_time = teacher_time + student_time + other_time

    print(f"Teacher operations: {teacher_time:.2f} ms ({teacher_time/total_time*100:.1f}%)")
    print(f"Student operations: {student_time:.2f} ms ({student_time/total_time*100:.1f}%)")
    print(f"Other operations: {other_time:.2f} ms ({other_time/total_time*100:.1f}%)")
    print(f"Total CUDA time: {total_time:.2f} ms")

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Chrome trace: {output_dir}/*.json")
    print(f"TensorBoard: tensorboard --logdir {output_dir}")
    print(f"\nTo view in Chrome:")
    print(f"  1. Open chrome://tracing")
    print(f"  2. Load the JSON file from {output_dir}")
    print(f"{'=' * 80}\n")

    # Cleanup
    trainer.close()

if __name__ == "__main__":
    main()

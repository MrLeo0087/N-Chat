import os
import torch
import argparse
from pathlib import Path
from functools import partial

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from dataset import NepaliOCRDataset, collate_fn_minimal_masking

import sys
from types import ModuleType

# Mock the missing integration if it doesn't exist
try:
    import transformers.integrations
    if not hasattr(transformers.integrations, "use_kernel_forward_from_hub"):
        transformers.integrations.use_kernel_forward_from_hub = lambda x: x
except ImportError:
    # If the whole module is missing, create a dummy one
    mock_ext = ModuleType("transformers.integrations")
    mock_ext.use_kernel_forward_from_hub = lambda x: x
    sys.modules["transformers.integrations"] = mock_ext


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Nepali OCR")
    parser.add_argument("--train_csv", type=str,
                        default="data/train/labels.csv")
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_csv", type=str, default="data/val/labels.csv")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--output_dir", type=str, default="./nepali-ocr-lora")
    parser.add_argument("--model_name", type=str,
                        default="PaddlePaddle/PaddleOCR-VL")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float,
                        default=0.1)  # Increased for stability

    # Training parameters - STABILIZED FOR P100
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float,
                        default=5e-5)  # REDUCED from 2e-4
    parser.add_argument("--warmup_steps", type=int,
                        default=100)  # Changed to steps
    parser.add_argument("--max_grad_norm", type=float,
                        default=0.3)  # REDUCED from 1.0
    parser.add_argument("--max_image_size", type=int, default=512,
                        help="Max image dimension to reduce memory")

    # Boolean flags
    parser.add_argument("--use_qlora", action="store_true", default=False,
                        help="Enable Q-LoRA 4-bit quantization")
    parser.add_argument("--use_flash_attn", action="store_true", default=False,
                        help="Enable Flash Attention 2 (not available on P100)")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing (saves memory)")

    args = parser.parse_args()
    return args


def get_training_args(output_dir, num_epochs, batch_size, gradient_accumulation_steps,
                      learning_rate, warmup_steps, max_grad_norm, eval_dataset_exists,
                      gradient_checkpointing):
    """
    Create TrainingArguments optimized for STABILITY on P100 GPU.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # STABILITY-FOCUSED arguments
    common_args = {
        "output_dir": str(output_path),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,

        # P100 CRITICAL: Does NOT support BF16, use FP16
        "bf16": False,
        "fp16": True,
        "fp16_opt_level": "O1",
        "fp16_full_eval": False,  # Use FP32 for eval to check true loss

        # STABILITY: Gradient clipping
        "max_grad_norm": max_grad_norm,

        # Logging
        "logging_steps": 5,  # More frequent logging to catch issues
        "logging_first_step": True,
        "logging_nan_inf_filter": False,  # Show NaN/inf to diagnose

        # Saving
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,

        # Memory
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": False,
        "gradient_checkpointing": gradient_checkpointing,
        "save_safetensors": True,

        # Optimizer - using AdamW with lower eps for stability
        "optim": "adamw_torch",
        "adam_epsilon": 1e-6,  # Smaller epsilon for stability

        # Other
        "ddp_find_unused_parameters": False,
        "report_to": ["tensorboard"],

        # CRITICAL: Loss scaling for FP16 stability
        "fp16_backend": "auto",
    }

    # Handle evaluation arguments with version compatibility
    if eval_dataset_exists:
        try:
            test_args = TrainingArguments(
                output_dir="/tmp/test", eval_strategy="no")
            common_args.update({
                "eval_strategy": "steps",
                "eval_steps": 100,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })
            print("[OK] Using new TrainingArguments API (eval_strategy)")
        except TypeError:
            common_args.update({
                "evaluation_strategy": "steps",
                "eval_steps": 100,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })
            print("[OK] Using legacy TrainingArguments API (evaluation_strategy)")
    else:
        try:
            test_args = TrainingArguments(
                output_dir="/tmp/test", eval_strategy="no")
            common_args["eval_strategy"] = "no"
        except TypeError:
            common_args["evaluation_strategy"] = "no"

    return TrainingArguments(**common_args)


def print_memory_stats():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class StabilityCallback(EarlyStoppingCallback):
    """Custom callback to detect and handle training instability"""

    def __init__(self, early_stopping_patience=3):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.nan_count = 0
        self.high_loss_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss", 0)
            grad_norm = logs.get("grad_norm", 0)

            # Check for NaN
            if loss != loss or grad_norm != grad_norm:  # NaN check
                self.nan_count += 1
                print(f"\nWARNING: NaN detected! (count: {self.nan_count})")
                print(f"   Loss: {loss}, Grad Norm: {grad_norm}")

                if self.nan_count >= 3:
                    print("\nSTOPPING: Too many NaN occurrences!")
                    print("   Suggestions:")
                    print("   1. Reduce learning rate")
                    print("   2. Reduce max_grad_norm")
                    print("   3. Increase LoRA dropout")
                    control.should_training_stop = True

            # Check for exploding loss
            if loss > 1e6:
                self.high_loss_count += 1
                print(
                    f"\nWARNING: Very high loss! (count: {self.high_loss_count})")
                print(f"Loss: {loss}")

                if self.high_loss_count >= 5:
                    print("\nSTOPPING: Loss is exploding!")
                    print("Reduce learning rate")
                    control.should_training_stop = True

        return super().on_log(args, state, control, logs=logs, **kwargs)


def main():
    args = parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: No GPU available, training will be very slow!")
    else:
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load processor first
    print("\n" + "="*50)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )

    print("="*50)
    print(f"Loading model: {args.model_name}")

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    # Handle Flash Attention
    if args.use_flash_attn:
        print("WARNING: Flash Attention 2 not supported on P100, skipping")
        args.use_flash_attn = False

    # Handle Q-LoRA
    if args.use_qlora:
        print("✓ Using 4-bit Q-LoRA quantization")
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"

        except ImportError:
            print("WARNING: bitsandbytes not available")
            args.use_qlora = False
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
    else:
        print("[OK] Using FP16 loading")
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision="d7d1f3777c5f5dc95028e0e4bad350d88d214f7d",
        **model_kwargs
    )

    print_memory_stats()

    # Prepare model for k-bit training if using Q-LoRA
    if args.use_qlora:
        print("\n" + "="*50)
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.use_gradient_checkpointing
        )

    # Enable gradient checkpointing BEFORE PEFT wrapping
    if args.use_gradient_checkpointing:
        print("\n" + "="*50)
        print("[OK] Enabling gradient checkpointing")
        if not args.use_qlora:
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Configure LoRA with STABILITY settings
    print("\n" + "="*50)
    print("Configuring LoRA:")
    print(
        f"  - r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights="gaussian",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print("\n" + "="*50)
    print("Trainable parameters:")
    model.print_trainable_parameters()
    print_memory_stats()

    # Prepare datasets
    print("\n" + "="*50)
    print("Loading datasets...")
    train_dataset = NepaliOCRDataset(
        args.train_csv,
        args.train_dir,
        processor=processor,
        max_image_size=None
    )
    print(f"[OK] Train samples: {len(train_dataset)}")

    eval_dataset = None
    if os.path.exists(args.val_csv):
        eval_dataset = NepaliOCRDataset(
            args.val_csv,
            args.val_dir,
            processor=processor,
            max_image_size=None
        )
        print(f"✓ Validation samples: {len(eval_dataset)}")

    # Create training arguments
    training_args = get_training_args(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        eval_dataset_exists=(eval_dataset is not None),
        gradient_checkpointing=args.use_gradient_checkpointing
    )

    # Create collate function
    collate_fn_with_processor = partial(
        collate_fn_minimal_masking,
        processor=processor,
        pad_to_multiple_of=8
    )

    # Setup callbacks with stability monitoring
    callbacks = [StabilityCallback(early_stopping_patience=3)]

    # Initialize trainer
    print("\n" + "="*50)
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn_with_processor,
        callbacks=callbacks,
    )

    # Print training info
    print("\n" + "="*50)
    print("STABILIZED TRAINING CONFIGURATION:")
    print(f"Model: {args.model_name}")
    print(
        f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(
        f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Q-LoRA: {args.use_qlora}")
    print(f"Gradient checkpointing: {args.use_gradient_checkpointing}")
    print("="*50 + "\n")

    # Train
    print("Starting training...")
    print_memory_stats()

    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "="*50)
            print(f"[ERROR]: {e}")
            print("="*50)
            raise
        else:
            raise

    # Save final adapter
    print("\n" + "="*50)
    print("Saving final model...")
    final_adapter_path = Path(args.output_dir) / "final_adapter"
    final_adapter_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_adapter_path)
    processor.save_pretrained(final_adapter_path)

    print(f"[OK] Training complete! Adapter saved to {final_adapter_path}")

    # Save training config
    import json
    config = {
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "use_qlora": args.use_qlora,
        "use_flash_attn": args.use_flash_attn,
        "use_gradient_checkpointing": args.use_gradient_checkpointing,
        "max_image_size": args.max_image_size,
    }
    with open(Path(args.output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"[OK] Config saved to {Path(args.output_dir) / 'training_config.json'}")
    print("="*50)
    print_memory_stats()


if __name__ == "__main__":
    main()

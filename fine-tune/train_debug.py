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
from peft import LoraConfig, get_peft_model, TaskType

from dataset import NepaliOCRDataset, collate_fn_minimal_masking


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
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float,
                        default=1e-5)  # Very conservative
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # Flags
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")
    parser.add_argument("--debug_data", action="store_true", default=False,
                        help="Run data validation before training")

    args = parser.parse_args()
    return args


def validate_dataset(dataset, processor, num_samples=5):
    """
    Validate dataset and collation to catch issues before training.
    """
    print("\n" + "="*70)
    print("DATASET VALIDATION")
    print("="*70)

    try:
        print(f"Total samples: {len(dataset)}")

        # Check a few samples
        for i in range(min(num_samples, len(dataset))):
            print(f"\n--- Sample {i} ---")
            sample = dataset[i]

            print(f"Text: '{sample['text']}'")
            print(f"Text length: {len(sample['text'])}")
            print(f"Image size: {sample['image'].size}")
            print(f"Image path: {sample['image_path']}")

            # Test tokenization
            tokens = processor.tokenizer.encode(sample['text'])
            print(f"Token IDs: {tokens}")
            print(f"Num tokens: {len(tokens)}")

            # Check for out-of-vocab tokens
            vocab_size = len(processor.tokenizer)
            if any(t >= vocab_size for t in tokens):
                print(
                    f" WARNING: Token IDs exceed vocab size {vocab_size}!")
                print(f"   Max token: {max(tokens)}")

        # Test collation
        print("\n" + "-"*70)
        print("Testing collation...")
        test_samples = [dataset[i] for i in range(min(2, len(dataset)))]

        batch = collate_fn_minimal_masking(test_samples, processor=processor)

        print(f"Batch keys: {batch.keys()}")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")
        print(f"labels shape: {batch['labels'].shape}")

        # Check labels
        for i in range(len(test_samples)):
            labels_i = batch['labels'][i]
            non_masked = (labels_i != -100).sum().item()
            total = len(labels_i)
            print(f"\nSample {i}:")
            print(f"  Total tokens: {total}")
            print(f"  Masked tokens (-100): {total - non_masked}")
            print(f"  Learning tokens: {non_masked}")
            print(f"  First 20 labels: {labels_i[:20].tolist()}")
            print(f"  Last 20 labels: {labels_i[-20:].tolist()}")

            if non_masked == 0:
                print("  ERROR: ALL TOKENS ARE MASKED!")
                print("  This will cause NaN loss - no learning signal!")
                return False

            # Check for invalid label IDs
            valid_labels = labels_i[labels_i != -100]
            if len(valid_labels) > 0:
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()
                vocab_size = len(processor.tokenizer)

                print(f"  Label range: {min_label} to {max_label}")

                if max_label >= vocab_size:
                    print(
                        f"  ERROR: Labels exceed vocab size {vocab_size}!")
                    return False

                if min_label < 0 and min_label != -100:
                    print(f"  ERROR: Invalid negative label {min_label}!")
                    return False

        print("\n" + "="*70)
        print("✓ Dataset validation PASSED")
        print("="*70)
        return True

    except Exception as e:
        print(f"\n[X] Dataset validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_training_args(output_dir, num_epochs, batch_size, gradient_accumulation_steps,
                      learning_rate, warmup_steps, max_grad_norm, eval_dataset_exists,
                      gradient_checkpointing):
    """
    Create TrainingArguments for full FP16 LoRA (no quantization).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    common_args = {
        "output_dir": str(output_path),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,

        # FP16 settings (no quantization)
        "bf16": False,
        "fp16": True,
        "fp16_opt_level": "O1",

        # Gradient clipping
        "max_grad_norm": max_grad_norm,

        # Logging
        "logging_steps": 1,  # Log every step for debugging
        "logging_first_step": True,
        "logging_nan_inf_filter": False,

        # Saving
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,

        # Memory
        "remove_unused_columns": False,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "gradient_checkpointing": gradient_checkpointing,
        "save_safetensors": True,

        # Optimizer
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,

        # Other
        "ddp_find_unused_parameters": False,
        "report_to": ["tensorboard"],
    }

    # Handle evaluation
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
        except TypeError:
            common_args.update({
                "evaluation_strategy": "steps",
                "eval_steps": 100,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })
    else:
        try:
            test_args = TrainingArguments(
                output_dir="/tmp/test", eval_strategy="no")
            common_args["eval_strategy"] = "no"
        except TypeError:
            common_args["evaluation_strategy"] = "no"

    return TrainingArguments(**common_args)


def print_model_info(model):
    """Print detailed model information."""
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Check for NaN/Inf in initial weights
    print("\nChecking initial weights...")
    has_nan = False
    has_inf = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any():
                print(f" [ERROR]: NaN in {name}")
                has_nan = True
            if torch.isinf(param).any():
                print(f"  [ERROR]: Inf in {name}")
                has_inf = True

    if not has_nan and not has_inf:
        print(" [PASS] All weights are valid (no NaN/Inf)")

    print("="*70)


class DebugCallback(EarlyStoppingCallback):
    """Enhanced callback with detailed debugging."""

    def __init__(self, early_stopping_patience=3):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.nan_count = 0
        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_count += 1

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss", 0)
            grad_norm = logs.get("grad_norm", 0)
            lr = logs.get("learning_rate", 0)

            # Detailed logging every step
            if self.step_count <= 10 or self.step_count % 10 == 0:
                print(f"\n--- Step {self.step_count} ---")
                print(f"  Loss: {loss}")
                print(f"  Grad Norm: {grad_norm}")
                print(f"  Learning Rate: {lr}")

                # Check GPU memory
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(
                        f"  GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB")

            # Check for NaN
            if loss != loss or grad_norm != grad_norm:
                self.nan_count += 1
                print(
                    f"\n  WARNING: NaN detected at step {self.step_count}! (count: {self.nan_count})")
                print(f"   Loss: {loss}, Grad Norm: {grad_norm}")

                # Check model weights for NaN
                if model is not None:
                    print("   Checking model weights...")
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"     NaN in gradient of {name}")
                            if torch.isnan(param).any():
                                print(f"     NaN in weight of {name}")

                if self.nan_count >= 3:
                    print("\n STOPPING: Too many NaN occurrences!")
                    control.should_training_stop = True

        return super().on_log(args, state, control, logs=logs, **kwargs)


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("NEPALI OCR - FULL FP16 LORA TRAINING (NO QUANTIZATION)")
    print("="*70)

    # GPU check
    if not torch.cuda.is_available():
        print("  WARNING: No GPU available!")
        return

    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

    device = torch.device("cuda")
    torch.cuda.empty_cache()

    # Load processor
    print("\n" + "="*70)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    print(" Processor loaded")

    # Load datasets
    print("\n" + "="*70)
    print("Loading datasets...")
    train_dataset = NepaliOCRDataset(
        args.train_csv,
        args.train_dir,
        processor=processor
    )
    print(f" Train samples: {len(train_dataset)}")

    eval_dataset = None
    if os.path.exists(args.val_csv):
        eval_dataset = NepaliOCRDataset(
            args.val_csv,
            args.val_dir,
            processor=processor
        )
        print(f"✓ Validation samples: {len(eval_dataset)}")

    # CRITICAL: Validate data before training
    if args.debug_data or True:  # Always validate by default
        if not validate_dataset(train_dataset, processor, num_samples=3):
            print("\n Dataset validation failed! Fix data issues before training.")
            return

    # Load model - FULL FP16, NO QUANTIZATION
    print("\n" + "="*70)
    print(f"Loading model: {args.model_name}")
    print("Mode: FULL FP16 (no quantization)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision="d7d1f3777c5f5dc95028e0e4bad350d88d214f7d",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(" Model loaded")

    # Print GPU memory after model load
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"GPU Memory after model load: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Enable gradient checkpointing
    if args.use_gradient_checkpointing:
        print("\n" + "="*70)
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print(" Gradient checkpointing enabled")

    # Configure LoRA
    print("\n" + "="*70)
    print("Configuring LoRA...")
    print(
        f"  r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

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
        init_lora_weights=True,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print_model_info(model)

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

    # Setup callbacks
    callbacks = [DebugCallback(early_stopping_patience=5)]

    # Initialize trainer
    print("\n" + "="*70)
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn_with_processor,
        callbacks=callbacks,
    )

    # Print training configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION:")
    print("="*70)
    print(f"  Model: {args.model_name}")
    print("  Mode: Full FP16 LoRA (no quantization)")
    print(
        f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Max grad norm: {args.max_grad_norm}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Gradient checkpointing: {args.use_gradient_checkpointing}")
    print("="*70)

    # Start training
    print("\nStarting training...")
    print("Watching for NaN issues...\n")

    try:
        trainer.train()

        # Save
        print("\n" + "="*70)
        print("Saving model...")
        final_adapter_path = Path(args.output_dir) / "final_adapter"
        final_adapter_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(final_adapter_path)
        processor.save_pretrained(final_adapter_path)

        print(f"[PASS] Model saved to {final_adapter_path}")
        print("="*70)

    except Exception as e:
        print(f"\n[X] Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Diagnostic script to test all components of the PEFT pipeline
"""

import sys
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

print("\n" + "=" * 70)
print("COMPREHENSIVE DIAGNOSTIC")
print("=" * 70)

# Check 1: Import and test minimal masking
print("\n1. Testing minimal masking import and function...")
try:
    from dataset import NepaliOCRDataset, collate_fn_minimal_masking

    processor = AutoProcessor.from_pretrained(
        "PaddlePaddle/PaddleOCR-VL",
        trust_remote_code=True
    )

    dataset = NepaliOCRDataset(
        "/kaggle/input/datasets/unspoiledegg/nep-handwritten-paddleocr-vl/data/train/labels.csv",
        "/kaggle/input/datasets/unspoiledegg/nep-handwritten-paddleocr-vl/data/train",
        processor=processor
    )

    sample = dataset[0]
    print(f"   Sample text: '{sample['text']}'")

    # Test collation
    batch = collate_fn_minimal_masking([sample], processor=processor)

    labels = batch['labels'][0]
    input_ids = batch['input_ids'][0]

    total = len(labels)
    learning = (labels != -100).sum().item()
    percentage = 100 * learning / total

    print(f"   Total tokens: {total}")
    print(f"   Learning tokens: {learning} ({percentage:.1f}%)")
    print(f"   Masked tokens: {total - learning} ({100 - percentage:.1f}%)")

    if percentage < 60:
        print("   [X] PROBLEM: Learning percentage too low!")
        print(f"   Expected ~80%, got {percentage:.1f}%")
        print("   Minimal masking is NOT working!")
    else:
        print("   [OK] Minimal masking working correctly")

    # Check if labels have valid tokens
    valid_labels = labels[labels != -100]
    if len(valid_labels) == 0:
        print("   [X] CRITICAL: No learning tokens at all!")
    else:
        print(f"   [OK] Has {len(valid_labels)} valid learning tokens")

        # Check label values
        max_label = valid_labels.max().item()
        min_label = valid_labels.min().item()
        vocab_size = len(processor.tokenizer)

        print(f"   Label range: {min_label} to {max_label}")
        print(f"   Vocab size: {vocab_size}")

        if max_label >= vocab_size:
            print("  [X] CRITICAL: Labels exceed vocab size!")
        else:
            print("   [OK] All labels within vocab range")

except Exception as e:
    print(f"  [X] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 2: Test model loading and forward pass
print("\n2. Testing model loading and forward pass...")
try:
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "PaddlePaddle/PaddleOCR-VL",
        revision="d7d1f3777c5f5dc95028e0e4bad350d88d214f7d",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("   [OK] Model loaded")

    # Check model dtype
    first_param = next(model.parameters())
    print(f"   Model dtype: {first_param.dtype}")
    print(f"   Model device: {first_param.device}")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print("   [OK] Gradient checkpointing enabled")

    # Apply LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
    )

    model = get_peft_model(model, lora_config)
    print("   [OK] LoRA applied")

    # Check trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.2f}%)")

    # Check for NaN in initial weights
    has_nan = False
    for name, param in model.named_parameters():
        if param.requires_grad and torch.isnan(param).any():
            print(f"   [X] NaN in initial weights: {name}")
            has_nan = True

    if not has_nan:
        print("   [OK] No NaN in initial weights")
    # Try a forward pass
    print("\n3. Testing forward pass with actual data...")

    # Get a small batch
    batch = collate_fn_minimal_masking([dataset[0]], processor=processor)

    # Move to model device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

    print(f"   Loss: {loss.item():.4f}")
    print(f"   Loss is NaN: {torch.isnan(loss).item()}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits has NaN: {torch.isnan(logits).any().item()}")

    if torch.isnan(loss):
        print("   [X] CRITICAL: Loss is NaN even without training!")
        print("   This suggests a label or model issue")

        # Debug: Check what the model is predicting
        print("\n   Debugging loss calculation...")
        print(f"   Labels shape: {batch['labels'].shape}")
        print(f"   Labels sample: {batch['labels'][0][:20].tolist()}")

        # Check if all labels are -100
        non_masked = (batch['labels'] != -100).sum().item()
        print(f"   Non-masked labels: {non_masked}")

        if non_masked == 0:
            print("   [X] CRITICAL: ALL LABELS ARE MASKED!")
            print("   This is why loss is NaN!")
        else:
            # Check label values
            valid_labels = batch['labels'][batch['labels'] != -100]
            print(
                f"   Valid label range: {
                    valid_labels.min().item()} to {
                    valid_labels.max().item()}"
            )
            print(f"   Vocab size: {len(processor.tokenizer)}")

            if valid_labels.max().item() >= len(processor.tokenizer):
                print("   [X] CRITICAL: Labels exceed vocab size!")
    else:
        print("   [OK] Forward pass successful, loss is valid")

    # Try with gradient computation
    print("\n4. Testing backward pass...")
    model.train()

    # Forward with gradients
    outputs = model(**batch)
    loss = outputs.loss

    print(f"   Loss: {loss.item():.4f}")

    # Backward
    try:
        loss.backward()
        print("   [OK] Backward pass successful")

        # Check gradients
        has_grad = False
        has_nan_grad = False

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any():
                    print(f"   [X] NaN gradient in: {name}")
                    has_nan_grad = True

        if not has_grad:
            print("   [!]  WARNING: No gradients computed!")
        elif has_nan_grad:
            print("   [X] CRITICAL: NaN in gradients!")
        else:
            print("   [OK] Gradients computed successfully, no NaN")

    except Exception as e:
        print(f"   [X] Backward pass failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"   [X] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

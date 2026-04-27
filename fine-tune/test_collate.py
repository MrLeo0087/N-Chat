"""
Quick test to verify the fixed collate_fn works correctly.
Run this before starting training to ensure labels are not all masked.
"""
import sys
from pathlib import Path
from transformers import AutoProcessor
from dataset import NepaliOCRDataset, collate_fn_minimal_masking

sys.path.insert(0, str(Path(__file__).parent))


def test_collation():
    print("\n" + "=" * 70)
    print("TESTING FIXED COLLATE FUNCTION")
    print("=" * 70)

    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        "PaddlePaddle/PaddleOCR-VL",
        trust_remote_code=True
    )
    print("✓ Processor loaded")

    # Load dataset
    print("\nLoading dataset...")
    dataset = NepaliOCRDataset(
        "/kaggle/input/datasets/unspoiledegg/nep-handwritten-paddleocr-vl/data/train/labels.csv",
        "/kaggle/input/datasets/unspoiledegg/nep-handwritten-paddleocr-vl/data/train",
        processor=processor
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Test first sample
    print("\n" + "=" * 70)
    print("Testing first sample...")
    print("=" * 70)

    sample = dataset[0]
    print(f"\nText: '{sample['text']}'")
    print(f"Image size: {sample['image'].size}")

    # Test collation
    print("\n" + "=" * 70)
    print("Testing collation with 2 samples...")
    print("=" * 70)

    batch = collate_fn_minimal_masking(
        [dataset[0], dataset[1]], processor=processor)

    print("\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    # Analyze labels
    print("\n" + "=" * 70)
    print("LABEL ANALYSIS")
    print("=" * 70)

    all_good = True

    for i in range(len(batch['labels'])):
        labels_i = batch['labels'][i]
        total = len(labels_i)
        non_masked = (labels_i != -100).sum().item()
        masked = total - non_masked

        print(f"\nSample {i}:")
        print(f"  Total tokens: {total}")
        print(f"  Masked tokens (-100): {masked}")
        print(f"  Learning tokens: {non_masked}")
        print(f"  Learning %: {100 * non_masked / total:.1f}%")

        if non_masked == 0:
            print("ERROR: ALL TOKENS MASKED!")
            all_good = False
        elif non_masked < 5:
            print("WARNING: Very few learning tokens!")
            all_good = False
        else:
            print("[OK]")

        # Show some actual learning tokens
        if non_masked > 0:
            learning_mask = labels_i != -100
            learning_ids = labels_i[learning_mask]
            decoded = processor.tokenizer.decode(learning_ids)
            print(f"  Learning text: '{decoded}'")

    # Final verdict
    print("\n" + "=" * 70)
    if all_good:
        print("COLLATION TEST PASSED!")
        print("   You can now train without NaN issues.")
    else:
        print("COLLATION TEST FAILED!")
        print("   Fix the issues above before training.")
    print("=" * 70 + "\n")

    return all_good


if __name__ == "__main__":
    try:
        success = test_collation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[X] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

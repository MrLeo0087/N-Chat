"""
Inference script for PaddleOCR-VL with LoRA adapter.
Usage: python inference.py --adapter_path ./nepali-ocr-lora/final_adapter --image_path test.jpg
"""

import torch
import argparse
from pathlib import Path
from PIL import Image

from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


def load_model_with_adapter(
    base_model_name="PaddlePaddle/PaddleOCR-VL",
    adapter_path=None,
    device="cuda"
):
    """
    Load PaddleOCR-VL model with LoRA adapter.

    Args:
        base_model_name: HuggingFace model name
        adapter_path: Path to LoRA adapter directory
        device: Device to load model on

    Returns:
        model, processor
    """
    print(f"\nLoading base model: {base_model_name}")

    # Load base model WITHOUT Flash Attention
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        dtype=torch.float16,  # Updated parameter name
        device_map="auto",
        low_cpu_mem_usage=True,
        # Don't use Flash Attention
        attn_implementation=None,
    )

    print("Base model loaded")

    # Load adapter if provided
    if adapter_path:
        print(f"\nLoading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False
        )
        print("LoRA adapter loaded")

        # Load processor from adapter path (has the correct tokenizer)
        processor = AutoProcessor.from_pretrained(
            adapter_path,
            trust_remote_code=True
        )
        print("Processor loaded from adapter")
    else:
        model = base_model
        processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        print("Using base model without adapter")

    model.eval()

    return model, processor


def run_ocr(
    model,
    processor,
    image_path,
    max_new_tokens=100,
    do_sample=False,
    temperature=1.0,
):
    """
    Run OCR on a single image.

    Args:
        model: The model
        processor: The processor
        image_path: Path to image file
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to sample (False = greedy)
        temperature: Sampling temperature

    Returns:
        OCR text result
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"\nImage: {image_path}")
    print(f"Size: {image.size}")

    # Create prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,  # Pass messages directly, not wrapped in list
        tokenize=False,
        add_generation_prompt=True
    )

    # Handle case where chat template returns a list
    if isinstance(text, list):
        text = text[0] if len(text) > 0 else ""

    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # Generate
    print("Generating OCR output...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip input)
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_length:]

    result = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return result.strip()


def run_batch_ocr(
    model,
    processor,
    image_paths,
    output_file=None,
    max_new_tokens=100,
):
    """
    Run OCR on multiple images.

    Args:
        model: The model
        processor: The processor
        image_paths: List of image paths
        output_file: Optional file to save results
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of OCR results
    """
    results = []

    print(f"\nProcessing {len(image_paths)} images...")

    for i, image_path in enumerate(image_paths):
        print(f"\n{'=' * 70}")
        print(f"Image {i + 1}/{len(image_paths)}")
        print(f"{'=' * 70}")

        try:
            result = run_ocr(
                model,
                processor,
                image_path,
                max_new_tokens=max_new_tokens
            )

            print("\nOCR Result:")
            print(f"  {result}")

            results.append({
                'image_path': str(image_path),
                'text': result,
                'success': True
            })

        except Exception as e:
            print(f"\n Error processing {image_path}: {e}")
            results.append({
                'image_path': str(image_path),
                'text': '',
                'success': False,
                'error': str(e)
            })

    # Save results if requested
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') a>s f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Results saved to {output_file}")

    return results


def process_test_csv(
    model,
    processor,
    test_csv,
    test_dir,
    output_csv=None,
    max_new_tokens=100,
):
    """
    Process test set from CSV file with ground truth labels.

    Args:
        model: The model
        processor: The processor
        test_csv: Path to CSV with columns: image_path, text
        test_dir: Root directory for images (images in test_dir/cropped/)
        output_csv: Path to save predictions CSV
        max_new_tokens: Maximum tokens to generate

    Returns:
        DataFrame with predictions and metrics
    """
    import pandas as pd
    from pathlib import Path

    print(f"\nLoading test set from: {test_csv}")
    df = pd.read_csv(test_csv)

    print(f"Total samples: {len(df)}")

    # Prepare results
    predictions = []
    ground_truths = []
    image_paths = []

    test_dir_path = Path(test_dir)

    for idx, row in df.iterrows():
        image_path = test_dir_path / "cropped" / row['image_path']
        ground_truth = str(row['text']).strip()

        print(f"\n{'=' * 70}")
        print(f"Sample {idx + 1}/{len(df)}")
        print(f"{'=' * 70}")
        print(f"Image: {row['image_path']}")
        print(f"Ground truth: {ground_truth}")

        try:
            # Check if image exists
            if not image_path.exists():
                print(f" Image not found: {image_path}")
                prediction = ""
            else:
                # Run inference
                prediction = run_ocr(
                    model,
                    processor,
                    image_path,
                    max_new_tokens=max_new_tokens
                )
                print(f"Prediction: {prediction}")

            predictions.append(prediction)
            ground_truths.append(ground_truth)
            image_paths.append(row['image_path'])

        except Exception as e:
            print(f" Error: {e}")
            predictions.append("")
            ground_truths.append(ground_truth)
            image_paths.append(row['image_path'])

    # Create results DataFrame
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'ground_truth': ground_truths,
        'prediction': predictions,
    })

    # Calculate character-level accuracy
    results_df['correct'] = results_df.apply(
        lambda row: row['ground_truth'] == row['prediction'],
        axis=1
    )

    # Calculate metrics
    total = len(results_df)
    correct = results_df['correct'].sum()
    accuracy = 100 * correct / total if total > 0 else 0

    # Calculate Character Error Rate (CER)
    def calculate_cer(gt, pred):
        """Calculate character error rate using Levenshtein distance."""
        if len(gt) == 0:
            return 1.0 if len(pred) > 0 else 0.0

        # Simple Levenshtein distance
        m, n = len(gt), len(pred)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gt[i - 1] == pred[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,    # deletion
                        dp[i][j - 1] + 1,    # insertion
                        dp[i - 1][j - 1] + 1   # substitution
                    )

        return dp[m][n] / len(gt)

    results_df['cer'] = results_df.apply(
        lambda row: calculate_cer(row['ground_truth'], row['prediction']),
        axis=1
    )

    avg_cer = results_df['cer'].mean() * 100

    # Save results
    if output_csv:
        results_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n✓ Results saved to {output_csv}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Total samples: {total}")
    print(f"Exact matches: {correct} ({accuracy:.2f}%)")
    print(f"Average CER: {avg_cer:.2f}%")
    print(f"{'=' * 70}")

    # Show some examples
    print("\nSample predictions:")
    print(f"{'-' * 70}")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        match = "✓" if row['correct'] else "✗"
        print(f"\n{match} Image: {row['image_path']}")
        print(f"  Ground truth: {row['ground_truth']}")
        print(f"  Prediction:   {row['prediction']}")
        print(f"  CER: {row['cer'] * 100:.1f}%")

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR Inference with PaddleOCR-VL + LoRA")

    # Model args
    parser.add_argument(
        "--base_model",
        type=str,
        default="PaddlePaddle/PaddleOCR-VL",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (e.g., ./nepali-ocr-lora/final_adapter)"
    )

    # Input args
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Single image path"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images"
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="Text file with list of image paths"
    )

    # CSV-based test set args
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="CSV file with columns: image_path, text (ground truth)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Root directory for test images (images in test_dir/cropped/)"
    )

    # Output args
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for batch results"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file with predictions and ground truth"
    )

    # Generation args
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PaddleOCR-VL Inference with LoRA")
    print("=" * 70)

    # Load model
    model, processor = load_model_with_adapter(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path
    )

    # Check if using CSV test set
    if args.test_csv:
        if not args.test_dir:
            print("\n Error: --test_dir is required when using --test_csv")
            return

        # Process CSV test set
        results_df = process_test_csv(
            model,
            processor,
            test_csv=args.test_csv,
            test_dir=args.test_dir,
            output_csv=args.output_csv,
            max_new_tokens=args.max_new_tokens
        )
        return

    # Collect image paths (original behavior)
    image_paths = []

    if args.image_path:
        image_paths.append(args.image_path)

    if args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(image_dir.glob(ext))

    if args.image_list:
        with open(args.image_list, 'r') as f:
            image_paths.extend([line.strip() for line in f if line.strip()])

    if not image_paths:
        print("\n No images specified!")
        print("\nUsage examples:")
        print("\n  # Evaluate on test set with metrics")
        print("  python inference.py \\")
        print("    --adapter_path ./nepali-ocr-lora/final_adapter \\")
        print("    --test_csv data/test/labels.csv \\")
        print("    --test_dir data/test \\")
        print("    --output_csv predictions.csv")
        print("\n  # Single image")
        print("  python inference.py --adapter_path ./nepali-ocr-lora/final_adapter --image_path test.jpg")
        print("\n  # Directory of images")
        print("  python inference.py --adapter_path ./nepali-ocr-lora/final_adapter --image_dir ./test_images")
        print("\n  # List of images")
        print("  python inference.py --adapter_path ./nepali-ocr-lora/final_adapter --image_list images.txt")
        return

    # Run inference
    if len(image_paths) == 1:
        # Single image
        result = run_ocr(
            model,
            processor,
            image_paths[0],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature
        )

        print(f"\n{'=' * 70}")
        print("RESULT")
        print(f"{'=' * 70}")
        print(f"{result}")
        print(f"{'=' * 70}")

    else:
        # Batch processing
        results = run_batch_ocr(
            model,
            processor,
            image_paths,
            output_file=args.output_file,
            max_new_tokens=args.max_new_tokens
        )

        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

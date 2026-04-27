import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model", type=str,
                        default="PaddlePaddle/PaddleOCR-VL")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision="d7d1f3777c5f5dc95028e0e4bad350d88d214f7d",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging adapter...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(args.output_path)

    # Copy processor
    processor = AutoProcessor.from_pretrained(
        args.adapter_path, trust_remote_code=True)
    processor.save_pretrained(args.output_path)

    print("Merge complete!")


if __name__ == "__main__":
    main()

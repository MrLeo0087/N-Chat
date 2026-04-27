#!/usr/bin/env python3
"""
Complete fix for PaddleOCR-VL compatibility issues.
This script clears the HuggingFace cache and ensures correct model version is loaded.
"""

import os
import shutil
from pathlib import Path

def clear_paddleocr_cache():
    """Clear cached PaddleOCR-VL files to force redownload of correct version."""
    cache_locations = [
        Path.home() / ".cache" / "huggingface" / "hub" / "models--PaddlePaddle--PaddleOCR-VL",
        Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "PaddlePaddle",
    ]
    
    print("Clearing PaddleOCR-VL cache...")
    
    for cache_dir in cache_locations:
        if cache_dir.exists():
            print(f"  Removing: {cache_dir}")
            shutil.rmtree(cache_dir)
            print(f"  ✓ Cleared")
        else:
            print(f"  - Not found: {cache_dir}")
    
    print("\n✓ Cache cleared")


def verify_transformers_version():
    """Verify transformers version is compatible."""
    import transformers
    version = transformers.__version__
    
    print(f"\nTransformers version: {version}")
    
    if version.startswith("5."):
        print("❌ ERROR: transformers 5.x is not compatible!")
        print("   Install: pip install transformers==4.46.3 --force-reinstall")
        return False
    elif version.startswith("4.46") or version.startswith("4.47"):
        print("✓ Version compatible")
        return True
    else:
        print(f"⚠️  WARNING: Untested version {version}")
        print("   Recommended: pip install transformers==4.46.3")
        return True


def main():
    print("="*70)
    print("PaddleOCR-VL Complete Fix")
    print("="*70)
    
    # Check transformers version
    if not verify_transformers_version():
        print("\nPlease install correct transformers version first:")
        print("  pip install transformers==4.46.3 --force-reinstall")
        return
    
    # Clear cache
    clear_paddleocr_cache()
    
    print("\n" + "="*70)
    print("✅ FIX COMPLETE")
    print("="*70)
    print("\nNow update your training script to use the pinned revision:")
    print("\nIn your model loading code, add:")
    print('  revision="2b77538ef936207f60c16b45082841068987d08c"')
    print("\nExample:")
    print("""
model = AutoModelForCausalLM.from_pretrained(
    "PaddlePaddle/PaddleOCR-VL",
    revision="2b77538ef936207f60c16b45082841068987d08c",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
""")
    print("\nThen run your training script.")


if __name__ == "__main__":
    main()

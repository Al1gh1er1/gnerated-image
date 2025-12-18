"""
Download and cache the SDXL Turbo model weights.
"""

import os
import sys
import traceback

# Check if required packages are installed
try:
    import torch
except ImportError as e:
    print(f"❌ Error: PyTorch is not installed: {e}")
    sys.exit(1)

try:
    from diffusers import AutoPipelineForText2Image
except ImportError as e:
    print(f"❌ Error: diffusers is not installed: {e}")
    sys.exit(1)


def download_models():
    """Download and cache SDXL Turbo model."""
    print("=" * 60)
    print("Downloading SDXL Turbo model...")
    print("=" * 60)
    
    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        print("\nStarting model download...")
        print("Note: This may take several minutes depending on your connection speed...")
        
        # Download SDXL Turbo pipeline
        # During Docker build, GPU is not available, so we don't load to device
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=cache_dir,
            device_map=None,  # Don't auto-map to device during build
        )

        print("✅ SDXL Turbo model downloaded successfully!")

        # Note: During Docker build, GPU is not available, so we skip GPU loading
        # The model will be loaded to GPU at runtime in handler.py
        print("ℹ️  Model cached successfully. GPU loading will happen at runtime.")

        print("=" * 60)
        print("✅ Model download and setup completed successfully!")
        print("=" * 60)
        return 0

    except Exception as e:
        print("=" * 60)
        print(f"❌ Error downloading model: {str(e)}")
        print("=" * 60)
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = download_models()
    sys.exit(exit_code)

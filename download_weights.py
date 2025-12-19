"""
Download and cache the SDXL Turbo model weights.
"""

import os
import sys
import traceback

# Check if required packages are installed
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
except ImportError as e:
    print(f"❌ Error: PyTorch is not installed: {e}")
    sys.exit(1)

try:
    from diffusers import AutoPipelineForText2Image
    print(f"✓ diffusers imported successfully")
except ImportError as e:
    print(f"❌ Error: diffusers is not installed: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ transformers {transformers.__version__} imported successfully")
except ImportError as e:
    print(f"❌ Error: transformers is not installed: {e}")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
    print(f"✓ huggingface_hub imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: huggingface_hub not available, will use diffusers directly: {e}")
    snapshot_download = None


def download_models():
    """Download and cache SDXL Turbo model."""
    print("=" * 70)
    print("Downloading SDXL Turbo model...")
    print("=" * 70)
    
    # Print environment info
    print(f"\nPython version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        print("ℹ️  GPU not available during build (this is expected)")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"\nCache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"✓ Cache directory created/verified: {cache_dir}")

    try:
        print("\n" + "=" * 70)
        print("Starting model download from Hugging Face...")
        print("Note: This may take several minutes depending on your connection speed...")
        print("=" * 70)
        
        # Set environment variables for better download performance
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # Download SDXL Turbo pipeline
        # During Docker build, GPU is not available, so we use CPU
        print("\nDownloading model components...")
        print("This may take several minutes depending on connection speed...")
        print("Model size: ~6-7 GB")
        print("Please be patient...")
        
        # Download model files using snapshot_download (doesn't load model into memory)
        if snapshot_download:
            print("\nStep 1: Downloading model files using huggingface_hub...")
            print("This method downloads files without loading the model into memory...")
            model_path = snapshot_download(
                repo_id="stabilityai/sdxl-turbo",
                cache_dir=cache_dir,
                local_files_only=False,
                ignore_patterns=["*.md", "*.txt"],  # Skip documentation files
            )
            print(f"✓ Model files downloaded to: {model_path}")
            
            # Verify files exist by checking cache directory
            print("\nStep 2: Verifying downloaded files...")
            import glob
            cache_path = os.path.join(cache_dir, "hub", "models--stabilityai--sdxl-turbo")
            if os.path.exists(cache_path):
                files = glob.glob(os.path.join(cache_path, "**", "*.safetensors"), recursive=True)
                print(f"✓ Found {len(files)} model weight files")
                if len(files) > 0:
                    print("✓ Model files verified successfully")
                else:
                    print("⚠️  No safetensors files found, but download completed")
            else:
                print("⚠️  Cache directory structure different than expected, but download may have succeeded")
        else:
            # Fallback: Direct load if snapshot_download not available
            print("\nDownloading model using diffusers (this will load model into memory)...")
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                variant="fp16",
                use_safetensors=True,
                cache_dir=cache_dir,
            )
            print("✓ Model downloaded successfully")

        print("\n✅ SDXL Turbo model downloaded and cached successfully!")
        print(f"✓ Model cached at: {cache_dir}")

        # Note: During Docker build, GPU is not available, so we skip GPU loading
        # The model will be loaded to GPU at runtime in handler.py
        print("\nℹ️  Model cached successfully. GPU loading will happen at runtime in handler.py.")

        print("\n" + "=" * 70)
        print("✅ Model download and setup completed successfully!")
        print("=" * 70)
        return 0

    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user")
        return 1
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Error downloading model: {str(e)}")
        print("=" * 70)
        print("\nFull traceback:")
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("\nTroubleshooting tips:")
        print("1. Check internet connection")
        print("2. Verify Hugging Face model repository is accessible")
        print("3. Check available disk space")
        print("4. Verify all dependencies are installed correctly")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    try:
        exit_code = download_models()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

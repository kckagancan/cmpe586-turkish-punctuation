"""Check the project environment without downloading data or models."""

import importlib.metadata
import platform
import sys

import torch

import turkish_punctuation


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}")
    print(f"Project: {turkish_punctuation.__version__}")
    for package in ("torch", "transformers", "datasets", "accelerate", "ipykernel"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU environment ready. Use a Colab GPU for training.")
    result = torch.tensor([1.0, 2.0]).sum().item()
    if result != 3.0:
        raise RuntimeError("PyTorch tensor check failed.")


if __name__ == "__main__":
    main()

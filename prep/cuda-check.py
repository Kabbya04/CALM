"""Verify the conda env can use CUDA for CALM (PyTorch + a tiny CLIP forward)."""

from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    print("CALM CUDA / package check")
    print("=" * 60)

    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(f"FAIL: PyTorch not installed ({exc})")
        print("Install CUDA wheels first, then: pip install -r requirements.txt")
        return 1

    print(f"torch:       {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"cuda build:  {torch.version.cuda}")
    print(f"cudnn:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'n/a'}")

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False.")
        print("Check NVIDIA driver, CUDA wheel (cu118/cu121/cu124), and that this is not a CPU-only build.")
        return 1

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:         {name}")
    print(f"VRAM:        {props.total_memory / (1024 ** 3):.1f} GiB")
    print(f"capability:  {props.major}.{props.minor}")

    try:
        x = torch.randn(1024, 1024, device=device)
        y = x @ x.T
        torch.cuda.synchronize()
        del x, y
        torch.cuda.empty_cache()
        print("matmul:      OK")
    except Exception as exc:
        print(f"FAIL: CUDA matmul failed: {exc}")
        return 1

    try:
        import transformers
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        import numpy as np

        print(f"transformers:{transformers.__version__}")
        model_name = "openai/clip-vit-base-patch32"
        print(f"loading:     {model_name} ...")
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        image = Image.fromarray(np.uint8(np.zeros((224, 224, 3))))
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
        assert feats.ndim == 2 and feats.shape[-1] > 0
        del model, processor, inputs, feats
        torch.cuda.empty_cache()
        print("CLIP encode: OK")
    except Exception as exc:
        print(f"FAIL: CLIP GPU check failed: {exc}")
        return 1

    for pkg in ("pandas", "tqdm", "sklearn", "timm", "PIL"):
        try:
            if pkg == "sklearn":
                import sklearn as mod
            elif pkg == "PIL":
                import PIL as mod
            else:
                mod = __import__(pkg)
            ver = getattr(mod, "__version__", "ok")
            print(f"{pkg:<12} {ver}")
        except ImportError:
            print(f"FAIL: missing package {pkg}")
            return 1

    print("=" * 60)
    print("PASS: CUDA cores usable; CALM GPU stack looks ready.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

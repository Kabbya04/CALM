import ssl

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

ssl._create_default_https_context = ssl._create_unverified_context

CLIP_B32 = "openai/clip-vit-base-patch32"


def load_clip(device: str, model_name: str = CLIP_B32):
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


@torch.no_grad()
def embed_pils(images, model, processor, device: str, batch_size: int = 32):
    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        features.append(model.get_image_features(**inputs))
    if not features:
        return torch.empty((0, 512), device=device)
    return F.normalize(torch.cat(features, dim=0), p=2, dim=1)


@torch.no_grad()
def embed_text(prompts, model, processor, device: str):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    features = model.get_text_features(**inputs)
    return F.normalize(features, p=2, dim=1)

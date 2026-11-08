"""
model.py - Layer 2: the trained ResNet that looks at the pixels.

Loads a model saved by src/training/train.py and returns how likely an image
is to be AI-generated (0.0 = real, 1.0 = AI).

Which model file is used:
  1. the REALBAKED_MODEL environment variable, if set, e.g.
         $env:REALBAKED_MODEL = "src\\models\\<run>\\best.pt"
  2. otherwise src/models/detector.pt - the shared model, committed to git
  3. otherwise the newest src/models/<run>/best.pt from your own training

To share a better model with the team, copy its best.pt over detector.pt
and commit it.
"""

import io
import os
import threading
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.training.common import (MODELS_DIR, PREP_SIZE, get_transforms,
                                 load_checkpoint)

_lock = threading.Lock()
_loaded = None   # (model, checkpoint info, transform, device, path) once loaded


def find_model_path():
    env = os.environ.get("REALBAKED_MODEL")
    if env:
        return Path(env)
    shared = MODELS_DIR / "detector.pt"
    if shared.exists():
        return shared
    runs = sorted(MODELS_DIR.glob("*/best.pt"))   # run folders start with date-time
    return runs[-1] if runs else None


def _load():
    """Load the model the first time it's needed, then keep it in memory."""
    global _loaded
    with _lock:
        if _loaded is None:
            path = find_model_path()
            if path is None or not path.exists():
                raise FileNotFoundError(
                    "No trained model found in src/models/. Train one with "
                    "src/training/train.py or copy a best.pt into a folder there.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, ckpt = load_checkpoint(path, device)
            model.eval()
            transform = get_transforms(ckpt["img_size"], train=False)
            _loaded = (model, ckpt, transform, device, path)
            print(f"[classifier] loaded {path} on {device}")
    return _loaded


def prepare_like_training(img):
    """Give the image exactly the treatment training images got in
    prepare_data.py: centre square at 256x256, re-saved as JPEG quality 95.
    The model only knows images that look like that."""
    img = ImageOps.fit(img.convert("RGB"), (PREP_SIZE, PREP_SIZE),
                       Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def predict(image_path):
    """Returns {"prob_ai": 0..1, "model": name, "model_path": path}."""
    model, ckpt, transform, device, path = _load()
    with Image.open(image_path) as img:
        x = transform(prepare_like_training(img)).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x).squeeze()).item()
    return {"prob_ai": prob, "model": ckpt["model_name"], "model_path": str(path)}

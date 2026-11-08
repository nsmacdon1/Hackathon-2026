"""
common.py - pieces shared by prepare_data.py, train.py and eval.py.

Keeping these in one place means training and evaluation always load images
and build the model in exactly the same way. If they ever differed, the model
would be tested on images that look different from what it learned on.
"""

import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths (worked out from this file's location, so the scripts work no matter
# which folder you run them from)
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]   # .../Hackathon-2026/src
DATA_DIR = SRC_DIR / "data"                     # prepared images go here
MODELS_DIR = SRC_DIR / "models"                 # trained models go here

# ---------------------------------------------------------------------------
# Labels and image settings
# ---------------------------------------------------------------------------

# The model outputs ONE number per image: how likely it is to be fake.
# So real = 0 and fake = 1.
CLASSES = {"real": 0, "fake": 1}

# prepare_data.py saves every image at this size (square, in pixels).
PREP_SIZE = 256

# Models pretrained on ImageNet expect colours scaled with these exact values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size, train):
    """How an image is turned into numbers the model can read.

    Training uses a RANDOM crop and a random mirror flip, so the model sees
    slightly different versions of each image every epoch. This is called
    "data augmentation" and stops it from memorising exact images.
    Evaluation uses a fixed centre crop, so results are repeatable.
    """
    # Resize makes any image work (e.g. ones you test by hand later). For
    # images from prepare_data.py it changes nothing: they're already 256.
    steps = [transforms.Resize(PREP_SIZE)]
    if train:
        steps += [transforms.RandomCrop(img_size),
                  transforms.RandomHorizontalFlip()]
    else:
        steps += [transforms.CenterCrop(img_size)]
    steps += [transforms.ToTensor(),                         # pixels -> numbers 0..1
              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Dataset: tells PyTorch where the images are and what their labels are
# ---------------------------------------------------------------------------

class RealFakeDataset(Dataset):
    """Reads the prepared images in data/real and data/fake.

    prepare_data.py names every file  <split>__<generator>__<number>.jpg
    (e.g. "train__imagenet_ai_0419_sdv4__00042.jpg"), so we can tell from the
    name alone which split it belongs to and which generator it came from.
    """

    def __init__(self, data_dir, split, transform):
        self.transform = transform
        self.items = []   # list of (file path, label, generator name)
        for class_name, label in CLASSES.items():
            folder = Path(data_dir) / class_name
            if not folder.is_dir():
                continue
            for entry in os.scandir(folder):
                if not entry.name.endswith(".jpg") or entry.name.count("__") != 2:
                    continue   # skips .gitkeep and anything not made by prepare_data
                file_split, generator, _ = entry.name.split("__")
                if file_split == split:
                    self.items.append((entry.path, label, generator))
        self.items.sort()
        if not self.items:
            raise FileNotFoundError(
                f"No '{split}' images found in {data_dir}. "
                "Run prepare_data.py first.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        # float label, because the loss function compares it to a probability
        return x, torch.tensor(label, dtype=torch.float32)

    def count(self, class_name):
        return sum(1 for _, label, _ in self.items if label == CLASSES[class_name])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(name, pretrained=True):
    """Load a well-known image model and change its last layer.

    The pretrained model ends in a layer that picks one of ImageNet's 1000
    categories ("goldfish", "tractor", ...). We swap that for a layer with a
    single output: the real-vs-fake score. Everything before it - the part
    that has learned to see edges, textures and shapes - is kept.

    The first time a model is used, its pretrained weights are downloaded to
    C:\\Users\\<you>\\.cache\\torch\\hub\\checkpoints\\ (ResNet-50 is ~100 MB).
    """
    model = models.get_model(name, weights="DEFAULT" if pretrained else None)
    if hasattr(model, "fc"):                                   # ResNet family
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif isinstance(getattr(model, "classifier", None), nn.Sequential):
        last = model.classifier[-1]                            # EfficientNet etc.
        model.classifier[-1] = nn.Linear(last.in_features, 1)
    else:
        raise ValueError(f"Don't know how to change the last layer of {name}")
    return model


def load_checkpoint(path, device):
    """Load a model saved by train.py. Returns (model, checkpoint info)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt

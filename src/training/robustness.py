"""
robustness.py - does the model still work on images that have been changed?

Real-world images rarely arrive untouched: they get re-saved as JPEG by apps
and social media, shrunk into thumbnails, blurred, or screenshotted. This
script applies each of those changes to the test images and re-measures
accuracy, so you can see how much the score drops.

How to read the results:
  - A small drop (a few percent) is normal.
  - A big drop in "fakes caught" after JPEG re-saving would suggest the model
    partly learned "compressed twice = real" (the real photos were JPEGs
    before prepare_data.py re-saved them; the AI images were PNGs) instead of
    real AI clues.

Run from the project folder:
    venv\\Scripts\\python src\\training\\robustness.py src\\models\\<run>\\best.pt
"""

import argparse
import io
from functools import partial

import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from common import DATA_DIR, RealFakeDataset, get_transforms, load_checkpoint
from eval import predict


# ---------------------------------------------------------------------------
# The changes we apply. Each takes a PIL image and returns a changed one.
# (They're plain functions at module level so Windows can send them to the
# parallel image-loading workers.)
# ---------------------------------------------------------------------------

def unchanged(img):
    return img


def jpeg(img, quality):
    """Re-save as JPEG in memory, like a messaging app or social media would."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def shrink_and_back(img, factor):
    """Shrink then enlarge again: detail is lost, like a resized upload."""
    w, h = img.size
    small = img.resize((int(w * factor), int(h * factor)), Image.Resampling.BICUBIC)
    return small.resize((w, h), Image.Resampling.BICUBIC)


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))


def screenshot(img):
    """A screenshot of an image shown slightly smaller on screen, saved as PNG
    (lossless, so only the resize changes the pixels)."""
    w, h = img.size
    return img.resize((int(w * 0.9), int(h * 0.9)), Image.Resampling.BILINEAR)


CONDITIONS = [
    ("original", unchanged),
    ("JPEG quality 90", partial(jpeg, quality=90)),
    ("JPEG quality 75", partial(jpeg, quality=75)),
    ("JPEG quality 50", partial(jpeg, quality=50)),
    ("shrink to 50% and back", partial(shrink_and_back, factor=0.5)),
    ("slight blur", partial(blur, radius=1)),
    ("screenshot (90% size)", screenshot),
]


class ChangedDataset(Dataset):
    """The test images, with one change applied before the usual transforms."""

    def __init__(self, base, change, transform):
        self.items, self.change, self.transform = base.items, change, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        with Image.open(path) as img:
            img = self.change(img.convert("RGB"))
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a .pt file saved by train.py")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    transform = get_transforms(ckpt["img_size"], train=False)
    base = RealFakeDataset(args.data_dir, args.split, transform)
    labels = torch.tensor([label for _, label, _ in base.items])
    print(f"Model: {ckpt['model_name']}, epoch {ckpt['epoch']}.  "
          f"Testing {len(base)} '{args.split}' images under {len(CONDITIONS)} conditions.\n")

    print(f"  {'condition':26s} {'overall':>8s} {'real correct':>13s} {'fakes caught':>13s}")
    for name, change in CONDITIONS:
        ds = ChangedDataset(base, change, transform)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
        preds = (predict(model, loader, device) > 0.5).long()
        overall = (preds == labels).float().mean().item()
        real_ok = (preds[labels == 0] == 0).float().mean().item()
        fake_ok = (preds[labels == 1] == 1).float().mean().item()
        print(f"  {name:26s} {overall:8.2%} {real_ok:13.2%} {fake_ok:13.2%}", flush=True)


if __name__ == "__main__":   # required on Windows when loading data in parallel
    main()

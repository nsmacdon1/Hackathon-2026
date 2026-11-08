"""
test_folder.py - test the detector on any folder of images, per generator.

Point it at a folder of AI images (and optionally a folder of real photos).
If a folder contains subfolders, each subfolder counts as one group - e.g.
Synthbuster's dalle3/, midjourney-v5/, stable-diffusion-xl/ ... - and gets
its own score. Images are prepared exactly the way the app prepares them.

For AI folders it reports "fakes caught"; for real folders "real correct".
With --chart it also saves a bar chart (PNG) for slides.

Run from the project folder:
    venv\\Scripts\\python src\\training\\test_folder.py --fake C:\\path\\to\\synthbuster --chart results.png
    venv\\Scripts\\python src\\training\\test_folder.py --fake <ai folder> --real <photo folder>

Add --limit 200 to test only 200 random images per group (faster).
"""

import argparse
import io
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from common import DATA_DIR, MODELS_DIR, PREP_SIZE, get_transforms, load_checkpoint

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Readable names for known dataset folders (anything else is shown as-is)
NICE_NAMES = {
    "dalle2": "DALL-E 2", "dalle3": "DALL-E 3", "firefly": "Adobe Firefly",
    "midjourney-v5": "Midjourney v5", "glide": "GLIDE",
    "stable-diffusion-1-3": "Stable Diffusion 1.3",
    "stable-diffusion-1-4": "Stable Diffusion 1.4",
    "stable-diffusion-2": "Stable Diffusion 2", "stable-diffusion-xl": "SDXL",
}


def prepare_like_training(img):
    """Same as src/classifier/model.py: centre square at 256x256, re-saved as
    JPEG quality 95 - so these results match what the app would say."""
    img = ImageOps.fit(img.convert("RGB"), (PREP_SIZE, PREP_SIZE), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def find_groups(folder):
    """{group name: [image paths]}. Subfolders with images are separate groups;
    images directly inside the folder form a group named after the folder."""
    folder = Path(folder)
    groups = {}
    for root, _, files in os.walk(folder):
        imgs = sorted(os.path.join(root, f) for f in files
                      if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        if imgs:
            rel = Path(root).relative_to(folder)
            name = folder.name if str(rel) == "." else str(rel).replace(os.sep, "/")
            groups[NICE_NAMES.get(name.split("/")[-1].lower(), name)] = imgs
    return groups


class FolderDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as img:
                return self.transform(prepare_like_training(img)), True
        except Exception:                                   # unreadable file
            return torch.zeros(3, 224, 224), False


@torch.no_grad()
def score(model, paths, transform, device, batch_size, workers):
    """Returns the AI-probability of every readable image."""
    loader = DataLoader(FolderDataset(paths, transform), batch_size=batch_size,
                        num_workers=workers)
    probs = []
    for x, ok in loader:
        p = torch.sigmoid(model(x.to(device)).squeeze(1)).cpu()
        probs.append(p[ok])
    return torch.cat(probs)


def save_chart(rows, path, title):
    """Horizontal bar chart: one bar per group, value at the bar's tip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    BLUE, TEXT, TEXT2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    rows = sorted(rows, key=lambda r: r[2])                 # lowest at the bottom
    fig_h = 1.2 + 0.42 * len(rows)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bar_h = 0.5
    for i, (label, kind, value, n) in enumerate(rows):
        width = value * 100
        if width > 0:
            ax.add_patch(FancyBboxPatch((0, i - bar_h / 2), width, bar_h,
                                        boxstyle="round,pad=0,rounding_size=0.6",
                                        mutation_aspect=0.25, linewidth=0,
                                        facecolor=BLUE))
            # square off the start of the bar: only the tip is rounded
            ax.add_patch(plt.Rectangle((0, i - bar_h / 2), min(width, 2), bar_h,
                                       linewidth=0, facecolor=BLUE))
        ax.text(width + 1.2, i, f"{value:.0%}", va="center", ha="left",
                fontsize=10, color=TEXT, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{label}  ({kind}, n={n})" for label, kind, _, n in rows],
                       fontsize=9.5, color=TEXT)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=TEXT2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(50, color=TEXT2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(50, len(rows) - 0.35, " coin flip", fontsize=8, color=TEXT2, va="bottom")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", help="folder of AI images (subfolders = generators)")
    p.add_argument("--real", help="folder of real photos (subfolders = sources)")
    p.add_argument("--checkpoint", default=str(MODELS_DIR / "detector.pt"))
    p.add_argument("--limit", type=int, default=0,
                   help="random images per group (0 = all)")
    p.add_argument("--with-training-set", action="store_true",
                   help="also show the SD v1.4 test images the model was built on, "
                        "as a reference bar")
    p.add_argument("--chart", help="save a bar chart here (PNG)")
    p.add_argument("--title", default="How often the detector catches each generator")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not args.fake and not args.real:
        sys.exit("Give --fake and/or --real")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    model.eval()
    transform = get_transforms(ckpt["img_size"], train=False)
    print(f"Model: {args.checkpoint} ({ckpt['model_name']})\n")

    jobs = []   # (label, "AI" or "real", paths)
    if args.with_training_set:
        # the real-photo reference only makes sense next to other real photos
        ref = [("fake", "AI")] + ([("real", "real")] if args.real else [])
        for cls, kind in ref:
            paths = sorted(str(f) for f in (DATA_DIR / cls).glob("test__*.jpg"))
            if paths:
                jobs.append(("Stable Diffusion 1.4 (trained on)", kind, paths))
    for folder, kind in ((args.fake, "AI"), (args.real, "real")):
        if folder:
            groups = find_groups(folder)
            if not groups:
                sys.exit(f"No images found in {folder}")
            jobs += [(name, kind, paths) for name, paths in groups.items()]

    rng = random.Random(0)
    rows = []
    print(f"  {'group':40s} {'images':>7s}  result")
    for label, kind, paths in jobs:
        if args.limit and len(paths) > args.limit:
            paths = rng.sample(paths, args.limit)
        probs = score(model, paths, transform, device, args.batch_size, args.workers)
        if len(probs) == 0:
            print(f"  {label:40s} {0:7d}  (no readable images)")
            continue
        if kind == "AI":
            value = (probs > 0.5).float().mean().item()
            desc = f"fakes caught {value:7.2%}"
        else:
            value = (probs <= 0.5).float().mean().item()
            desc = f"real correct {value:7.2%}"
        print(f"  {label:40s} {len(probs):7d}  {desc}   (avg AI score {probs.mean():.0%})",
              flush=True)
        rows.append((label, kind, value, len(probs)))

    if args.chart and rows:
        save_chart(rows, args.chart, args.title)
        print(f"\nChart saved to {args.chart}")


if __name__ == "__main__":   # required on Windows when loading data in parallel
    main()

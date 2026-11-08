"""
prepare_data.py - turn GenImage folders into a clean training set.

GenImage stores each generator like this:
    <generator folder>/train/ai/       AI images   (PNG, 512x512)
    <generator folder>/train/nature/   real photos (mostly JPEG, mixed sizes)
    <generator folder>/val/ai/  and  val/nature/   held-out images

This script:
  1. takes a random sample of each folder (you don't need all 160k images),
  2. resizes and centre-crops every image to 256x256,
  3. re-saves every image as JPEG at the same quality,
  4. writes them into src/data/real and src/data/fake.

Steps 2 and 3 matter. Without them, the real photos are JPEGs of mixed sizes
and the AI images are 512x512 PNGs, and a model can score well just by
spotting "JPEG = real" - without learning anything about AI images.

Splits:
  train  - what the model learns from          (from GenImage's train/)
  val    - checked after every epoch            (also from train/, kept apart)
  test   - used once at the end for the score   (from GenImage's val/)

Run from the project folder:
    venv\\Scripts\\python src\\training\\prepare_data.py --src C:\\Users\\natal\\datasets\\genimage

--src can be the folder of one generator, or a folder holding several -
every generator found inside is used. Add --clean to replace images from an
earlier run.
"""

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from common import CLASSES, DATA_DIR, PREP_SIZE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# GenImage folder name -> our class name
GENIMAGE_CLASSES = {"ai": "fake", "nature": "real"}


def find_generator_dirs(root, max_depth=3):
    """Find every folder that contains train/ai or val/ai (one per generator).

    Stops going deeper once it finds one, so it never has to list the
    hundreds of thousands of image files themselves.
    """
    found = []

    def walk(folder, depth):
        if (folder / "train" / "ai").is_dir() or (folder / "val" / "ai").is_dir():
            found.append(folder)
            return
        if depth < max_depth:
            for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
                walk(sub, depth + 1)

    walk(Path(root), 0)
    return found


def list_images(folder):
    if not folder.is_dir():
        return []
    return sorted(e.path for e in os.scandir(folder)
                  if os.path.splitext(e.name)[1].lower() in IMAGE_EXTS)


def process_one(job):
    """Open one image, resize + crop it, save as JPEG. Runs in parallel."""
    src, dst, size, quality = job
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            # ImageOps.fit = scale so the shorter side is `size`, then cut the
            # centre square. Every image ends up exactly size x size.
            img = ImageOps.fit(img, (size, size), Image.Resampling.BICUBIC)
            img.save(dst, "JPEG", quality=quality)
        return None
    except Exception as e:          # corrupt/unreadable file: skip it
        return f"{src}: {e}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", nargs="+", required=True,
                   help="GenImage folder(s) to read from")
    p.add_argument("--out", default=str(DATA_DIR),
                   help="where to write real/ and fake/ (default: src/data)")
    p.add_argument("--train-per-class", type=int, default=10000,
                   help="images per class per generator for train+val (default 10000)")
    p.add_argument("--test-per-class", type=int, default=1000,
                   help="images per class per generator for test (default 1000)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="share of the train sample kept for validation (default 0.1)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed: same seed = same sample every time")
    p.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 2) - 1),
                   help="images processed in parallel")
    p.add_argument("--clean", action="store_true",
                   help="delete images from a previous run first")
    args = p.parse_args()

    out = Path(args.out)
    rng = random.Random(args.seed)

    # --- Don't silently mix old and new images -----------------------------
    for class_name in CLASSES:
        folder = out / class_name
        folder.mkdir(parents=True, exist_ok=True)
        old = [f for f in folder.iterdir() if f.suffix == ".jpg"]
        if old and not args.clean:
            sys.exit(f"{folder} already has {len(old)} images. "
                     "Re-run with --clean to replace them.")
        for f in old:
            f.unlink()

    # --- Find generators ----------------------------------------------------
    gen_dirs = []
    for src in args.src:
        found = find_generator_dirs(src)
        if not found:
            sys.exit(f"No GenImage folders (with train/ai or val/ai) found in {src}")
        gen_dirs += found
    print("Generators found:")
    for g in gen_dirs:
        print(f"  {g.name}   ({g})")

    # --- Decide which images go where ---------------------------------------
    jobs = []
    summary = []
    for gen_dir in gen_dirs:
        gen = gen_dir.name.replace("__", "_")   # "__" separates parts of filenames
        for gi_name, class_name in GENIMAGE_CLASSES.items():
            # train + val come from GenImage's train/ folder
            pool = list_images(gen_dir / "train" / gi_name)
            if len(pool) < args.train_per_class:
                print(f"  note: only {len(pool)} images in {gen}/train/{gi_name}, using all")
            picked = rng.sample(pool, min(args.train_per_class, len(pool)))
            n_val = round(len(picked) * args.val_frac)
            splits = [("val", picked[:n_val]), ("train", picked[n_val:])]

            # test comes from GenImage's val/ folder (never seen in training)
            pool = list_images(gen_dir / "val" / gi_name)
            if len(pool) < args.test_per_class:
                print(f"  note: only {len(pool)} images in {gen}/val/{gi_name}, using all")
            splits.append(("test", rng.sample(pool, min(args.test_per_class, len(pool)))))

            for split, files in splits:
                summary.append((gen, class_name, split, len(files)))
                for i, f in enumerate(files):
                    dst = out / class_name / f"{split}__{gen}__{i:06d}.jpg"
                    jobs.append((f, str(dst), PREP_SIZE, args.quality))

    print("\nPlan:")
    for gen, class_name, split, n in summary:
        print(f"  {gen:30s} {class_name:5s} {split:5s} {n:6d}")
    print(f"\nProcessing {len(jobs)} images with {args.workers} workers...")

    # --- Process in parallel ------------------------------------------------
    errors = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for done, err in enumerate(pool.map(process_one, jobs, chunksize=64), 1):
            if err:
                errors.append(err)
            if done % 1000 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    print(f"\nDone. {len(jobs) - len(errors)} images written to {out}")
    if errors:
        print(f"{len(errors)} images could not be read and were skipped, e.g.:")
        for e in errors[:5]:
            print("  ", e)


if __name__ == "__main__":   # required on Windows for parallel processing
    main()

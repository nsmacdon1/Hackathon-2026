"""
eval.py - measure how well a trained model does on images it has never seen.

Uses the "test" images from prepare_data.py: they come from GenImage's val/
folder, so the model saw none of them during training. That makes this the
honest score - the one to report.

Run from the project folder:
    venv\\Scripts\\python src\\training\\eval.py src\\models\\<run>\\best.pt
"""

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from common import DATA_DIR, RealFakeDataset, get_transforms, load_checkpoint


@torch.no_grad()   # no training here, so skip the extra bookkeeping
def predict(model, loader, device):
    """Returns the model's fake-probability for every image, in order."""
    model.eval()
    probs = []
    for x, _ in loader:
        logits = model(x.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu())   # score -> 0..1
    return torch.cat(probs)


def pct(a, b):
    return f"{a / b:7.2%}" if b else "    n/a"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a .pt file saved by train.py")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=0.5,
                   help="fake-probability above which an image is called fake")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Model: {ckpt['model_name']}, epoch {ckpt['epoch']}, "
          f"val acc {ckpt['val_acc']:.2%}")

    ds = RealFakeDataset(args.data_dir, args.split,
                         get_transforms(ckpt["img_size"], train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    print(f"Evaluating on {len(ds)} '{args.split}' images...\n")

    probs = predict(model, loader, device)
    labels = torch.tensor([label for _, label, _ in ds.items])
    preds = (probs > args.threshold).long()

    # --- The four possible outcomes -----------------------------------------
    real_ok = ((labels == 0) & (preds == 0)).sum().item()   # real, called real
    real_bad = ((labels == 0) & (preds == 1)).sum().item()  # real, called fake (false alarm)
    fake_ok = ((labels == 1) & (preds == 1)).sum().item()   # fake, caught
    fake_bad = ((labels == 1) & (preds == 0)).sum().item()  # fake, missed
    n_real, n_fake = real_ok + real_bad, fake_ok + fake_bad

    print(f"Overall accuracy: {pct(real_ok + fake_ok, len(ds))}\n")
    print("                    called real   called fake")
    print(f"  real images  {real_ok:13d} {real_bad:13d}    correct {pct(real_ok, n_real)}")
    print(f"  fake images  {fake_bad:13d} {fake_ok:13d}    caught  {pct(fake_ok, n_fake)}")
    print()
    print(f"  False alarms (real called fake): {pct(real_bad, n_real)}")
    print(f"  Missed fakes (fake called real): {pct(fake_bad, n_fake)}")

    # --- Per generator: where does the model struggle? ----------------------
    per_gen = defaultdict(lambda: [0, 0, 0, 0])  # real ok, real total, fake ok, fake total
    for (_, label, gen), pred in zip(ds.items, preds.tolist()):
        if label == 0:
            per_gen[gen][0] += pred == 0
            per_gen[gen][1] += 1
        else:
            per_gen[gen][2] += pred == 1
            per_gen[gen][3] += 1
    if len(per_gen) > 1:
        print("\nPer generator:")
        print(f"  {'generator':30s} {'real correct':>13s} {'fakes caught':>13s}")
        for gen, (ro, rt, fo, ft) in sorted(per_gen.items()):
            print(f"  {gen:30s} {pct(ro, rt):>13s} {pct(fo, ft):>13s}")


if __name__ == "__main__":   # required on Windows when loading data in parallel
    main()

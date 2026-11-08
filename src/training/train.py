"""
train.py - fine-tune a pretrained model to tell real images from AI images.

Needs the images made by prepare_data.py. Each run saves into its own folder,
src/models/<date-time>_<model>/, so a worse run never overwrites a better one:
    best.pt   the epoch with the highest validation accuracy  <- use this one
    last.pt   the most recent epoch
    log.txt   the results of every epoch

Run from the project folder:
    venv\\Scripts\\python src\\training\\train.py
    venv\\Scripts\\python src\\training\\train.py --epochs 10 --model resnet18

Continue training a saved model (e.g. on new data):
    venv\\Scripts\\python src\\training\\train.py --resume src\\models\\<run>\\best.pt
"""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import (DATA_DIR, MODELS_DIR, RealFakeDataset, build_model,
                    get_transforms, load_checkpoint)

# ---------------------------------------------------------------------------
# Settings - the defaults. Each can also be changed from the command line.
# ---------------------------------------------------------------------------

MODEL_NAME = "resnet50"   # or resnet18 (faster), efficientnet_b0, ...
EPOCHS = 5                # passes through the whole training set
BATCH_SIZE = 32           # images per step. Lower to 16 if "CUDA out of memory"
LEARNING_RATE = 1e-4      # how big each adjustment is. Small = careful
IMG_SIZE = 224            # size of the crop the model sees
NUM_WORKERS = 4           # processes loading images while the GPU works
# "Mixed precision" speeds up newer GPUs (RTX cards, which have tensor cores),
# but measured on the GTX 1660 Ti it was over 2x SLOWER - so it's off.
# Turn it on with --amp if you train on an RTX card or Colab.
USE_AMP = False


def run_epoch(model, loader, device, loss_fn, optimizer=None, scaler=None,
              use_amp=False, label=""):
    """One pass over a dataset. Trains if an optimizer is given, otherwise
    only measures. Returns (average loss, accuracy)."""
    training = optimizer is not None
    model.train(training)   # switches dropout/batch-norm to the right mode
    total_loss, correct, seen = 0.0, 0, 0
    start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(training), \
             torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x).squeeze(1)    # one raw score per image
            loss = loss_fn(logits, y)       # how wrong the scores are

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()   # work out how to adjust each weight
            scaler.step(optimizer)          # ...and adjust them
            scaler.update()

        # score > 0 means probability > 50% -> the model says "fake"
        preds = (logits > 0).float()
        correct += (preds == y).sum().item()
        total_loss += loss.item() * len(y)
        seen += len(y)

        if step % 50 == 0 or step == len(loader):
            speed = seen / (time.time() - start)
            print(f"  {label} {step}/{len(loader)}  loss {total_loss / seen:.4f}  "
                  f"acc {correct / seen:.2%}  ({speed:.0f} img/s)", flush=True)

    return total_loss / seen, correct / seen


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--amp", action="store_true",
                   help="turn on mixed precision (faster on RTX cards, slower on GTX)")
    p.add_argument("--resume", help="a saved .pt file to continue training from")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    args = p.parse_args()

    # --- GPU check ----------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        # Let PyTorch try a few ways of running each layer and keep the
        # fastest for this GPU (all images are the same size, so this pays off)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("WARNING: no GPU found - training on the CPU will be very slow.")
    use_amp = (USE_AMP or args.amp) and device.type == "cuda"

    # --- Data ---------------------------------------------------------------
    train_ds = RealFakeDataset(args.data_dir, "train", get_transforms(IMG_SIZE, train=True))
    val_ds = RealFakeDataset(args.data_dir, "val", get_transforms(IMG_SIZE, train=False))
    print(f"Train: {len(train_ds)} images ({train_ds.count('real')} real, "
          f"{train_ds.count('fake')} fake)   Val: {len(val_ds)} images")

    loader_opts = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda",
                       persistent_workers=args.workers > 0)
    # shuffle=True: a new random order each epoch, so the model can't learn
    # anything from the order the images arrive in
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_opts)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_opts)

    # --- Model --------------------------------------------------------------
    if args.resume:
        model, old = load_checkpoint(args.resume, device)
        model_name = old["model_name"]
        print(f"Continuing from {args.resume} ({model_name}, "
              f"val acc {old['val_acc']:.2%})")
    else:
        model_name = args.model
        print(f"Loading pretrained {model_name} (downloads on first use)...")
        model = build_model(model_name).to(device)

    # Loss for yes/no problems: compares the score to the label (0 or 1)
    loss_fn = nn.BCEWithLogitsLoss()
    # AdamW adjusts every weight in the model, a little each step
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Slowly lower the learning rate so the last epochs make fine adjustments
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # --- Where to save ------------------------------------------------------
    run_dir = Path(args.models_dir) / f"{datetime.now():%Y%m%d-%H%M%S}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "log.txt", "w", encoding="utf-8")
    log.write(f"model={model_name} epochs={args.epochs} batch={args.batch_size} "
              f"lr={args.lr} amp={use_amp} resume={args.resume}\n"
              f"train={len(train_ds)} val={len(val_ds)}\n")
    print(f"Saving to {run_dir}\n")

    # --- Training loop ------------------------------------------------------
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, loss_fn,
                                          optimizer, scaler, use_amp, "train")
        val_loss, val_acc = run_epoch(model, val_loader, device, loss_fn,
                                      use_amp=use_amp, label="val  ")
        scheduler.step()

        if math.isnan(train_loss):
            print("Loss became NaN (not a number). Try a lower --lr"
                  + (", or run without --amp." if use_amp else "."))
            break

        minutes = (time.time() - start) / 60
        line = (f"epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.2%} | "
                f"val loss {val_loss:.4f} acc {val_acc:.2%} | {minutes:.1f} min")
        print(line)
        log.write(line + "\n")
        log.flush()

        ckpt = {"model_name": model_name, "img_size": IMG_SIZE,
                "state_dict": model.state_dict(), "epoch": epoch,
                "val_acc": val_acc}
        torch.save(ckpt, run_dir / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  new best - saved best.pt")
        print()

    log.close()
    print(f"Finished. Best validation accuracy: {best_acc:.2%}")
    print("Test it with:")
    print(f"  venv\\Scripts\\python src\\training\\eval.py {run_dir / 'best.pt'}")


if __name__ == "__main__":   # required on Windows when loading data in parallel
    main()

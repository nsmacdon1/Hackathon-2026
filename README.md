# Real or Baked

Real or Baked is a web app that estimates whether an image was taken with a
camera or made by an AI image generator. It checks the file's metadata first,
then runs a trained neural network on the pixels, and combines both into one
verdict: **Real**, **Unsure** or **Baked** (AI-generated), with the reasons
behind it.

## How it works

| Layer | File | What it does |
|---|---|---|
| 1. Provenance | `src/detector/provenance.py` | Reads C2PA Content Credentials, EXIF camera data and PNG text chunks where tools like Stable Diffusion store prompts. |
| 2. Pixel classifier | `src/classifier/model.py` | A ResNet-50 fine-tuned to tell real photos from AI images, using the shared model in `src/models/detector.pt`. |
| 3. Frequency analysis | `src/classifier/frequency.py` | Planned, not implemented. |
| Aggregator | `src/aggregate/pipeline.py` | Combines the layers into one score and verdict. |

How the aggregator decides:

- **AI evidence in the metadata decides on its own.** A C2PA manifest marking
  the image as AI-generated, a generator prompt stored in a PNG, or an AI tool
  named in the EXIF software field makes the verdict Baked.
- **Camera metadata pulls the score toward real.** Full camera data (make,
  model and exposure settings) counts for 70% of the score, partial data for
  35%. With full camera data, the pixel model alone can push the verdict no
  further than Unsure. EXIF is easy to fake, so it never proves an image is
  real by itself.
- **Otherwise the pixel classifier decides.** A score of 70% or more is Baked,
  30% or less is Real, and anything in between is Unsure.

## Results

All numbers are for the model in `src/models/detector.pt`, trained on the
Stable Diffusion 1.4 part of GenImage.

**Held-out test set** (2,000 GenImage images the model never trained on):
98.3% accuracy.

**Robustness.** The first version scored 99% on clean images but caught only
12% of AI images after JPEG compression at quality 50: it had learned that
"compressed" means "real". Retraining with random compression, resizing and
blur fixed this:

| Change applied to the test images | AI images caught, before fix | After fix |
|---|---|---|
| None | 99% | 99% |
| JPEG quality 75 | 47% | 98% |
| JPEG quality 50 | 12% | 97% |
| Shrunk to 50% and back | 51% | 98% |
| Screenshot (resized to 90%) | 92% | 99% |

**Other generators** (Synthbuster, a separate dataset not used in training,
500 images per generator): Stable Diffusion 1.4 83%, Midjourney v5 27%,
GLIDE 20% of AI images caught. The model mostly recognizes the generator it
was trained on. A test model that also trained on GenImage's GLIDE and
Midjourney images raised GLIDE to 66%, but Midjourney stayed at 28% with only
327 Midjourney training images. Charts are in `results/`.

## Limitations

- Accuracy drops sharply on generators the model has not seen, such as
  Midjourney, DALL-E and newer models.
- The classifier was trained on photographs. Diagrams, logos and screenshots
  of text are outside what it learned, and its answers on them are unreliable.
- Modern high-resolution phone photos differ from the ImageNet photos it
  learned from, which can cause false alarms. Camera metadata reduces this.
- Metadata can be removed or faked. Screenshots and most social media
  uploads remove it.

## Setup

Requirements: Python 3.11 or newer. An NVIDIA GPU is needed for training; the
app also runs on a CPU, more slowly.

1. Clone the repository:

```bash
git clone https://github.com/nsmacdon1/Hackathon-2026.git
cd Hackathon-2026
```

2. Create and activate a virtual environment.

On macOS and Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
```

If PowerShell says running scripts is disabled, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

3. Install PyTorch. For GPU support, pick Stable, your OS, Pip, Python and a
CUDA version at https://pytorch.org/get-started/locally/ and run the command
it shows, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu134
```

Check that the GPU is found (it should print `True` and your card's name):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

4. Install the other dependencies:

```bash
pip install -r requirements.txt
```

## Running the app

From the project folder:

```bash
python src/main.py
```

Then open http://127.0.0.1:7860 in a browser.

The app loads `src/models/detector.pt`. To try a different model, set the
`REALBAKED_MODEL` environment variable to its path.

## Training your own model

The scripts in `src/training/` train and test the pixel classifier on
[GenImage](https://github.com/GenImage-Dataset/GenImage). Download at least
one generator's folder (for example Stable Diffusion V1.4) and extract it
outside the project. Images and trained models are ignored by git, except
`src/models/detector.pt`.

```bash
python src/training/prepare_data.py --src C:\path\to\genimage
python src/training/train.py
python src/training/eval.py src/models/<run folder>/best.pt
```

| Script | Purpose |
|---|---|
| `prepare_data.py` | Samples images, re-saves them as 256x256 JPEGs so file format can't give the answer away, and splits them into train, validation and test sets in `src/data/`. |
| `train.py` | Fine-tunes a pretrained ResNet-50. Each run is saved to its own folder in `src/models/`. Use `--resume` to continue from a saved model. |
| `eval.py` | Accuracy on the held-out test images, per generator. |
| `robustness.py` | Re-tests accuracy after JPEG compression, resizing, blur and screenshots. |
| `test_folder.py` | Tests any folder of images (one score per subfolder) and can draw a bar chart. |

Run any script with `--help` to see its options. To share a better model with
the team, copy its `best.pt` over `src/models/detector.pt` and commit it.

## Project structure

```
src/
  main.py              web app (Gradio)
  detector/            Layer 1: provenance and metadata
  classifier/          Layer 2: pixel classifier
  aggregate/           combines the layers into a verdict
  training/            data preparation, training and evaluation scripts
  models/detector.pt   the trained model the app uses
results/               evaluation charts
```

## Datasets and licenses

- [GenImage](https://github.com/GenImage-Dataset/GenImage): training data.
  Licensed CC BY-NC-SA 4.0 (non-commercial).
- [Synthbuster](https://zenodo.org/records/10066460): evaluation on other
  generators. Licensed CC BY-NC-SA 4.0.

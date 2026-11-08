# Real or Baked

AI image detection project.

## Requirements

- Python 3.8 or higher
- pip

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/nsmacdon1/Hackathon-2026.git
cd Hackathon-2026
```

2. Create and activate a virtual environment:

On macOS and Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

When packages are added to requirements.txt, install them with:

```bash
pip install -r requirements.txt
```

4. Install PyTorch with GPU support (NVIDIA cards). It isn't in
requirements.txt because the right command depends on your CUDA version:
pick Stable / Windows / Pip / Python / CUDA on https://pytorch.org/get-started/locally/
and run the command it shows, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu134
```

Check that the GPU is found (should print `True` and your card's name):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Training the classifier

The model is trained on [GenImage](https://github.com/GenImage-Dataset/GenImage).
Download at least one generator's folder (e.g. Stable Diffusion V1.4) and
extract it **outside** the project, e.g. `C:\Users\<you>\datasets\genimage`.
Images and trained models are ignored by git.

```bash
python src/training/prepare_data.py --src C:\Users\<you>\datasets\genimage
python src/training/train.py
python src/training/eval.py src/models/<run folder>/best.pt
```

- `prepare_data.py` samples images, re-saves them all as 256x256 JPEGs and
  writes them to `src/data/real` and `src/data/fake`.
- `train.py` fine-tunes a pretrained ResNet-50 and saves each run to its own
  folder in `src/models/`. Run with `--help` to see the options.
- `eval.py` reports accuracy on the held-out test images.

## How to Run

Run the main application script:

```bash
python main.py
```

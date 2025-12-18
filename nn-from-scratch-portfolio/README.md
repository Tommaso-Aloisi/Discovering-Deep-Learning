# Neural Network from Scratch (NumPy)

A small **portfolio project** showing how to build and train a neural network *without* deep learning frameworks.

## What’s inside
- `src/nn_scratch.py`: minimal “Torch-like” API (layers, activations, losses, SGD + momentum)
- `notebooks/nn_from_scratch.ipynb`: demo notebook (training curves + accuracy)
- `data/`: optional local dataset (not committed by default)

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/nn_from_scratch.ipynb`.

### Dataset
- If `data/mini_mnist.npz` exists, the notebook uses it.
- Otherwise it falls back to `sklearn.datasets.load_digits`.

## Why this repo exists
I wanted a clean, reviewer-friendly repo I can link in a CV:
it demonstrates fundamentals (backprop, losses, optimization) with readable code and reproducible results.

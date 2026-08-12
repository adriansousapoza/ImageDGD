# ImageDGD

Deep Generative Decoder (DGD) for image generation on FashionMNIST: a decoder network paired with per-sample latent representations optimized directly (no encoder), regularized by a Gaussian Mixture Model prior fit with [`tgmm`](https://adriansousapoza.github.io/tgmm/).

## Installation

GPU acceleration (cuDF/cuML via RAPIDS) is optional but recommended — it speeds up the PCA/UMAP latent-space visualizations significantly. Without it, everything still runs on CPU/scikit-learn, just slower. Uses a local [uv](https://github.com/astral-sh/uv) virtual environment (`.venv/`, gitignored) rather than a global one.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't already installed

uv venv .venv --python 3.12
uv pip install --python .venv \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu13==26.4.*" "cuml-cu13==26.4.*" \
    -r requirements.txt

source .venv/bin/activate
```

`cudf-cu13`/`cuml-cu13` need a matching CUDA 13 toolkit; see the [RAPIDS install guide](https://docs.rapids.ai/install/) for other CUDA versions (swap the `-cu13` suffix, e.g. `-cu12`) or to add other RAPIDS components. Without RAPIDS, just drop that middle `uv pip install` line — `uv venv .venv --python 3.12 && uv pip install --python .venv -r requirements.txt` is enough to run everything on CPU.

Verify the install:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
```

## Usage

Start with `notebooks/dgd_training_demo.ipynb` for a full training walkthrough, then `notebooks/dgd_test_inference.ipynb` for inference on held-out data. `notebooks/run_experiments.ipynb` runs a batch of ablations defined as config overrides in `config/experiment/`.

## Project Structure

- `config/` - Hydra base config (`config.yaml`) and per-experiment overrides (`experiment/`)
- `data/` - Dataset storage (downloaded on first run)
- `experiments/` - Timestamped training/inference run outputs (gitignored)
- `models/` - Saved model checkpoints
- `notebooks/` - Training, inference, and ablation notebooks
- `src/` - Source code
  - `data/` - Data loading utilities
  - `models/` - Model implementations (DGD, GMM via `tgmm`, PCA, representation layer)
  - `training/` - Training loop
  - `visualization/` - Plotting and report generation

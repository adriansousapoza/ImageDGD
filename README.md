# ImageDGD

Deep Generative Decoder (DGD) for image generation on FashionMNIST: a decoder network paired with per-sample latent representations optimized directly (no encoder), regularized by a Gaussian Mixture Model prior fit with [`tgmm`](https://adriansousapoza.github.io/tgmm/).

## Installation

GPU acceleration (cuML/cuDF via RAPIDS) is optional but recommended — it speeds up the PCA/UMAP latent-space visualizations significantly. Without it, everything still runs on CPU/scikit-learn, just slower.

```bash
./install_rapids.sh       # sets up ~/.venvs/rapids_cuda13 with RAPIDS + PyTorch + requirements.txt
source activate_rapids.sh # activate it
```

For manual installation or a different CUDA version, follow the [RAPIDS install guide](https://docs.rapids.ai/install/) to pick the right `cudf-cuXX`/`cuml-cuXX` wheels for your system, then:

```bash
uv pip install torch torchvision   # or the CUDA-matched index for your platform
uv pip install -r requirements.txt # includes tgmm
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

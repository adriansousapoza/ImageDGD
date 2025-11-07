# ImageDGD

Deep Generative Decoder for Image Generation using Gaussian Mixture Models.

## Installation

This project uses RAPIDS libraries for GPU-accelerated computing, including cuML for machine learning operations and the `tgmm` package for Gaussian Mixture Models.

### Prerequisites

- Recommended: NVIDIA GPU with CUDA 13.0 or higher
- [UV](https://github.com/astral-sh/uv) package manager (recommended for fast installation)
- Python 3.10 or higher

### RAPIDS Environment Installation (Recommended)

The easiest way to get started is to use the provided installation script that sets up a global RAPIDS environment:

```bash
# Make the script executable
chmod +x install_rapids.sh

# Run the installation
./install_rapids.sh
```

This will:
- Create a global virtual environment at `~/.venvs/rapids_cuda13`
- Install RAPIDS 25.10 with CUDA 13 support (cuDF, cuML, cuGraph)
- Install PyTorch with CUDA 13.0
- Install all required dependencies including `tgmm`
- Set up a convenient `rapids` alias for activation

### Activating the Environment

After installation, activate the RAPIDS environment using:

```bash
source activate_rapids.sh
```

Or use the alias (if you added it to your shell configuration):

```bash
rapids
```

### Alternative: Manual Installation

If you prefer to install manually, you have two options:

#### Option 1: With RAPIDS (GPU-Accelerated, Recommended)

```bash
# Create a virtual environment
uv venv ~/.venvs/rapids_cuda13

# Activate it
source ~/.venvs/rapids_cuda13/bin/activate

# Install RAPIDS with CUDA 13
uv pip install --index https://pypi.nvidia.com \
    cudf-cu13==25.10.* \
    cuml-cu13==25.10.* \
    cugraph-cu13==25.10.*

# Install PyTorch with CUDA 13
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu130

# Install other dependencies
uv pip install -r requirements.txt

# Install tgmm
uv pip install tgmm
```

#### Option 2: Without RAPIDS (CPU-Only)

```bash
# Create a virtual environment
uv venv ~/.venvs/imagedgd

# Activate it
source ~/.venvs/imagedgd/bin/activate

# Install PyTorch (CPU or CUDA version without RAPIDS)
uv pip install torch torchvision

# Install other dependencies
uv pip install -r requirements.txt

# Install tgmm
uv pip install tgmm
```

**⚠️ Performance Warning:** Without RAPIDS and CUDA 13, PCA, UMAP, and t-SNE visualizations will run on CPU using scikit-learn instead of GPU-accelerated cuML, resulting in **significantly slower** dimensionality reduction computations, especially for large datasets. Training and inference will also be slower without GPU acceleration.


## Usage

See the `notebooks/dgd_training_demo.ipynb` for a complete training example.



## Project Structure

- `config/` - Configuration files for training
- `data/` - Dataset storage
- `figures/` - Dataset storage
- `models/` - Saved model checkpoints
- `notebooks/` - Jupyter notebooks for demos and experiments
- `src/` - Source code
  - `data/` - Data loading utilities
  - `models/` - Model implementations (DGD, GMM, PCA)
  - `training/` - Training loop and utilities
  - `visualization/` - Visualization functions

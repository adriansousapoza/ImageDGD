# ImageDGD

Deep Generative Decoder for Image Generation using Gaussian Mixture Models.

## Installation

### Install from requirements.txt

```bash
pip install -r requirements.txt
```

### Install tgmm (Torch Gaussian Mixture Model)

This project uses the `tgmm` package for Gaussian Mixture Model implementation:

```bash
pip install tgmm
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/adriansousapoza/tgmm.git
```

**Note:** If `tgmm` is not installed, the project will fall back to a local implementation, but using the `tgmm` package is recommended for better performance and updates.

## Usage

See the `notebooks/dgd_training_demo.ipynb` for a complete training example.

# ImageDGD

A clean MLOps implementation of Deep Gaussian Decoder (DGD) for image generation using PyTorch, MLflow, and Optuna.

## Features

- **Deep Gaussian Decoder (DGD)**: Novel approach combining representation learning with Gaussian Mixture Models
- **MLflow Integration**: Complete experiment tracking and model management
- **Optuna Optimization**: Automated hyperparameter tuning
- **Hydra Configuration**: Flexible configuration management
- **Clean MLOps Structure**: Production-ready codebase organization

## Repository Structure

```
ImageDGD/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── data/                  # Data configurations
│   ├── model/                 # Model configurations
│   ├── training/              # Training configurations
│   └── optimization/          # Optimization configurations
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model definitions and utilities
│   ├── training/              # Training logic
│   ├── optimization/          # Hyperparameter optimization
│   └── visualization/         # Plotting and visualization
├── scripts/                   # Executable scripts
│   ├── cli.py                 # Command line interface
│   ├── train.py              # Training script
│   ├── optimize.py           # Optimization script
│   ├── evaluate.py           # Evaluation script
│   └── setup.sh              # Environment setup
├── data/                      # Data directory
├── figures/                   # Generated figures
├── logs/                      # Log files
├── mlruns/                    # MLflow tracking data
├── optuna_studies/            # Optuna study databases
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to repository
cd ImageDGD

# Run setup script
./scripts/setup.sh

# Activate environment
source venv/bin/activate
```

### 2. Basic Training

```bash
# Train with default configuration
python scripts/cli.py train

# Train with custom parameters
python scripts/cli.py train -o training.epochs=100 model.representation.n_features=8
```

### 2.1. Weak GPU Training (NEW!)

For users with limited GPU memory or computational resources:

```bash
# Quick test run (10% data, 50 epochs, optimized settings)
python scripts/cli.py train-weak-gpu

# Ultra-fast test (5% data, 10 epochs, no figure saving)
python scripts/cli.py train-weak-gpu -s 0.05 -e 10 --no-save-figures

# Silent mode for minimal output (20% data, 30 epochs)
python scripts/cli.py train-weak-gpu -s 0.2 -e 30 --no-verbose

# Test your setup
python scripts/test_weak_gpu.py
```

**Weak GPU Features:**
- **Data Subset**: Use `--subset-fraction` to train on a fraction of the data
- **Verbose Control**: Use `--verbose/--no-verbose` for detailed or compact output
- **Optimized Settings**: Smaller models, reduced batch sizes, fewer components
- **Memory Efficient**: Disabled figure saving and reduced plotting frequency

### 3. Hyperparameter Optimization

```bash
# Run optimization
python scripts/cli.py optimize

# Run optimization with custom settings
python scripts/cli.py optimize -o optimization.n_trials=50
```

### 4. Model Evaluation

```bash
# Evaluate trained model
python scripts/cli.py evaluate -m model.pth
```

### 5. Experiment Tracking

```bash
# Launch MLflow UI
python scripts/cli.py mlflow-ui
```

Note: All hyperparameter optimization results are tracked in MLflow. No separate dashboard is needed.

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/config_weak_gpu.yaml`: **NEW!** Weak GPU optimized configuration
- `configs/model/dgd_conv.yaml`: Model architecture
- `configs/model/weak_gpu.yaml`: **NEW!** Smaller model for weak GPUs
- `configs/data/fashion_mnist.yaml`: Dataset configuration  
- `configs/data/weak_gpu.yaml`: **NEW!** Subset data configuration
- `configs/training/default.yaml`: Training parameters
- `configs/training/weak_gpu.yaml`: **NEW!** Reduced training settings
- `configs/optimization/optuna_default.yaml`: Optimization settings

### New Training Parameters

The training function now supports:

```python
# In notebooks or scripts
train_model(
    # ... existing parameters ...
    use_subset=True,    # Indicates subset mode is enabled
    verbose=True        # Controls output verbosity
)
```

- **`use_subset`**: Boolean flag indicating data subset usage (actual subsetting handled by `IndexedDataset`)
- **`verbose`**: If `True`, shows detailed output with all loss components and timing. If `False`, shows compact progress updates.

### Example Configuration Override

```bash
# Change model architecture
python scripts/cli.py train -o model.representation.n_features=10 model.decoder.hidden_dims=[256,128,64]

# Adjust training parameters
python scripts/cli.py train -o training.epochs=300 training.lambda_gmm=2.0

# Use different dataset subset
python scripts/cli.py train -o data.use_subset=true data.subset_fraction=0.2

# Control verbosity and subset usage
python scripts/cli.py train -o verbose=false data.use_subset=true data.subset_fraction=0.05
```

## Model Architecture

The DGD model consists of:

1. **Representation Layer**: Learnable embeddings for each sample
2. **Convolutional Decoder**: Generates images from latent representations
3. **Gaussian Mixture Model**: Provides probabilistic structure in latent space

## MLflow Integration

All experiments are automatically tracked with MLflow:

- **Parameters**: Model and training configurations
- **Metrics**: Training/validation losses, timing information
- **Artifacts**: Model checkpoints, configuration files, figures
- **Tags**: Framework, model type, dataset information

Access the MLflow UI at `http://localhost:5000` to:
- Compare experiments
- Visualize metrics
- Download model artifacts
- Track experiment lineage

## Hyperparameter Optimization

Optuna integration provides:

- **Automated Search**: TPE sampler for efficient parameter exploration
- **Pruning**: Early stopping of unpromising trials
- **Parallel Execution**: Multiple workers for faster optimization
- **Study Management**: Persistent storage of optimization history

### Optimization Search Space

Current search includes:
- Model architecture parameters (latent dimensions, hidden layers)
- Training hyperparameters (learning rates, regularization)
- GMM configuration (number of components)

## Visualization

Comprehensive visualization suite:

- **Training Curves**: Loss progression over epochs
- **Latent Space**: PCA, t-SNE, UMAP projections
- **Reconstructions**: Original vs reconstructed images
- **Generation**: Samples from learned GMM
- **GMM Components**: Visualization of mixture components

## Development

### Adding New Models

1. Implement model in `src/models/`
2. Add configuration in `configs/model/`
3. Update trainer if needed
4. Add tests

### Adding New Datasets

1. Implement data loader in `src/data/`
2. Add configuration in `configs/data/`
3. Update visualization if needed

### Custom Optimizations

1. Define search space in `configs/optimization/`
2. Customize objective function in `src/optimization/`
3. Add custom pruning/sampling strategies

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for complete dependency list.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{imagedgd2025,
  title={ImageDGD: MLOps Implementation of Deep Gaussian Decoder},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ImageDGD}
}
```

## License

MIT License - see LICENSE file for details. 
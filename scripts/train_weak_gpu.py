#!/usr/bin/env python3
"""
Training script optimized for weak GPUs with subset and verbose options.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import create_dataloaders, get_sample_batches
from src.training import DGDTrainer
from src.visualization import plot_images

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_device(config: DictConfig) -> torch.device:
    """Setup compute device."""
    if config.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """Main training function for weak GPUs."""
    
    # Override configurations for weak GPU
    config.data.use_subset = config.get("use_subset", True)
    config.data.subset_fraction = config.get("subset_fraction", 0.1)
    config.training.epochs = config.get("max_epochs", 50)  # Reduce epochs for quick testing
    config.training.logging.plot_interval = config.get("plot_interval", 10)
    config.training.logging.save_figures = config.get("save_figures", False)  # Disable to save time
    
    # Verbose mode
    verbose = config.get("verbose", True)
    
    # Print configuration if verbose
    if verbose:
        print("=== WEAK GPU TRAINING MODE ===")
        print(f"Using subset: {config.data.use_subset} ({config.data.subset_fraction:.1%})")
        print(f"Max epochs: {config.training.epochs}")
        print(f"Verbose mode: {verbose}")
        print(f"Save figures: {config.training.logging.save_figures}")
        print("=" * 30)
    
    # Setup device
    device = setup_device(config)
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # Create data loaders
    if verbose:
        print("Creating data loaders...")
    train_loader, test_loader, class_names = create_dataloaders(config)
    sample_data = get_sample_batches(train_loader, test_loader, device)
    
    # Plot sample images (only in verbose mode and if not disabled)
    if verbose and config.training.logging.save_figures:
        indices_train, images_train, labels_train, indices_test, images_test, labels_test = sample_data
        plot_images(images_train.cpu(), labels_train.cpu(), 'Fashion MNIST Train samples', epoch=None)
        plot_images(images_test.cpu(), labels_test.cpu(), 'Fashion MNIST Test samples', epoch=None)
    
    # Create and run trainer (without MLflow for simplicity)
    if verbose:
        print("Starting training...")
    
    trainer = DGDTrainer(config, device, verbose=verbose)
    results = trainer.train(train_loader, test_loader, sample_data, class_names)
    
    # Save model locally
    model_path = "weak_gpu_model.pth"
    torch.save({
        'model_state_dict': results["model"].state_dict(),
        'rep_state_dict': results["rep"].state_dict(),
        'test_rep_state_dict': results["test_rep"].state_dict(),
        'config': config
    }, model_path)
    
    if verbose:
        print(f"Model saved to {model_path}")
        print("Training completed successfully!")
        print(f"Final train loss: {results['final_train_loss']:.4f}")
        print(f"Final test loss: {results['final_test_loss']:.4f}")
    else:
        print(f"Completed: {results['final_train_loss']:.4f} train, {results['final_test_loss']:.4f} test")


if __name__ == "__main__":
    main()

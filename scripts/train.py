#!/usr/bin/env python3
"""
Main training script for ImageDGD.
"""

import os
import sys
import torch
import mlflow
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
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device


def setup_mlflow(config: DictConfig):
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    # Create or get experiment
    try:
        experiment = mlflow.create_experiment(config.mlflow.experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    
    logger.info(f"Using MLflow experiment: {config.mlflow.experiment_name}")
    return experiment


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """Main training function."""
    
    # Check for verbose mode override
    verbose = config.get("verbose", True)
    
    # Print configuration
    if verbose:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(config))
    
    # Setup device
    device = setup_device(config)
    
    # Setup MLflow
    experiment = setup_mlflow(config)
    
    # Set random seed
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # Create data loaders
    if verbose:
        logger.info("Creating data loaders...")
    train_loader, test_loader, class_names = create_dataloaders(config)
    sample_data = get_sample_batches(train_loader, test_loader, device)
    
    # Plot sample images (only in verbose mode)
    if verbose:
        indices_train, images_train, labels_train, indices_test, images_test, labels_test = sample_data
        plot_images(images_train.cpu(), labels_train.cpu(), 'Fashion MNIST Train samples', epoch=None)
        plot_images(images_test.cpu(), labels_test.cpu(), 'Fashion MNIST Test samples', epoch=None)
    
    # Start MLflow run
    with mlflow.start_run(run_name=config.mlflow.run_name):
        # Log configuration
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        
        # Log tags
        for key, value in config.mlflow.tags.items():
            mlflow.set_tag(key, value)
        
        # Create and run trainer
        if verbose:
            logger.info("Starting training...")
        trainer = DGDTrainer(config, device, verbose=verbose)
        results = trainer.train(train_loader, test_loader, sample_data, class_names)
        
        # Log final results
        mlflow.log_metrics({
            "final_train_loss": results["final_train_loss"],
            "final_test_loss": results["final_test_loss"],
            "total_training_time": results["total_time"]
        })
        
        # Save model artifacts
        model_path = "model"
        torch.save({
            'model_state_dict': results["model"].state_dict(),
            'rep_state_dict': results["rep"].state_dict(),
            'test_rep_state_dict': results["test_rep"].state_dict(),
            'config': config
        }, model_path + ".pth")
        
        mlflow.log_artifact(model_path + ".pth", "models")
        
        if verbose:
            logger.info("Training completed successfully!")
            logger.info(f"Final train loss: {results['final_train_loss']:.4f}")
            logger.info(f"Final test loss: {results['final_test_loss']:.4f}")
        else:
            print(f"Training completed: {results['final_train_loss']:.4f} train, {results['final_test_loss']:.4f} test")


if __name__ == "__main__":
    main()

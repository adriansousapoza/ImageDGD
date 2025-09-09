#!/usr/bin/env python3
"""
Hyperparameter optimization script for ImageDGD using Optuna.
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

from src.optimization import OptunaOptimizer

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
    experiment_name = f"{config.mlflow.experiment_name}_optimization"
    try:
        experiment = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    
    logger.info(f"Using MLflow experiment: {experiment_name}")
    return experiment


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """Main optimization function."""
    
    # Check if optimization is enabled
    if not config.optimization.enabled:
        logger.error("Optimization is disabled in config. Set optimization.enabled=true")
        return
    
    # Print configuration
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
    
    # Create database directory if it doesn't exist
    os.makedirs("optuna_studies", exist_ok=True)
    
    # Start MLflow run for the entire optimization
    with mlflow.start_run(run_name=f"optimization_{config.optimization.study_name}"):
        # Log optimization configuration
        mlflow.log_params({
            "study_name": config.optimization.study_name,
            "n_trials": config.optimization.n_trials,
            "direction": config.optimization.direction,
            "pruner": config.optimization.pruner.type,
            "sampler": config.optimization.sampler.type
        })
        
        # Create and run optimizer
        logger.info("Starting hyperparameter optimization...")
        optimizer = OptunaOptimizer(config, device)
        study = optimizer.optimize()
        
        # Log best results
        mlflow.log_metrics({
            "best_value": study.best_value,
            "n_trials": len(study.trials)
        })
        
        # Log best parameters
        for param_name, value in study.best_params.items():
            mlflow.log_param(f"best_{param_name}", value)
        
        # Get best configuration
        best_config = optimizer.get_best_config(study)
        
        # Save best configuration
        best_config_path = "best_config.yaml"
        with open(best_config_path, 'w') as f:
            OmegaConf.save(best_config, f)
        
        mlflow.log_artifact(best_config_path, "configs")
        
        logger.info("Optimization completed successfully!")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best configuration saved to: {best_config_path}")


if __name__ == "__main__":
    main()

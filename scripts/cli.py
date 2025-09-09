#!/usr/bin/env python3
"""
Command line interface for ImageDGD.
"""

import click
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """ImageDGD: Deep Gaussian Decoder for Image Generation"""
    pass


@cli.command()
@click.option("--config", "-c", default="config", help="Configuration name")
@click.option("--overrides", "-o", multiple=True, help="Configuration overrides")
def train(config, overrides):
    """Train the DGD model."""
    click.echo("Starting training...")
    
    # Build command
    cmd = [
        "python", "scripts/train.py",
        f"--config-name={config}"
    ]
    
    # Add overrides
    for override in overrides:
        cmd.append(override)
    
    # Execute
    os.system(" ".join(cmd))


@cli.command()
@click.option("--subset-fraction", "-s", default=0.1, help="Fraction of data to use (0.0-1.0)")
@click.option("--epochs", "-e", default=50, help="Number of epochs")
@click.option("--verbose/--no-verbose", default=True, help="Verbose output")
@click.option("--save-figures/--no-save-figures", default=False, help="Save training figures")
@click.option("--config", "-c", default="config_weak_gpu", help="Configuration name")
def train_weak_gpu(subset_fraction, epochs, verbose, save_figures, config):
    """Train the DGD model optimized for weak GPUs."""
    click.echo("Starting weak GPU training...")
    click.echo(f"Using {subset_fraction:.1%} of data, {epochs} epochs, verbose={verbose}")
    
    # Build command
    cmd = [
        "python", "scripts/train_weak_gpu.py",
        f"--config-name={config}",
        f"data.subset_fraction={subset_fraction}",
        f"training.epochs={epochs}",
        f"verbose={verbose}",
        f"save_figures={save_figures}"
    ]
    
    # Execute
    os.system(" ".join(cmd))


@cli.command()
@click.option("--config", "-c", default="config", help="Configuration name")
@click.option("--overrides", "-o", multiple=True, help="Configuration overrides")
def optimize(config, overrides):
    """Run hyperparameter optimization."""
    click.echo("Starting hyperparameter optimization...")
    
    # Build command
    cmd = [
        "python", "scripts/optimize.py",
        f"--config-name={config}"
    ]
    
    # Add overrides
    for override in overrides:
        cmd.append(override)
    
    # Execute
    os.system(" ".join(cmd))


@cli.command()
@click.option("--model-path", "-m", required=True, help="Path to trained model")
@click.option("--config", "-c", default="config", help="Configuration name")
@click.option("--overrides", "-o", multiple=True, help="Configuration overrides")
def evaluate(model_path, config, overrides):
    """Evaluate a trained model."""
    click.echo(f"Evaluating model: {model_path}")
    
    # Build command
    cmd = [
        "python", "scripts/evaluate.py",
        f"--config-name={config}",
        f"model_path={model_path}"
    ]
    
    # Add overrides
    for override in overrides:
        cmd.append(override)
    
    # Execute
    os.system(" ".join(cmd))


@cli.command()
@click.option("--port", "-p", default=5000, help="Port for MLflow UI")
def mlflow_ui(port):
    """Launch MLflow UI."""
    click.echo(f"Starting MLflow UI on port {port}...")
    os.system(f"mlflow ui --port {port}")


if __name__ == "__main__":
    cli()

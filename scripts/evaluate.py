#!/usr/bin/env python3
"""
Evaluation script for trained ImageDGD models.
"""

import os
import sys
import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import create_dataloaders, get_sample_batches
from src.models import DGD, ConvDecoder, RepresentationLayer, GaussianMixture
from src.visualization import plot_images, plot_gmm_images, plot_gmm_samples, LatentSpaceVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    model_config = config.model
    
    # Recreate model components
    # This is a simplified version - in practice you'd want to recreate
    # the exact same architecture used during training
    decoder = ConvDecoder(
        latent_dim=model_config.representation.n_features,
        hidden_dims=model_config.decoder.hidden_dims,
        output_channels=model_config.decoder.output_channels,
        output_size=model_config.decoder.output_size,
        use_batch_norm=model_config.decoder.use_batch_norm,
        activation=model_config.decoder.activation,
        final_activation=model_config.decoder.final_activation,
        dropout_rate=model_config.decoder.dropout_rate,
        init_size=model_config.decoder.init_size
    ).to(device)
    
    # Create representation layers (you'd need dataset size info)
    # This is a placeholder - you'd need to store dataset info in checkpoint
    rep = RepresentationLayer(
        dist=model_config.representation.distribution,
        dist_options={
            "n_samples": 60000,  # Fashion-MNIST train size
            "dim": model_config.representation.n_features,
            "radius": model_config.representation.radius,
        },
        device=device
    )
    
    test_rep = RepresentationLayer(
        dist=model_config.representation.distribution,
        dist_options={
            "n_samples": 10000,  # Fashion-MNIST test size
            "dim": model_config.representation.n_features,
            "radius": model_config.representation.radius,
        },
        device=device
    )
    
    # Create GMM
    gmm = GaussianMixture(
        n_features=model_config.representation.n_features,
        n_components=model_config.gmm.n_components,
        covariance_type=model_config.gmm.covariance_type,
        init_params=model_config.gmm.init_params,
        device=device,
        random_state=config.random_seed,
        verbose=model_config.gmm.verbose,
        max_iter=model_config.gmm.max_iter,
        tol=model_config.gmm.tol,
        n_init=model_config.gmm.n_init,
        warm_start=model_config.gmm.warm_start
    )
    
    # Create full model
    model = DGD(decoder, rep, gmm)
    
    # Load state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    rep.load_state_dict(checkpoint['rep_state_dict'])
    test_rep.load_state_dict(checkpoint['test_rep_state_dict'])
    
    model.eval()
    
    return model, rep, test_rep, gmm, config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """Main evaluation function."""
    
    # Get model path from command line or config
    model_path = config.get("model_path", "model.pth")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model, rep, test_rep, gmm, model_config = load_model(model_path, device)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, test_loader, class_names = create_dataloaders(config)
    sample_data = get_sample_batches(train_loader, test_loader, device)
    
    # Evaluate model
    logger.info("Running evaluation...")
    
    with torch.no_grad():
        # Calculate test loss
        test_loss = 0.0
        recon_loss = 0.0
        gmm_loss = 0.0
        
        for i, (index, x, _) in enumerate(test_loader):
            x, index = x.to(device), index.to(device)
            
            z = test_rep(index)
            y = model.decoder(z)
            
            batch_recon_loss = torch.nn.functional.mse_loss(y, x, reduction='sum')
            batch_gmm_loss = -torch.sum(gmm.score_samples(z))
            
            recon_loss += batch_recon_loss.item()
            gmm_loss += batch_gmm_loss.item()
            test_loss += (batch_recon_loss + batch_gmm_loss).item()
        
        # Normalize losses
        test_loss /= len(test_loader.dataset)
        recon_loss /= len(test_loader.dataset)
        gmm_loss /= len(test_loader.dataset)
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Reconstruction Loss: {recon_loss:.4f}")
        logger.info(f"GMM Loss: {gmm_loss:.4f}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # Sample data for visualization
        indices_train, images_train, labels_train, indices_test, images_test, labels_test = sample_data
        
        # Plot original images
        plot_images(images_train.cpu(), labels_train.cpu(), 'Original Train Images', epoch='eval')
        plot_images(images_test.cpu(), labels_test.cpu(), 'Original Test Images', epoch='eval')
        
        # Plot reconstructions
        z_train = rep(indices_train)
        reconstructions_train = model.decoder(z_train)
        plot_images(reconstructions_train.cpu(), labels_train.cpu(), 'Reconstructed Train Images', epoch='eval')
        
        z_test = test_rep(indices_test)
        reconstructions_test = model.decoder(z_test)
        plot_images(reconstructions_test.cpu(), labels_test.cpu(), 'Reconstructed Test Images', epoch='eval')
        
        # Plot GMM samples
        plot_gmm_images(
            model.decoder, gmm, "GMM Component Means",
            epoch='eval', top_n=model_config.model.gmm.n_components, 
            device=device
        )
        plot_gmm_samples(
            model.decoder, gmm, "Generated Images from GMM",
            n_samples=model_config.model.gmm.n_components, 
            epoch='eval', device=device
        )
        
        # Plot latent space
        visualizer = LatentSpaceVisualizer()
        
        z_train_all = rep.z.detach()
        z_test_all = test_rep.z.detach()
        
        # Get all labels
        train_labels = torch.tensor([train_loader.dataset.dataset[i][1] for i in train_loader.dataset.indices])
        test_labels = torch.tensor([test_loader.dataset.dataset[i][1] for i in test_loader.dataset.indices])
        
        # Plot different dimensionality reduction techniques
        visualizer.visualize(
            z_train_all, train_labels, z_test_all, test_labels, gmm,
            method='pca', title="Latent Space - PCA",
            label_names=class_names, epoch='eval'
        )
        
        visualizer.visualize(
            z_train_all, train_labels, z_test_all, test_labels, gmm,
            method='umap', n_neighbors=20, n_components=2, min_dist=0.01,
            title="Latent Space - UMAP",
            random_state=config.random_seed,
            label_names=class_names, epoch='eval'
        )
        
        # Generate random samples
        logger.info("Generating random samples...")
        n_samples = 64
        random_samples = gmm.sample(n_samples)[0]
        generated_images = model.decoder(random_samples)
        
        # Create dummy labels for visualization
        dummy_labels = torch.arange(n_samples) % len(class_names)
        plot_images(generated_images.cpu(), dummy_labels, 'Generated Random Images', epoch='eval')
        
        logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()

"""
Test configuration loading and basic functionality.
"""

import sys
import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import create_dataloaders
from src.models import RepresentationLayer, ConvDecoder, GaussianMixture
from src.training import DGDTrainer


def test_config_loading():
    """Test that configuration loads correctly."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    config = OmegaConf.load(config_path)
    
    assert config.project_name == "ImageDGD"
    assert config.random_seed == 42
    assert "model" in config
    assert "data" in config
    assert "training" in config


def test_representation_layer():
    """Test representation layer creation."""
    dist_options = {
        "n_samples": 100,
        "dim": 5,
        "radius": 0.1,
    }
    
    rep = RepresentationLayer(
        dist="uniform_ball",
        dist_options=dist_options,
        device="cpu"
    )
    
    assert rep.n_rep == 100
    assert rep.dim == 5
    assert rep.z.shape == (100, 5)


def test_decoder_creation():
    """Test convolutional decoder creation."""
    decoder = ConvDecoder(
        latent_dim=5,
        hidden_dims=[128, 64, 32],
        output_channels=1,
        output_size=(28, 28),
        use_batch_norm=True,
        activation='leaky_relu',
        final_activation='sigmoid',
        dropout_rate=0.1
    )
    
    # Test forward pass
    z = torch.randn(32, 5)
    output = decoder(z)
    
    assert output.shape == (32, 1, 28, 28)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # sigmoid output


def test_gmm_creation():
    """Test GMM creation."""
    gmm = GaussianMixture(
        n_features=5,
        n_components=10,
        covariance_type='diag',
        device='cpu'
    )
    
    # Test fitting
    data = torch.randn(1000, 5)
    gmm.fit(data)
    
    assert gmm.weights_.shape == (10,)
    assert gmm.means_.shape == (10, 5)
    assert gmm.fitted_


def test_data_loading():
    """Test data loading functionality."""
    # Create minimal config
    config = OmegaConf.create({
        "data": {
            "dataset_name": "FashionMNIST",
            "root_dir": "./data",
            "download": True,
            "transforms": {
                "train": ["ToTensor"],
                "test": ["ToTensor"]
            },
            "batch_size": 32,
            "shuffle_train": True,
            "shuffle_test": False,
            "num_workers": 0,
            "pin_memory": False,
            "use_subset": True,
            "subset_fraction": 0.01,
            "class_names": [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]
        }
    })
    
    train_loader, test_loader, class_names = create_dataloaders(config)
    
    assert len(class_names) == 10
    assert len(train_loader) > 0
    assert len(test_loader) > 0
    
    # Test batch
    batch = next(iter(train_loader))
    indices, images, labels = batch
    
    assert images.shape[1:] == (1, 28, 28)  # Fashion-MNIST shape
    assert len(indices) == len(images) == len(labels)


if __name__ == "__main__":
    pytest.main([__file__])

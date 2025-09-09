#!/usr/bin/env python3
"""
Verify that the ImageDGD setup is working correctly.
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check that all required imports work."""
    print("Checking imports...")
    
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn
        import umap
        import mlflow
        import optuna
        import hydra
        from omegaconf import DictConfig, OmegaConf
        print("✓ All core dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from src.data import create_dataloaders
        from src.models import RepresentationLayer, ConvDecoder, GaussianMixture
        from src.training import DGDTrainer
        from src.optimization import OptunaOptimizer
        from src.visualization import LatentSpaceVisualizer
        print("✓ All project modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Module import error: {e}")
        return False


def check_device():
    """Check available compute devices."""
    print("\nChecking compute devices...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return True


def check_configs():
    """Check configuration files."""
    print("\nChecking configuration files...")
    
    config_dir = Path(__file__).parent.parent / "configs"
    
    required_configs = [
        "config.yaml",
        "model/dgd_conv.yaml",
        "data/fashion_mnist.yaml",
        "training/default.yaml",
        "optimization/optuna_default.yaml"
    ]
    
    all_exist = True
    for config_file in required_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"✓ {config_file}")
        else:
            print(f"✗ {config_file} - Missing")
            all_exist = False
    
    return all_exist


def check_directories():
    """Check required directories."""
    print("\nChecking directories...")
    
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/optimization",
        "src/visualization",
        "scripts",
        "configs"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - Missing")
            all_exist = False
    
    return all_exist


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test representation layer
        from src.models import RepresentationLayer
        rep = RepresentationLayer(
            dist="uniform_ball",
            dist_options={"n_samples": 10, "dim": 5, "radius": 0.1},
            device="cpu"
        )
        assert rep.z.shape == (10, 5)
        print("✓ RepresentationLayer")
        
        # Test decoder
        from src.models import ConvDecoder
        decoder = ConvDecoder(
            latent_dim=5,
            hidden_dims=[32, 16],
            output_channels=1,
            output_size=(28, 28)
        )
        z = torch.randn(4, 5)
        output = decoder(z)
        assert output.shape == (4, 1, 28, 28)
        print("✓ ConvDecoder")
        
        # Test GMM
        from src.models import GaussianMixture
        gmm = GaussianMixture(n_features=5, n_components=3, device='cpu')
        data = torch.randn(100, 5)
        gmm.fit(data)
        assert gmm.fitted_
        print("✓ GaussianMixture")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("ImageDGD Setup Verification")
    print("=" * 50)
    
    checks = [
        check_imports,
        check_device,
        check_configs,
        check_directories,
        test_basic_functionality
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All checks passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Run training: python scripts/cli.py train")
        print("2. Run optimization: python scripts/cli.py optimize")
        print("3. Launch MLflow UI: python scripts/cli.py mlflow-ui")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
ImageDGD: Deep Gaussian Decoder for Image Generation
"""

from .data import create_dataloaders, get_sample_batches
from .models import RepresentationLayer, DGD, ConvDecoder, GaussianMixture, GMMInitializer
from .training import DGDTrainer
from .visualization import LatentSpaceVisualizer, plot_training_losses, plot_images, plot_gmm_images, plot_gmm_samples

# Optional MLOps components - only import if dependencies are available
try:
    from .optimization import OptunaOptimizer
    __all__ = [
        'create_dataloaders',
        'get_sample_batches', 
        'RepresentationLayer',
        'DGD',
        'ConvDecoder',
        'GaussianMixture',
        'GMMInitializer',
        'DGDTrainer',
        'OptunaOptimizer',
        'LatentSpaceVisualizer',
        'plot_training_losses',
        'plot_images',
        'plot_gmm_images',
        'plot_gmm_samples'
    ]
except ImportError:
    print("Warning: Optuna integration not available. Hyperparameter optimization will not work.")
    __all__ = [
        'create_dataloaders',
        'get_sample_batches', 
        'RepresentationLayer',
        'DGD',
        'ConvDecoder',
        'GaussianMixture',
        'GMMInitializer',
        'DGDTrainer',
        'LatentSpaceVisualizer',
        'plot_training_losses',
        'plot_images',
        'plot_gmm_images',
        'plot_gmm_samples'
    ]

__version__ = "0.1.0"

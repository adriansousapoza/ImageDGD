"""
ImageDGD: Deep Gaussian Decoder for Image Generation
"""

from .data import create_dataloaders, get_sample_batches
from .models import RepresentationLayer, DGD, ConvDecoder
from .training import DGDTrainer
from .visualization import (
    plot_images_by_class,
    plot_generated_samples,
    plot_latent_umap,
    plot_latent_tsne,
    plot_latent_pca,
    plot_latent_comparison,
    extract_latent_codes
)
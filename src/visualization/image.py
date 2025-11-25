"""
Image grid visualization utilities for DGD models.
"""

import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf


def organize_by_class(images: torch.Tensor, labels: torch.Tensor, 
                     n_classes: int = 10, n_per_class: int = 5) -> List[torch.Tensor]:
    """
    Organize images by class in label order (0 to n_classes-1).
    
    Parameters:
    ----------
    images: Tensor of images with shape (N, C, H, W)
    labels: Tensor of labels with shape (N,)
    n_classes: Total number of classes
    n_per_class: Number of samples to collect per class
    
    Returns:
    -------
    List of tensors, one per class, each containing up to n_per_class images
    """
    organized = []
    for class_idx in range(n_classes):
        # Find indices of samples belonging to this class
        class_mask = labels == class_idx
        class_images = images[class_mask]
        
        # Take up to n_per_class samples
        n_samples = min(n_per_class, len(class_images))
        organized.append(class_images[:n_samples])
    
    return organized


def plot_image_grid(images_by_class: List[torch.Tensor], 
                   class_names: List[str],
                   title: str = "Image Grid",
                   n_rows: int = 5,
                   cmap: str = 'viridis',
                   denormalize: bool = True,
                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Plot images organized by class in a grid layout.
    
    Parameters:
    ----------
    images_by_class: List of image tensors, one per class
    class_names: List of class names
    title: Title for the plot
    n_rows: Number of rows in the grid (samples per class)
    cmap: Colormap to use
    denormalize: Whether to denormalize images from [-1,1] to [0,1]
    figsize: Figure size (width, height). If None, auto-calculated
    
    Returns:
    -------
    Matplotlib figure object
    """
    n_cols = len(class_names)
    
    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Handle case where axes is 1D (single row or column)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (class_idx, class_images) in enumerate(zip(range(len(class_names)), images_by_class)):
        class_name = class_names[class_idx]
        
        for row in range(n_rows):
            if row < len(class_images):
                img = class_images[row].cpu().squeeze()
                
                # Denormalize if needed
                if denormalize:
                    img = torch.clamp((img + 1) / 2, 0, 1)
                
                axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
                axes[row, col].axis('off')
            else:
                # If not enough samples, leave blank
                axes[row, col].axis('off')
        
        # Add class name as column title (only once at top)
        axes[0, col].set_title(class_name, fontsize=10, pad=5)
    
    plt.tight_layout()
    return fig


def plot_generated_samples(images: torch.Tensor,
                          labels: Optional[torch.Tensor] = None,
                          title: str = "Generated Samples",
                          n_cols: int = 8,
                          cmap: str = 'gray',
                          denormalize: bool = True,
                          figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Plot generated images in a grid layout.
    
    Parameters:
    ----------
    images: Tensor of images with shape (N, C, H, W)
    labels: Optional labels/component IDs for each image
    title: Title for the plot
    n_cols: Number of columns in the grid
    cmap: Colormap to use
    denormalize: Whether to denormalize images from [-1,1] to [0,1]
    figsize: Figure size (width, height). If None, auto-calculated
    
    Returns:
    -------
    Matplotlib figure object
    """
    n_samples = len(images)
    n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division
    
    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    for i in range(len(axes_flat)):
        if i < n_samples:
            img = images[i].cpu().squeeze()
            
            # Denormalize if needed
            if denormalize:
                img = torch.clamp((img + 1) / 2, 0, 1)
            
            axes_flat[i].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes_flat[i].axis('off')
            
            # Add label if provided
            if labels is not None:
                label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                axes_flat[i].set_title(f'Sample {i+1}\n{label}', fontsize=8)
            else:
                axes_flat[i].set_title(f'Sample {i+1}', fontsize=8)
        else:
            # Hide unused subplots
            axes_flat[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_original_and_reconstructed(images: torch.Tensor,
                                   reconstructions: torch.Tensor,
                                   labels: torch.Tensor,
                                   class_names: List[str],
                                   split_name: str = "Train",
                                   n_per_class: int = 5,
                                   cmap: str = 'viridis',
                                   figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot original images and their reconstructions side by side.
    
    Parameters:
    ----------
    images: Original images tensor (N, C, H, W)
    reconstructions: Reconstructed images tensor (N, C, H, W)
    labels: Labels tensor (N,)
    class_names: List of class names
    split_name: Name of the data split (e.g., "Train", "Test")
    n_per_class: Number of samples per class
    cmap: Colormap to use
    figsize: Figure size. If None, auto-calculated
    
    Returns:
    -------
    Tuple of (original_figure, reconstruction_figure)
    """
    n_classes = len(class_names)
    
    # Organize by class
    originals_by_class = organize_by_class(images, labels, n_classes, n_per_class)
    reconstructions_by_class = organize_by_class(reconstructions, labels, n_classes, n_per_class)
    
    # Plot originals
    fig_orig = plot_image_grid(
        originals_by_class,
        class_names,
        title=f"{split_name}: Original Images by Class (Label Order)",
        n_rows=n_per_class,
        cmap=cmap,
        denormalize=True,
        figsize=figsize
    )
    
    # Plot reconstructions
    fig_recon = plot_image_grid(
        reconstructions_by_class,
        class_names,
        title=f"{split_name}: Reconstructed Images by Class (Label Order)",
        n_rows=n_per_class,
        cmap=cmap,
        denormalize=True,
        figsize=figsize
    )
    
    return fig_orig, fig_recon


def visualize_reconstruction_quality(model,
                                    rep_layer,
                                    test_rep_layer,
                                    sample_data: Tuple,
                                    class_names: List[str],
                                    device: torch.device,
                                    n_per_class: int = 5,
                                    cmap: str = 'viridis') -> dict:
    """
    Comprehensive visualization of reconstruction quality for train and test sets.
    
    Parameters:
    ----------
    model: DGD model
    rep_layer: Training representation layer
    test_rep_layer: Test representation layer
    sample_data: Tuple of (indices_train, images_train, labels_train,
                           indices_test, images_test, labels_test)
    class_names: List of class names
    device: Device to run inference on
    n_per_class: Number of samples per class
    cmap: Colormap to use
    
    Returns:
    -------
    Dictionary containing:
        - train_mse: Training MSE statistics
        - test_mse: Test MSE statistics
        - figures: Dictionary of matplotlib figures
    """
    import torch.nn.functional as F
    
    # Set model to evaluation mode
    model.eval()
    
    # Unpack sample data
    indices_train, images_train, labels_train, indices_test, images_test, labels_test = sample_data
    
    # Move to device
    indices_train = indices_train.to(device)
    images_train = images_train.to(device)
    indices_test = indices_test.to(device)
    images_test = images_test.to(device)
    
    with torch.no_grad():
        # Generate reconstructions
        z_train = rep_layer(indices_train)
        recon_train = model.decoder(z_train)
        
        z_test = test_rep_layer(indices_test)
        recon_test = model.decoder(z_test)
        
        # Compute reconstruction errors
        train_mse = F.mse_loss(recon_train, images_train, reduction='none').mean(dim=[1,2,3])
        test_mse = F.mse_loss(recon_test, images_test, reduction='none').mean(dim=[1,2,3])
    
    # Create visualizations
    train_orig_fig, train_recon_fig = plot_original_and_reconstructed(
        images_train, recon_train, labels_train, class_names,
        split_name="Train", n_per_class=n_per_class, cmap=cmap
    )
    
    test_orig_fig, test_recon_fig = plot_original_and_reconstructed(
        images_test, recon_test, labels_test, class_names,
        split_name="Test", n_per_class=n_per_class, cmap=cmap
    )
    
    return {
        'train_mse': {
            'mean': train_mse.mean().item(),
            'std': train_mse.std().item()
        },
        'test_mse': {
            'mean': test_mse.mean().item(),
            'std': test_mse.std().item()
        },
        'figures': {
            'train_original': train_orig_fig,
            'train_reconstruction': train_recon_fig,
            'test_original': test_orig_fig,
            'test_reconstruction': test_recon_fig
        }
    }


def plot_images_by_class(images: torch.Tensor,
                        labels: torch.Tensor,
                        class_names: List[str],
                        title: str = "Images by Class",
                        n_per_class: int = 5,
                        cmap: str = 'viridis',
                        denormalize: bool = True,
                        figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Universal function to plot images organized by class in a grid.
    Works for any set of images: originals, reconstructions, generations, etc.
    
    Parameters:
    ----------
    images: Tensor of images with shape (N, C, H, W)
    labels: Tensor of labels with shape (N,)
    class_names: List of class names
    title: Title for the plot
    n_per_class: Number of samples per class to display
    cmap: Colormap to use
    denormalize: Whether to denormalize images from [-1,1] to [0,1]
    figsize: Figure size (width, height). If None, auto-calculated
    
    Returns:
    -------
    Matplotlib figure object
    """
    n_classes = len(class_names)
    
    # Organize images by class
    images_by_class = organize_by_class(images, labels, n_classes, n_per_class)
    
    # Plot using the grid function
    fig = plot_image_grid(
        images_by_class,
        class_names,
        title=title,
        n_rows=n_per_class,
        cmap=cmap,
        denormalize=denormalize,
        figsize=figsize
    )
    
    return fig

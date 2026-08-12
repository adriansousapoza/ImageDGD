"""
Image grid visualization utilities for DGD models.
"""

import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import numpy as np


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
                          figsize: Optional[Tuple[int, int]] = None,
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
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

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_ground_truth_and_reconstructions_by_class(
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    title: str = "Ground Truth vs Reconstructed",
    n_per_class: int = 5,
    cmap: str = 'viridis',
    denormalize: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True) -> plt.Figure:
    """
    Plot ground-truth and reconstructed images side by side, per class.

    For each class, produces a GT column immediately followed by a Recon
    column (2 * n_classes columns total). `images` and `reconstructions`
    must be the same length and in the same sample order (e.g. both indexed
    by the same `indices` tensor), so that `labels[i]` identifies the same
    underlying sample in both tensors.

    Parameters:
    ----------
    images: Ground-truth images, shape (N, C, H, W)
    reconstructions: Reconstructed images, shape (N, C, H, W), same order as images
    labels: Tensor of labels with shape (N,)
    class_names: List of class names
    title: Title for the plot
    n_per_class: Number of samples per class to display
    cmap: Colormap to use
    denormalize: Whether to map images from [-1,1] to [0,1] before display.
        Leave False for decoders with a sigmoid final activation, whose
        output (and ToTensor-loaded targets) already live in [0,1].
    figsize: Figure size (width, height). If None, auto-calculated
    save_path: Optional path to save the figure
    show: Whether to display the figure

    Returns:
    -------
    Matplotlib figure object
    """
    n_classes = len(class_names)
    gt_by_class = organize_by_class(images, labels, n_classes, n_per_class)
    recon_by_class = organize_by_class(reconstructions, labels, n_classes, n_per_class)

    n_cols = n_classes * 2
    n_rows = n_per_class

    if figsize is None:
        figsize = (n_cols * 1.2, n_rows * 1.4 + 0.6)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    def _draw_column(col: int, class_images: torch.Tensor):
        for row in range(n_rows):
            ax = axes[row, col]
            if row < len(class_images):
                img = class_images[row].detach().cpu().squeeze()
                if denormalize:
                    img = torch.clamp((img + 1) / 2, 0, 1)
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.axis('off')

    for class_idx in range(n_classes):
        gt_col, recon_col = class_idx * 2, class_idx * 2 + 1
        _draw_column(gt_col, gt_by_class[class_idx])
        _draw_column(recon_col, recon_by_class[class_idx])

        axes[0, gt_col].set_title(f"{class_names[class_idx]}\nGT", fontsize=8)
        axes[0, recon_col].set_title("Recon", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_images_by_class(images: torch.Tensor,
                        labels: torch.Tensor,
                        class_names: List[str],
                        title: str = "Images by Class",
                        n_per_class: int = 5,
                        cmap: str = 'viridis',
                        denormalize: bool = True,
                        figsize: Optional[Tuple[int, int]] = None,
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
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
    save_path: Optional path to save the figure
    show: Whether to display the figure

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

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

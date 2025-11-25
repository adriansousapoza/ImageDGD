"""
Latent space visualization functions for DGD models.

This module provides functions to visualize high-dimensional latent representations
using dimensionality reduction techniques: UMAP, t-SNE, and PCA.

Example usage:
    from src.visualization.latent import plot_latent_comparison, extract_latent_codes
    
    # Extract latent codes from trained model
    latent_codes, labels = extract_latent_codes(model, train_loader, device)
    
    # Compare all methods side-by-side
    fig = plot_latent_comparison(latent_codes, labels, 
                                  methods=('pca', 'tsne', 'umap'),
                                  save_path='figures/latent_comparison.png')
    
    # Or use individual methods
    from src.visualization.latent import plot_latent_pca, plot_latent_tsne, plot_latent_umap
    
    fig_pca = plot_latent_pca(latent_codes, labels, n_components=2)
    fig_tsne = plot_latent_tsne(latent_codes, labels, perplexity=30)
    fig_umap = plot_latent_umap(latent_codes, labels, n_neighbors=15)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import torch


def plot_latent_umap(
    latent_codes: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'UMAP Projection of Latent Space',
    save_path: Optional[str] = None,
    alpha: float = 0.6,
    s: int = 20,
    cmap: str = 'tab10',
    show_legend: bool = True,
    random_state: int = 42
) -> plt.Figure:
    """
    Visualize latent space using UMAP dimensionality reduction.
    
    Args:
        latent_codes: Latent representations, shape (n_samples, latent_dim)
        labels: Optional labels for coloring points, shape (n_samples,)
        n_neighbors: Number of neighbors for UMAP (controls local vs global structure)
        min_dist: Minimum distance between points in embedding
        metric: Distance metric to use
        figsize: Figure size as (width, height)
        title: Plot title
        save_path: Optional path to save the figure
        alpha: Point transparency
        s: Point size
        cmap: Colormap name
        show_legend: Whether to show legend when labels are provided
        random_state: Random seed for reproducibility
        
    Returns:
        matplotlib Figure object
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    # Convert to numpy if torch tensor
    if isinstance(latent_codes, torch.Tensor):
        latent_codes = latent_codes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2
    )
    embedding = reducer.fit_transform(latent_codes)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap=cmap, alpha=alpha, s=s, edgecolors='none'
        )
        if show_legend:
            legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="best")
            ax.add_artist(legend)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            alpha=alpha, s=s, edgecolors='none'
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latent_tsne(
    latent_codes: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    learning_rate: Union[float, str] = 'auto',
    metric: str = 'euclidean',
    figsize: Tuple[int, int] = (10, 8),
    title: str = 't-SNE Projection of Latent Space',
    save_path: Optional[str] = None,
    alpha: float = 0.6,
    s: int = 20,
    cmap: str = 'tab10',
    show_legend: bool = True,
    random_state: int = 42
) -> plt.Figure:
    """
    Visualize latent space using t-SNE dimensionality reduction.
    
    Args:
        latent_codes: Latent representations, shape (n_samples, latent_dim)
        labels: Optional labels for coloring points, shape (n_samples,)
        perplexity: t-SNE perplexity parameter (typical: 5-50)
        n_iter: Number of optimization iterations
        learning_rate: Learning rate for optimization
        metric: Distance metric to use
        figsize: Figure size as (width, height)
        title: Plot title
        save_path: Optional path to save the figure
        alpha: Point transparency
        s: Point size
        cmap: Colormap name
        show_legend: Whether to show legend when labels are provided
        random_state: Random seed for reproducibility
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.manifold import TSNE
    
    # Convert to numpy if torch tensor
    if isinstance(latent_codes, torch.Tensor):
        latent_codes = latent_codes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        metric=metric,
        random_state=random_state
    )
    embedding = tsne.fit_transform(latent_codes)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap=cmap, alpha=alpha, s=s, edgecolors='none'
        )
        if show_legend:
            legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="best")
            ax.add_artist(legend)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            alpha=alpha, s=s, edgecolors='none'
        )
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latent_pca(
    latent_codes: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'PCA Projection of Latent Space',
    save_path: Optional[str] = None,
    alpha: float = 0.6,
    s: int = 20,
    cmap: str = 'tab10',
    show_legend: bool = True,
    show_variance: bool = True
) -> plt.Figure:
    """
    Visualize latent space using PCA dimensionality reduction.
    
    Args:
        latent_codes: Latent representations, shape (n_samples, latent_dim)
        labels: Optional labels for coloring points, shape (n_samples,)
        n_components: Number of principal components (2 or 3)
        figsize: Figure size as (width, height)
        title: Plot title
        save_path: Optional path to save the figure
        alpha: Point transparency
        s: Point size
        cmap: Colormap name
        show_legend: Whether to show legend when labels are provided
        show_variance: Whether to show explained variance in axis labels
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.decomposition import PCA
    
    # Convert to numpy if torch tensor
    if isinstance(latent_codes, torch.Tensor):
        latent_codes = latent_codes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(latent_codes)
    
    # Create plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=labels, cmap=cmap, alpha=alpha, s=s, edgecolors='none'
            )
            if show_legend:
                legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="best")
                ax.add_artist(legend)
            plt.colorbar(scatter, ax=ax, label='Class')
        else:
            ax.scatter(
                embedding[:, 0], embedding[:, 1],
                alpha=alpha, s=s, edgecolors='none'
            )
        
        if show_variance:
            ax.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            ax.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        else:
            ax.set_xlabel('PC 1', fontsize=12)
            ax.set_ylabel('PC 2', fontsize=12)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=labels, cmap=cmap, alpha=alpha, s=s, edgecolors='none'
            )
            if show_legend:
                legend = ax.legend(*scatter.legend_elements(), title="Classes", loc="best")
                ax.add_artist(legend)
            plt.colorbar(scatter, ax=ax, label='Class', pad=0.1)
        else:
            ax.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                alpha=alpha, s=s, edgecolors='none'
            )
        
        if show_variance:
            ax.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            ax.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
            ax.set_zlabel(f'PC 3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=12)
        else:
            ax.set_xlabel('PC 1', fontsize=12)
            ax.set_ylabel('PC 2', fontsize=12)
            ax.set_zlabel('PC 3', fontsize=12)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    else:
        raise ValueError("n_components must be 2 or 3")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latent_comparison(
    latent_codes: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    methods: Tuple[str, ...] = ('pca', 'tsne', 'umap'),
    figsize: Tuple[int, int] = (18, 5),
    title: str = 'Latent Space Visualization',
    save_path: Optional[str] = None,
    alpha: float = 0.6,
    s: int = 20,
    cmap: str = 'tab10',
    show_legend: bool = True,
    random_state: int = 42,
    **kwargs
) -> plt.Figure:
    """
    Create a side-by-side comparison of different dimensionality reduction methods.
    
    Args:
        latent_codes: Latent representations, shape (n_samples, latent_dim)
        labels: Optional labels for coloring points, shape (n_samples,)
        methods: Tuple of methods to compare ('pca', 'tsne', 'umap')
        figsize: Figure size as (width, height)
        title: Overall plot title
        save_path: Optional path to save the figure
        alpha: Point transparency
        s: Point size
        cmap: Colormap name
        show_legend: Whether to show legend when labels are provided
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments passed to specific methods
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Convert to numpy if torch tensor
    if isinstance(latent_codes, torch.Tensor):
        latent_codes = latent_codes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(latent_codes)
            method_title = 'PCA'
            xlabel = f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
            ylabel = f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
            
        elif method.lower() == 'tsne':
            perplexity = kwargs.get('perplexity', 30.0)
            n_iter = kwargs.get('n_iter', 1000)
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=random_state
            )
            embedding = tsne.fit_transform(latent_codes)
            method_title = 't-SNE'
            xlabel = 't-SNE 1'
            ylabel = 't-SNE 2'
            
        elif method.lower() == 'umap':
            try:
                import umap
                n_neighbors = kwargs.get('n_neighbors', 15)
                min_dist = kwargs.get('min_dist', 0.1)
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=random_state,
                    n_components=2
                )
                embedding = reducer.fit_transform(latent_codes)
                method_title = 'UMAP'
                xlabel = 'UMAP 1'
                ylabel = 'UMAP 2'
            except ImportError:
                print(f"Warning: UMAP not installed, skipping. Install with: pip install umap-learn")
                continue
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'tsne', 'umap'")
        
        # Plot
        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=labels, cmap=cmap, alpha=alpha, s=s, edgecolors='none'
            )
            if show_legend and idx == n_methods - 1:
                legend = ax.legend(*scatter.legend_elements(), title="Classes", 
                                 loc="center left", bbox_to_anchor=(1, 0.5))
                ax.add_artist(legend)
        else:
            ax.scatter(
                embedding[:, 0], embedding[:, 1],
                alpha=alpha, s=s, edgecolors='none'
            )
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(method_title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def extract_latent_codes(model, dataloader, device='cuda'):
    """
    Extract latent codes from a trained DGD model.
    
    Args:
        model: Trained DGD model with representation layer
        dataloader: DataLoader to extract latent codes from
        device: Device to run extraction on
        
    Returns:
        Tuple of (latent_codes, labels) as numpy arrays
    """
    model.eval()
    latent_codes = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                data, label = batch[0], batch[1]
            else:
                data = batch
                label = None
            
            data = data.to(device)
            
            # Extract latent codes (assumes model has .representation or .encode method)
            if hasattr(model, 'representation'):
                indices = torch.arange(len(data), device=device)
                z = model.representation(indices)
            elif hasattr(model, 'encode'):
                z = model.encode(data)
            else:
                raise AttributeError("Model must have 'representation' or 'encode' method")
            
            latent_codes.append(z.cpu())
            if label is not None:
                labels_list.append(label.cpu())
    
    latent_codes = torch.cat(latent_codes, dim=0).numpy()
    labels_array = torch.cat(labels_list, dim=0).numpy() if labels_list else None
    
    return latent_codes, labels_array

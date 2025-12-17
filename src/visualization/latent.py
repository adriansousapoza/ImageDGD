"""
Latent space visualization utilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def _try_cuml_import():
    """Try to import cuML components, fall back to sklearn if unavailable."""
    try:
        from cuml import PCA as cuPCA
        from cuml import UMAP as cuUMAP
        from cuml import TSNE as cuTSNE
        return cuPCA, cuUMAP, cuTSNE, True
    except ImportError:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        try:
            from umap import UMAP
        except ImportError:
            UMAP = None
        return PCA, UMAP, TSNE, False


def plot_latent_space(
    representations: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    gmm=None,
    class_names: Optional[list] = None,
    title: str = "Latent Space Visualization",
    figsize: Tuple[int, int] = (20, 6),
    alpha: float = 0.6,
    s: int = 20,
    cmap: str = 'tab10',
    save_path: Optional[str] = None,
    show: bool = True,
    n_components: int = 2,
    random_state: int = 42,
    verbose: bool = False
) -> plt.Figure:
    """
    Visualize latent space using PCA, UMAP, and t-SNE with optional GMM components overlay.
    Uses cuML for GPU acceleration when available.
    
    Parameters:
    ----------
    representations: Latent representations (N x D tensor)
    labels: Optional class labels (N tensor)
    gmm: Optional fitted GaussianMixture model
    class_names: Optional list of class names
    title: Main title for the figure
    figsize: Figure size
    alpha: Point transparency
    s: Point size
    cmap: Colormap for points
    save_path: Optional path to save the figure
    show: Whether to display the figure
    n_components: Number of components for dimensionality reduction (should be 2)
    random_state: Random seed for reproducibility
    verbose: Whether to print diagnostic information
    
    Returns:
    -------
    matplotlib.figure.Figure: The created figure
    """
    # Convert to numpy and move to CPU if needed
    if isinstance(representations, torch.Tensor):
        z = representations.detach().cpu().numpy()
    else:
        z = np.array(representations)
    
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.array(labels)
    
    # Import appropriate libraries (cuML or sklearn)
    PCA, UMAP, TSNE, using_cuml = _try_cuml_import()
    
    if verbose:
        print(f"Using {'cuML (GPU)' if using_cuml else 'sklearn (CPU)'} for dimensionality reduction")
    
    # Prepare data for cuML if available
    if using_cuml:
        import cudf
        z_gpu = cudf.DataFrame(z)
    else:
        z_gpu = z
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. PCA
    if verbose:
        print("Computing PCA...")
    if using_cuml:
        # cuML PCA doesn't accept random_state parameter
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components, random_state=random_state)
    z_pca = pca.fit_transform(z_gpu)
    
    if using_cuml:
        # Convert to numpy if it's a cudf object
        if hasattr(z_pca, 'to_numpy'):
            z_pca = z_pca.to_numpy()
        elif hasattr(z_pca, 'values'):
            z_pca = z_pca.values
        # If already numpy array, no conversion needed
    
    # Compute explained variance
    if hasattr(pca, 'explained_variance_ratio_'):
        var_ratio = pca.explained_variance_ratio_
        if using_cuml:
            # Convert to numpy if it's a cudf object
            if hasattr(var_ratio, 'to_numpy'):
                var_ratio = var_ratio.to_numpy()
            elif hasattr(var_ratio, 'values'):
                var_ratio = var_ratio.values
        var_text = f"({var_ratio[0]*100:.1f}%, {var_ratio[1]*100:.1f}%)"
    else:
        var_text = ""
    
    _plot_2d_projection(
        axes[0], z_pca, labels, class_names,
        f"PCA {var_text}", alpha, s, cmap
    )
    
    # Add GMM components to PCA plot
    if gmm is not None:
        _add_gmm_overlay_pca(axes[0], gmm, pca, using_cuml)
    
    # 2. UMAP
    if UMAP is not None:
        if verbose:
            print("Computing UMAP...")
        try:
            umap = UMAP(n_components=n_components, random_state=random_state, n_neighbors=15, min_dist=0.1)
            z_umap = umap.fit_transform(z_gpu)
            
            if using_cuml:
                # Convert to numpy if it's a cudf object
                if hasattr(z_umap, 'to_numpy'):
                    z_umap = z_umap.to_numpy()
                elif hasattr(z_umap, 'values'):
                    z_umap = z_umap.values
            
            _plot_2d_projection(
                axes[1], z_umap, labels, class_names,
                "UMAP", alpha, s, cmap
            )
        except Exception as e:
            if verbose:
                print(f"UMAP failed: {e}")
            axes[1].text(0.5, 0.5, 'UMAP failed', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title("UMAP")
    else:
        axes[1].text(0.5, 0.5, 'UMAP not available\nInstall: pip install umap-learn', 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("UMAP")
    
    # 3. t-SNE
    if verbose:
        print("Computing t-SNE...")
    try:
        if using_cuml:
            # cuML TSNE: n_neighbors should be at least 3 * perplexity
            # Don't pass random_state to avoid brute_force_knn warning
            tsne = TSNE(n_components=n_components, perplexity=30, n_iter=1000)
        else:
            tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=30, n_iter=1000)
        z_tsne = tsne.fit_transform(z_gpu)
        
        if using_cuml:
            # Convert to numpy if it's a cudf object
            if hasattr(z_tsne, 'to_numpy'):
                z_tsne = z_tsne.to_numpy()
            elif hasattr(z_tsne, 'values'):
                z_tsne = z_tsne.values
        
        _plot_2d_projection(
            axes[2], z_tsne, labels, class_names,
            "t-SNE", alpha, s, cmap
        )
    except Exception as e:
        if verbose:
            print(f"t-SNE failed: {e}")
        axes[2].text(0.5, 0.5, 't-SNE failed', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("t-SNE")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def _plot_2d_projection(
    ax: plt.Axes,
    z_2d: np.ndarray,
    labels: Optional[np.ndarray],
    class_names: Optional[list],
    title: str,
    alpha: float,
    s: int,
    cmap: str
):
    """Helper function to plot 2D projections."""
    if labels is not None:
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, alpha=alpha, s=s, cmap=cmap)
        
        # Add legend if class names provided
        if class_names is not None:
            unique_labels = np.unique(labels)
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=scatter.cmap(scatter.norm(label)), 
                                 markersize=8, label=class_names[int(label)])
                      for label in unique_labels if int(label) < len(class_names)]
            ax.legend(handles=handles, loc='upper right', fontsize=8)
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=alpha, s=s, c='blue')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.3)


def _add_gmm_overlay_pca(ax: plt.Axes, gmm, pca, using_cuml: bool):
    """Add GMM component ellipses to PCA plot."""
    try:
        # Get GMM parameters
        means = gmm.means_.detach().cpu().numpy() if isinstance(gmm.means_, torch.Tensor) else gmm.means_
        covariances = gmm.covariances_.detach().cpu().numpy() if isinstance(gmm.covariances_, torch.Tensor) else gmm.covariances_
        weights = gmm.weights_.detach().cpu().numpy() if isinstance(gmm.weights_, torch.Tensor) else gmm.weights_
        
        # Transform means to PCA space
        if using_cuml:
            import cudf
            means_gpu = cudf.DataFrame(means)
            means_pca = pca.transform(means_gpu)
            # Convert to numpy if needed
            if hasattr(means_pca, 'to_numpy'):
                means_pca = means_pca.to_numpy()
            elif hasattr(means_pca, 'values'):
                means_pca = means_pca.values
        else:
            means_pca = pca.transform(means)
        
        # Get PCA components for covariance transformation
        if using_cuml:
            components = pca.components_
            if hasattr(components, 'to_numpy'):
                components = components.to_numpy()
            elif hasattr(components, 'values'):
                components = components.values
        else:
            components = pca.components_
        
        # Plot each component
        for i, (mean_pca, cov, weight) in enumerate(zip(means_pca, covariances, weights)):
            # Transform covariance to PCA space
            if gmm.covariance_type == 'full':
                cov_pca = components @ cov @ components.T
            elif gmm.covariance_type == 'diag':
                cov_pca = components @ np.diag(cov) @ components.T
            elif gmm.covariance_type == 'spherical':
                # cov is a scalar for spherical covariance
                n_features = components.shape[1]
                cov_pca = components @ (cov * np.eye(n_features)) @ components.T
            else:  # tied
                cov_pca = components @ cov @ components.T
            
            # Extract 2D covariance
            cov_2d = cov_pca[:2, :2]
            
            # Compute eigenvalues and eigenvectors for ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            # Width and height are 2 standard deviations
            width, height = 2 * np.sqrt(eigenvalues)
            
            # Create ellipse (2-sigma contour)
            ellipse = Ellipse(
                mean_pca[:2], width, height, angle=angle,
                facecolor='none', edgecolor='red', linewidth=2, alpha=0.7,
                label=f'GMM {i} (w={weight:.2f})' if i == 0 else f'GMM {i} (w={weight:.2f})'
            )
            ax.add_patch(ellipse)
            
            # Add component center
            ax.scatter(mean_pca[0], mean_pca[1], c='red', s=100, marker='x', linewidths=3, zorder=10)
        
        # Add legend for GMM components
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
    except Exception as e:
        print(f"Warning: Could not add GMM overlay to PCA plot: {e}")

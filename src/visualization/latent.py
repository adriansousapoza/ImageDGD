"""
Latent space visualization utilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
from typing import Optional, Tuple
import warnings
import math
warnings.filterwarnings('ignore')

# Import tgmm plotting helpers
try:
    from tgmm.plotting import get_covariance_matrix, ensure_tensor_on_cpu, create_colormap
    TGMM_PLOTTING_AVAILABLE = True
except ImportError:
    TGMM_PLOTTING_AVAILABLE = False
    warnings.warn("tgmm.plotting helpers not available. Using fallback implementation.")


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
    """Add GMM component ellipses to PCA plot using tgmm helper functions."""
    try:
        # Get GMM parameters using tgmm helpers if available
        if TGMM_PLOTTING_AVAILABLE:
            means = ensure_tensor_on_cpu(gmm.means_, dtype=torch.float32).numpy()
            weights = ensure_tensor_on_cpu(gmm.weights_, dtype=torch.float32).numpy()
        else:
            means = gmm.means_.detach().cpu().numpy() if isinstance(gmm.means_, torch.Tensor) else gmm.means_
            weights = gmm.weights_.detach().cpu().numpy() if isinstance(gmm.weights_, torch.Tensor) else gmm.weights_
        
        n_components = gmm.n_components
        
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
        
        # Get the actual number of features from GMM
        n_features = gmm.n_features
        
        # Get colors for ellipses using tgmm colormap helper
        if TGMM_PLOTTING_AVAILABLE:
            ellipse_colors = create_colormap('turbo', n_components)
        else:
            # Fallback: use matplotlib colormap
            cmap = plt.get_cmap('turbo')
            if n_components == 1:
                ellipse_colors = [cmap(0.5)]
            else:
                ellipse_colors = [cmap(i / (n_components - 1)) for i in range(n_components)]
        
        # Define ellipse parameters (matching tgmm defaults)
        ellipse_std_devs = [2]
        ellipse_alpha = 0.5
        ellipse_fill = True
        ellipse_line_style = 'dotted'
        ellipse_line_width = 2
        ellipse_line_color = 'black'
        ellipse_line_alpha = 1
        
        # Plot each component
        for i in range(n_components):
            mean_pca_i = means_pca[i]
            weight = weights[i]
            
            # Get covariance matrix using tgmm helper if available
            if TGMM_PLOTTING_AVAILABLE:
                cov = get_covariance_matrix(gmm, i).numpy()
            else:
                # Fallback implementation
                covariances = gmm.covariances_.detach().cpu().numpy() if isinstance(gmm.covariances_, torch.Tensor) else gmm.covariances_
                
                # Handle different covariance types
                if gmm.covariance_type == 'full':
                    cov = covariances[i]
                elif gmm.covariance_type == 'diag':
                    cov = np.diag(covariances[i])
                elif gmm.covariance_type == 'spherical':
                    cov = covariances[i] * np.eye(n_features)
                elif gmm.covariance_type == 'tied_full':
                    cov = covariances
                elif gmm.covariance_type == 'tied_diag':
                    cov = np.diag(covariances)
                elif gmm.covariance_type == 'tied_spherical':
                    cov = covariances.item() * np.eye(n_features) if isinstance(covariances, np.ndarray) else covariances * np.eye(n_features)
                else:
                    # Unknown type, skip
                    print(f"Warning: Unknown covariance type {gmm.covariance_type}")
                    continue
            
            # Transform covariance to PCA space
            cov_pca = components @ cov @ components.T
            cov_2d = cov_pca[:2, :2]
            
            # Compute eigenvalues and eigenvectors for ellipse (matching tgmm approach)
            vals, vecs = np.linalg.eigh(cov_2d)
            idx = np.argsort(vals)[::-1]  # Sort descending
            vals, vecs = vals[idx], vecs[:, idx]
            
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            
            # Create ellipses for each standard deviation (matching tgmm)
            for j, std_dev in enumerate(ellipse_std_devs):
                width = 2.0 * std_dev * np.sqrt(vals[0])
                height = 2.0 * std_dev * np.sqrt(vals[1])
                
                # Adjust alpha for multiple ellipses (fade inner ellipses)
                current_alpha = ellipse_alpha * (1 - j * 0.3 / len(ellipse_std_devs))
                
                # Create face color with proper alpha
                if ellipse_fill:
                    face_color_with_alpha = (*to_rgba(ellipse_colors[i])[:3], current_alpha)
                else:
                    face_color_with_alpha = 'none'
                
                # Create edge color with proper alpha
                edge_color_with_alpha = (*to_rgba(ellipse_line_color)[:3], ellipse_line_alpha)
                
                ellipse = Ellipse(
                    (mean_pca_i[0], mean_pca_i[1]),
                    width, height,
                    angle=angle,
                    facecolor=face_color_with_alpha,
                    edgecolor=edge_color_with_alpha,
                    linewidth=ellipse_line_width,
                    linestyle=ellipse_line_style,
                    label=f'Component {i+1} (w={weight:.2f})' if j == 0 else None
                )
                ax.add_patch(ellipse)
            
            # Add component center (matching tgmm style)
            ax.scatter(mean_pca_i[0], mean_pca_i[1], 
                      c='black', marker='h', s=100, 
                      linewidths=2, zorder=10,
                      label='Component Mean' if i == 0 else None)
        
        # Add legend entry for ellipse boundaries (matching tgmm)
        sigma_labels = [f"{std}σ" for std in ellipse_std_devs]
        sigma_text = "[" + ", ".join(sigma_labels) + "]"
        ax.plot([], [], c=ellipse_line_color, linestyle=ellipse_line_style,
               linewidth=ellipse_line_width, alpha=ellipse_line_alpha,
               label=f'{sigma_text}')
        
        # Add legend for GMM components
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
    except Exception as e:
        print(f"Warning: Could not add GMM overlay to PCA plot: {e}")

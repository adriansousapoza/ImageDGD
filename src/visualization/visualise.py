import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
import os
import logging
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass

# Try to use cuML for GPU acceleration, fall back to sklearn if not available
try:
    from cuml import PCA, TSNE
    from cuml.manifold import UMAP
    GPU_AVAILABLE = True
    print("✓ Using cuML (GPU-accelerated) for dimensionality reduction")
except ImportError:
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE
    from umap import UMAP
    GPU_AVAILABLE = False
    print("⚠️ cuML not available, using sklearn (CPU) for dimensionality reduction")

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    figure_size_2_plots: Tuple[int, int] = (18, 8)
    figure_size_3_plots: Tuple[int, int] = (22, 8)
    scatter_point_size: int = 10
    mean_marker_size: int = 100
    grid_alpha: float = 0.3
    scatter_alpha: float = 0.7
    gmm_samples_default: int = 2000
    default_dpi: int = 200
    max_cols_per_row: int = 5
    default_fontsize_title: int = 16
    default_fontsize_subtitle: int = 14
    default_fontsize_labels: int = 12

def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist.
    
    Parameters
    ----------
    directory : str
        Path to the directory to create
        
    Raises
    ------
    OSError
        If directory creation fails
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise


def save_figure(fig: plt.Figure, 
                filename: str, 
                base_dir: str = "figures", 
                subdir: Optional[str] = None, 
                dpi: int = 200,
                format: str = "png", 
                close_fig: bool = True) -> None:
    """
    Save a matplotlib figure to a file with proper error handling.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Base filename without extension
    base_dir : str
        Base directory for saving figures
    subdir : str or None
        Subdirectory within base_dir
    dpi : int
        Resolution for saving the figure
    format : str
        File format (png recommended for plots with many points)
    close_fig : bool
        Whether to close the figure after saving
        
    Raises
    ------
    OSError
        If file saving fails
    """
    try:
        # Create full directory path
        if subdir:
            directory = os.path.join(base_dir, subdir)
        else:
            directory = base_dir
        
        # Ensure directory exists
        ensure_dir(directory)
        
        # Create full file path with extension
        filepath = os.path.join(directory, f"{filename}.{format}")
        
        # Save the figure
        fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
        logger.info(f"Saved figure: {filepath}")
            
        # Close figure to free memory if requested
        if close_fig:
            plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to save figure {filename}: {e}")
        raise


class LatentSpaceVisualizer:
    """
    A unified class for visualizing latent spaces with different dimensionality reduction techniques.
    Provides visualization for PCA, UMAP, Kernel PCA, and t-SNE with GMM integration.
    """
    
    def __init__(self, 
                 config: Optional[VisualizationConfig] = None,
                 custom_colors: Optional[List[str]] = None) -> None:
        """
        Initialize the visualizer with common settings.
        
        Parameters
        ----------
        config : VisualizationConfig, optional
            Configuration object for visualization parameters
        custom_colors : list, optional
            Custom color palette for visualization
        """
        self.config = config or VisualizationConfig()
        
        # Default color palette
        self.custom_colors = custom_colors or [
            '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
            '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'
        ]
        
        logger.debug("Initialized LatentSpaceVisualizer")
    def _validate_inputs(self, 
                        z_train: Union[torch.Tensor, np.ndarray],
                        labels_train: Union[torch.Tensor, np.ndarray],
                        z_test: Union[torch.Tensor, np.ndarray],
                        labels_test: Union[torch.Tensor, np.ndarray]) -> None:
        """Validate input data dimensions and types.
        
        Parameters
        ----------
        z_train, z_test : torch.Tensor or numpy.ndarray
            Latent representations for training and testing data
        labels_train, labels_test : torch.Tensor or numpy.ndarray
            Labels for training and testing data
            
        Raises
        ------
        ValueError
            If data dimensions don't match or data is invalid
        """
        # Check if data is not empty
        if len(z_train) == 0 or len(z_test) == 0:
            raise ValueError("Input data cannot be empty")
            
        # Check dimension consistency
        if len(z_train) != len(labels_train):
            raise ValueError(f"Training data length mismatch: {len(z_train)} vs {len(labels_train)}")
        if len(z_test) != len(labels_test):
            raise ValueError(f"Test data length mismatch: {len(z_test)} vs {len(labels_test)}")
            
        # Check feature dimensions match
        z_train_shape = z_train.shape if hasattr(z_train, 'shape') else np.array(z_train).shape
        z_test_shape = z_test.shape if hasattr(z_test, 'shape') else np.array(z_test).shape
        
        if len(z_train_shape) != 2 or len(z_test_shape) != 2:
            raise ValueError("Input data must be 2D (samples x features)")
            
        if z_train_shape[1] != z_test_shape[1]:
            raise ValueError(f"Feature dimension mismatch: {z_train_shape[1]} vs {z_test_shape[1]}")
    
    def _convert_to_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Efficiently convert tensor to numpy array.
        
        Parameters
        ----------
        data : torch.Tensor or numpy.ndarray
            Input data to convert
            
        Returns
        -------
        numpy.ndarray
            Converted numpy array
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)
        
    def _prepare_data(self, 
                     z_train: Union[torch.Tensor, np.ndarray],
                     labels_train: Union[torch.Tensor, np.ndarray],
                     z_test: Union[torch.Tensor, np.ndarray],
                     labels_test: Union[torch.Tensor, np.ndarray],
                     label_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for visualization by converting tensors to numpy arrays.
        
        Parameters
        ----------
        z_train, z_test : torch.Tensor or numpy.ndarray
            Latent representations for training and testing data
        labels_train, labels_test : torch.Tensor or numpy.ndarray
            Labels for training and testing data
        label_names : list, optional
            Names for the class labels
            
        Returns
        -------
        tuple
            Processed numpy arrays and label names
            
        Raises
        ------
        ValueError
            If input validation fails
        """
        # Validate inputs
        self._validate_inputs(z_train, labels_train, z_test, labels_test)
        
        # Convert tensors to numpy arrays
        z_train_np = self._convert_to_numpy(z_train)
        z_test_np = self._convert_to_numpy(z_test)
        labels_train_np = self._convert_to_numpy(labels_train)
        labels_test_np = self._convert_to_numpy(labels_test)
        
        # Set default label names if not provided
        if label_names is None:
            max_label = max(np.max(labels_train_np), np.max(labels_test_np))
            label_names = [str(i) for i in range(int(max_label) + 1)]
            
        logger.debug(f"Prepared data - Train: {z_train_np.shape}, Test: {z_test_np.shape}")
        return z_train_np, labels_train_np, z_test_np, labels_test_np, label_names
    
    def _sample_from_gmm(self, gmm: Any, n_samples: int = None) -> Optional[np.ndarray]:
        """Sample from GMM with proper error handling.
        
        Parameters
        ----------
        gmm : GaussianMixture
            Fitted GMM model
        n_samples : int, optional
            Number of samples to generate
            
        Returns
        -------
        numpy.ndarray or None
            Generated samples, or None if sampling fails
        """
        if n_samples is None:
            n_samples = self.config.gmm_samples_default
            
        # Check if GMM has been fitted
        if not hasattr(gmm, 'fitted_') or not gmm.fitted_:
            logger.debug("GMM not fitted yet, skipping sampling")
            return None
            
        # Check if GMM has weights (means it's been fitted)
        if not hasattr(gmm, 'weights_') or gmm.weights_ is None:
            logger.debug("GMM has no weights, skipping sampling")
            return None
            
        try:
            if hasattr(gmm, 'sample'):
                samples, _ = gmm.sample(n_samples)
            elif hasattr(gmm, 'sample_n'):
                samples = gmm.sample_n(n_samples)
            else:
                logger.warning("GMM object has no sampling method")
                return None
            
            if hasattr(samples, 'detach'):
                samples = samples.detach().cpu().numpy()
            return samples
        except Exception as e:
            logger.warning(f"Failed to sample from GMM: {e}")
            return None
        
    def _setup_figure(self, title: str, include_gmm_samples: bool = True) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Set up the figure and axes for visualization.
        
        Parameters
        ----------
        title : str
            Main title for the plot
        include_gmm_samples : bool
            Whether to include a third subplot for GMM samples
            
        Returns
        -------
        tuple
            Figure and axes objects
        """
        if include_gmm_samples:
            fig, axes = plt.subplots(1, 3, figsize=self.config.figure_size_3_plots)
        else:
            fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size_2_plots)
            
        fig.suptitle(title, fontsize=self.config.default_fontsize_title, y=0.98)
        return fig, axes
    
    def _plot_points_and_means(self, 
                              ax: plt.Axes, 
                              data: np.ndarray, 
                              labels: np.ndarray, 
                              means: Optional[np.ndarray], 
                              label_names: List[str], 
                              title_prefix: str,
                              x_label: str, 
                              y_label: str) -> None:
        """
        Plot data points colored by category and GMM means.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on
        data : numpy.ndarray
            2D reduced data to plot
        labels : numpy.ndarray
            Labels for the data points
        means : numpy.ndarray
            GMM means in the reduced space
        label_names : list
            Names for the class labels
        title_prefix : str
            Prefix for the subplot title
        x_label, y_label : str
            Labels for the x and y axes
        """
        # Plot points by category
        for i, name in enumerate(label_names):
            mask = labels == i
            if np.any(mask):  # Only plot if there are points for this label
                ax.scatter(data[mask, 0], data[mask, 1], 
                          s=self.config.scatter_point_size, 
                          color=self.custom_colors[i % len(self.custom_colors)], 
                          label=f"{name}", alpha=self.config.scatter_alpha)
        
        # Plot GMM means with black stars (only if means are available)
        if means is not None and len(means) > 0:
            ax.scatter(means[:, 0], means[:, 1], 
                      s=self.config.mean_marker_size, c='black', marker='*', label="GMM Means")
        
        # Add labels and title
        ax.set_title(f"{title_prefix} Data", fontsize=self.config.default_fontsize_subtitle)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=self.config.grid_alpha)
    
    def _plot_gmm_samples(self, 
                         ax: plt.Axes, 
                         samples: np.ndarray, 
                         means: Optional[np.ndarray], 
                         label_names: List[str], 
                         transform_fn: Callable[[np.ndarray], np.ndarray], 
                         x_label: str, 
                         y_label: str) -> None:
        """
        Plot samples drawn from the GMM model.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on
        samples : numpy.ndarray
            Samples drawn from the GMM
        means : numpy.ndarray
            GMM means in the reduced space
        label_names : list
            Names for the class labels
        transform_fn : callable
            Function to transform samples to the reduced space
        x_label, y_label : str
            Labels for the x and y axes
        """
        # Transform GMM samples to the reduced space
        samples_reduced = transform_fn(samples)
        
        ax.scatter(samples_reduced[:, 0], samples_reduced[:, 1], 
                  s=self.config.scatter_point_size, c='grey', alpha=0.5, label="GMM Samples")
        if means is not None and len(means) > 0:
            ax.scatter(means[:, 0], means[:, 1], 
                      s=self.config.mean_marker_size, c='black', marker='*', label="GMM Means")
        
        # Add labels and title
        ax.set_title("GMM Samples", fontsize=self.config.default_fontsize_subtitle)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=self.config.grid_alpha)
    
    def _add_ellipses(self, ax, means, covariances, transform_components=None, covariance_type=None):
        """
        Add covariance ellipses to the plot for Gaussian components.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to plot on
        means : numpy.ndarray
            GMM means in the reduced space
        covariances : numpy.ndarray
            GMM covariance matrices
        transform_components : numpy.ndarray, optional
            PCA components for transforming covariance matrices
        covariance_type : str, optional
            Type of GMM covariance ('full', 'diag', 'spherical')
        """
        # Early return if no transform components are provided or means are None
        if transform_components is None or means is None or means.size == 0:
            return
            
        # Plot ellipses for each Gaussian component
        for i, mean in enumerate(means):
            # Handle different covariance types
            if covariance_type == 'spherical':
                cov_matrix = np.eye(transform_components.shape[1]) * covariances[i]
            elif covariance_type == 'diag':
                cov_matrix = np.diag(covariances[i])
            else:  # full covariance
                cov_matrix = covariances[i]
            
            # Transform covariance matrix to the reduced space
            cov_matrix_transformed = transform_components @ cov_matrix @ transform_components.T
            
            # Compute eigenvalues and eigenvectors for the ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_transformed)
            eigenvalues = np.sqrt(eigenvalues)  # Convert to standard deviations
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            # Create the ellipse
            ellipse = Ellipse(
                xy=mean,
                width=2 * eigenvalues[0],  # 1 std in x-direction
                height=2 * eigenvalues[1],  # 1 std in y-direction
                angle=angle,
                edgecolor='black',
                linestyle='--',
                facecolor='none',
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(ellipse)
    
    def _extract_gmm_parameters(self, gmm: Any) -> Dict[str, Any]:
        """Extract parameters from GMM model.
        
        Parameters
        ----------
        gmm : GaussianMixture
            Fitted GMM model
            
        Returns
        -------
        dict
            Dictionary containing GMM parameters
        """
        params = {}
        
        # Check if GMM is fitted and has valid means
        if (hasattr(gmm, 'means_') and 
            gmm.means_ is not None and 
            hasattr(gmm, 'converged_') and 
            gmm.converged_ is not None):
            
            try:
                means_np = self._convert_to_numpy(gmm.means_)
                # Check if means contain valid values (not NaN or infinite)
                if np.all(np.isfinite(means_np)) and means_np.size > 0:
                    params['means'] = means_np
                else:
                    params['means'] = None
            except (AttributeError, TypeError, ValueError):
                params['means'] = None
        else:
            params['means'] = None
        
        # Extract covariances if available
        if (hasattr(gmm, 'covariances_') and 
            gmm.covariances_ is not None and 
            params['means'] is not None):
            try:
                cov_np = self._convert_to_numpy(gmm.covariances_)
                if np.all(np.isfinite(cov_np)) and cov_np.size > 0:
                    params['covariances'] = cov_np
                    params['covariance_type'] = getattr(gmm, 'covariance_type', 'full')
                else:
                    params['covariances'] = None
                    params['covariance_type'] = None
            except (AttributeError, TypeError, ValueError):
                params['covariances'] = None
                params['covariance_type'] = None
        else:
            params['covariances'] = None
            params['covariance_type'] = None
            
        return params
    
    def _apply_pca(self, 
                   z_all: np.ndarray, 
                   z_train: np.ndarray, 
                   z_test: np.ndarray, 
                   means: np.ndarray,
                   random_state: Optional[int],
                   **kwargs) -> Dict[str, Any]:
        """Apply PCA dimensionality reduction.
        
        Returns
        -------
        dict
            Dictionary containing reduced data and reducer object
        """
        pca_kwargs = {'n_components': 2}
        if random_state is not None and not GPU_AVAILABLE:
            # cuML's PCA doesn't support random_state
            pca_kwargs['random_state'] = random_state
            
        pca = PCA(**pca_kwargs)
        pca.fit(z_all)
        
        # Convert results to numpy if using cuML (returns cuDF/cupy arrays)
        z_train_reduced = pca.transform(z_train)
        z_test_reduced = pca.transform(z_test)
        means_reduced = pca.transform(means) if means is not None and means.size > 0 else None
        
        if GPU_AVAILABLE:
            import cupy as cp
            z_train_reduced = cp.asnumpy(z_train_reduced) if hasattr(z_train_reduced, 'values') else z_train_reduced
            z_test_reduced = cp.asnumpy(z_test_reduced) if hasattr(z_test_reduced, 'values') else z_test_reduced
            if means_reduced is not None:
                means_reduced = cp.asnumpy(means_reduced) if hasattr(means_reduced, 'values') else means_reduced
            components = cp.asnumpy(pca.components_) if hasattr(pca.components_, 'values') else pca.components_
        else:
            components = pca.components_
        
        return {
            'z_train_reduced': z_train_reduced,
            'z_test_reduced': z_test_reduced,
            'means_reduced': means_reduced,
            'reducer': pca,
            'x_label': kwargs.get('xlabel', "Principal Component 1"),
            'y_label': kwargs.get('ylabel', "Principal Component 2"),
            'transform_components': components
        }
    
    def _apply_umap(self, 
                    z_all: np.ndarray, 
                    z_train: np.ndarray, 
                    z_test: np.ndarray, 
                    means: np.ndarray,
                    random_state: Optional[int],
                    **kwargs) -> Dict[str, Any]:
        """Apply UMAP dimensionality reduction."""
        umap_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['n_neighbors', 'min_dist', 'metric', 'n_components']}
        umap_kwargs.setdefault('n_components', 2)
        if random_state is not None:
            umap_kwargs['random_state'] = random_state
        
        reducer = UMAP(**umap_kwargs)
        z_all_reduced = reducer.fit_transform(z_all)
        
        # Convert cuML results to numpy if needed
        if GPU_AVAILABLE:
            import cupy as cp
            z_all_reduced = cp.asnumpy(z_all_reduced) if hasattr(z_all_reduced, '__cuda_array_interface__') else np.asarray(z_all_reduced)
        
        split = len(z_train)
        z_train_reduced = z_all_reduced[:split]
        z_test_reduced = z_all_reduced[split:]
        
        # Transform means
        if means is not None and means.size > 0:
            means_reduced = reducer.transform(means)
            if GPU_AVAILABLE:
                import cupy as cp
                means_reduced = cp.asnumpy(means_reduced) if hasattr(means_reduced, '__cuda_array_interface__') else np.asarray(means_reduced)
        else:
            means_reduced = None
            
        return {
            'z_train_reduced': z_train_reduced,
            'z_test_reduced': z_test_reduced,
            'means_reduced': means_reduced,
            'reducer': reducer,
            'x_label': "UMAP Dimension 1",
            'y_label': "UMAP Dimension 2",
            'transform_components': None
        }

    
    def _finalize_plot(self, fig, axes):
        """
        Finalize the plot by adding a legend and adjusting layout.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to finalize
        axes : numpy.ndarray
            Array of matplotlib axes
        """
        # Collect handles and labels from all axes
        all_handles = []
        all_labels = []
        
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in all_labels:  # Avoid duplicates
                    all_handles.append(h)
                    all_labels.append(l)
        
        # Add a single legend for all subplots
        fig.legend(all_handles, all_labels, loc='center right', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Make space for title and legend

        
    def visualize(self, 
                  z_train: Union[torch.Tensor, np.ndarray], 
                  labels_train: Union[torch.Tensor, np.ndarray],
                  z_test: Union[torch.Tensor, np.ndarray], 
                  labels_test: Union[torch.Tensor, np.ndarray],
                  gmm: Any, 
                  method: str = 'pca',
                  title: Optional[str] = None,
                  label_names: Optional[List[str]] = None,
                  random_state: Optional[int] = None,
                  **kwargs) -> plt.Figure:
        """
        Visualize latent space with the specified dimensionality reduction technique.
        
        Parameters
        ----------
        z_train, z_test : torch.Tensor or numpy.ndarray
            Latent representations for training and testing data
        labels_train, labels_test : torch.Tensor or numpy.ndarray
            Labels for training and testing data
        gmm : GaussianMixture
            Fitted GMM model
        method : str
            Dimensionality reduction method: 'pca', 'umap', 'kpca', or 'tsne'
        title : str, optional
            Main title for the plot (defaults based on method)
        label_names : list, optional
            Names for the class labels
        random_state : int, optional
            Random seed for reproducibility (if None, results will vary between runs)
        **kwargs : dict
            Additional parameters for specific methods:
            - PCA: None
            - UMAP: 'n_neighbors', 'min_dist', etc.
            - Kernel PCA: 'kernel', 'gamma'
            - t-SNE: 'perplexity', 'max_iter'
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
            
        Raises
        ------
        ValueError
            If input validation fails or unknown method is specified
        """
        logger.info(f"Starting visualization with method: {method}")
        
        # Prepare data
        z_train_np, labels_train_np, z_test_np, labels_test_np, label_names = \
            self._prepare_data(z_train, labels_train, z_test, labels_test, label_names)
        
        # Combine datasets for joint projection
        z_all = np.vstack([z_train_np, z_test_np])
        
        # Extract GMM parameters
        gmm_params = self._extract_gmm_parameters(gmm)
        
        # Generate samples from GMM if available
        gmm_samples = self._sample_from_gmm(gmm)
        include_gmm_samples = gmm_samples is not None
        
        # Set default title if not provided
        if title is None:
            title = f"Latent Space - {method.upper()}"
        
        # Add epoch to title if provided
        if 'epoch' in kwargs and kwargs['epoch'] is not None:
            title = f"Epoch {kwargs['epoch']} - {title}"
        
        # Create figure
        fig, axes = self._setup_figure(title, include_gmm_samples)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reduction_result = self._apply_pca(z_all, z_train_np, z_test_np, 
                                             gmm_params['means'], random_state, **kwargs)
        elif method.lower() == 'umap':
            reduction_result = self._apply_umap(z_all, z_train_np, z_test_np, 
                                              gmm_params['means'], random_state, **kwargs)
        else:
            # For now, fall back to the original implementation for kpca and tsne
            return self._visualize_legacy(z_train_np, labels_train_np, z_test_np, 
                                        labels_test_np, gmm, method, title, 
                                        label_names, random_state, **kwargs)
        
        # Plot data points and means
        self._plot_points_and_means(axes[0], reduction_result['z_train_reduced'], 
                                  labels_train_np, reduction_result['means_reduced'], 
                                  label_names, "Training", 
                                  reduction_result['x_label'], reduction_result['y_label'])
        self._plot_points_and_means(axes[1], reduction_result['z_test_reduced'], 
                                  labels_test_np, reduction_result['means_reduced'], 
                                  label_names, "Testing", 
                                  reduction_result['x_label'], reduction_result['y_label'])
        
        # Add ellipses for covariance (PCA only)
        if (gmm_params['covariances'] is not None and 
            reduction_result['transform_components'] is not None):
            self._add_ellipses(axes[0], reduction_result['means_reduced'], 
                             gmm_params['covariances'], 
                             reduction_result['transform_components'], 
                             gmm_params['covariance_type'])
            self._add_ellipses(axes[1], reduction_result['means_reduced'], 
                             gmm_params['covariances'], 
                             reduction_result['transform_components'], 
                             gmm_params['covariance_type'])
        
        # Add GMM samples if available
        if include_gmm_samples and len(axes) > 2:
            self._plot_gmm_samples(axes[2], gmm_samples, reduction_result['means_reduced'], 
                                 label_names, reduction_result['reducer'].transform, 
                                 reduction_result['x_label'], reduction_result['y_label'])
        
        # Finalize and save the plot
        self._finalize_plot(fig, axes)
        epoch = kwargs.get('epoch', 'final')
        filename = f"epoch_{epoch}"
        subdir = f"latent/{method.lower()}"
        save_figure(fig, filename, subdir=subdir)
        
        logger.info(f"Completed visualization for method: {method}")
        return fig
    
    def _visualize_legacy(self, z_train_np, labels_train_np, z_test_np, labels_test_np, 
                         gmm, method, title, label_names, random_state, **kwargs):
        """Legacy implementation for kpca and tsne methods."""
        # Combine datasets for joint projection
        z_all = np.vstack([z_train_np, z_test_np])
        split = len(z_train_np)
        
        # Extract GMM parameters safely
        gmm_params = self._extract_gmm_parameters(gmm)
        means = gmm_params['means']
        covariances = gmm_params['covariances']
        covariance_type = gmm_params['covariance_type']
            
        # Generate samples from GMM if available
        gmm_samples = self._sample_from_gmm(gmm)
        include_gmm_samples = gmm_samples is not None
        
        # Create figure
        fig, axes = self._setup_figure(title, include_gmm_samples)
        
        # Apply dimensionality reduction based on the specified method
        if method.lower() == 'kpca':
            # Kernel PCA
            kernel = kwargs.get('kernel', 'rbf')
            gamma = kwargs.get('gamma', None)
            
            # Only set random_state if explicitly provided
            kpca_kwargs = {'n_components': 2, 'kernel': kernel, 'gamma': gamma}
            if random_state is not None:
                kpca_kwargs['random_state'] = random_state
                
            kpca = KernelPCA(**kpca_kwargs)
            kpca.fit(z_all)
            
            z_train_reduced = kpca.transform(z_train_np)
            z_test_reduced = kpca.transform(z_test_np)
            means_reduced = kpca.transform(means) if means is not None and means.size > 0 else None
            
            # Labels for axes
            x_label = f"Kernel PCA Dimension 1 ({kernel})"
            y_label = f"Kernel PCA Dimension 2 ({kernel})"
            
            # Plot data points and means
            self._plot_points_and_means(axes[0], z_train_reduced, labels_train_np, 
                                      means_reduced, label_names, "Training", x_label, y_label)
            self._plot_points_and_means(axes[1], z_test_reduced, labels_test_np, 
                                      means_reduced, label_names, "Testing", x_label, y_label)
            
            # Add GMM samples if available
            if include_gmm_samples and gmm_samples is not None and len(axes) > 2:
                self._plot_gmm_samples(axes[2], gmm_samples, means_reduced, label_names, 
                                     kpca.transform, x_label, y_label)
                
        elif method.lower() == 'tsne':
            # t-SNE
            perplexity = kwargs.get('perplexity', 30)
            max_iter = kwargs.get('max_iter', 1000)
            
            # Adjust perplexity based on dataset size
            # Rule: perplexity should be less than (n_samples / 3)
            # Also, # of neighbors should be at least 3 * perplexity
            n_samples = len(z_all)
            max_perplexity = (n_samples - 1) // 3
            if perplexity > max_perplexity:
                old_perplexity = perplexity
                perplexity = max(5, max_perplexity)  # Minimum perplexity of 5
                logger.warning(f"Adjusted t-SNE perplexity from {old_perplexity} to {perplexity} "
                             f"due to small dataset size (n={n_samples})")
                print(f"⚠️ Adjusted t-SNE perplexity from {old_perplexity} to {perplexity} for dataset with {n_samples} samples")
            
            # Only set random_state if explicitly provided
            tsne_kwargs = {'n_components': 2, 'perplexity': perplexity}
            # Both sklearn and cuML use n_iter parameter
            tsne_kwargs['n_iter'] = max_iter
                
            if random_state is not None:
                tsne_kwargs['random_state'] = random_state
                
            tsne = TSNE(**tsne_kwargs)
            z_all_reduced = tsne.fit_transform(z_all)
            
            # Convert cuML results to numpy if needed
            if GPU_AVAILABLE:
                import cupy as cp
                z_all_reduced = cp.asnumpy(z_all_reduced) if hasattr(z_all_reduced, '__cuda_array_interface__') else np.asarray(z_all_reduced)
            
            z_train_reduced = z_all_reduced[:split]
            z_test_reduced = z_all_reduced[split:]
            
            # Project GMM means to t-SNE space (approximate) - only if means are available
            means_reduced = []
            if means is not None and means.size > 0:
                for mean in means:
                    distances = np.sum((z_all - mean) ** 2, axis=1)
                    closest_idx = np.argmin(distances)
                    means_reduced.append(z_all_reduced[closest_idx])
                means_reduced = np.array(means_reduced)
            else:
                means_reduced = None
            
            # Labels for axes
            x_label = "t-SNE Dimension 1"
            y_label = "t-SNE Dimension 2"
            
            # Plot data points and means
            self._plot_points_and_means(axes[0], z_train_reduced, labels_train_np, 
                                      means_reduced, label_names, "Training", x_label, y_label)
            self._plot_points_and_means(axes[1], z_test_reduced, labels_test_np, 
                                      means_reduced, label_names, "Testing", x_label, y_label)
            
            # For t-SNE, we need a special approach for GMM samples since t-SNE doesn't have transform
            if include_gmm_samples and gmm_samples is not None and len(axes) > 2:
                # For t-SNE, we'll use nearest neighbor in the original space to locate in t-SNE space
                samples_reduced = []
                for sample in gmm_samples:
                    distances = np.sum((z_all - sample) ** 2, axis=1)
                    closest_idx = np.argmin(distances)
                    samples_reduced.append(z_all_reduced[closest_idx])
                samples_reduced = np.array(samples_reduced)
                
                # Create a mock transform function that just returns the pre-computed values
                def tsne_proxy_transform(x):
                    return samples_reduced
                
                self._plot_gmm_samples(axes[2], gmm_samples, means_reduced, label_names, 
                                     tsne_proxy_transform, x_label, y_label)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'umap', "
                           f"'kpca', 'tsne'.")
        
        # Finalize and save the plot
        self._finalize_plot(fig, axes)
        epoch = kwargs.get('epoch', 'final')
        filename = f"epoch_{epoch}"
        subdir = f"latent/{method.lower()}"
        save_figure(fig, filename, subdir=subdir)
        
        return fig
    


def plot_training_losses(train_losses: List[float], 
                        test_losses: List[float], 
                        recon_train_losses: List[float], 
                        recon_test_losses: List[float],
                        gmm_train_losses: List[float], 
                        gmm_test_losses: List[float], 
                        title: Optional[str] = None) -> plt.Figure:
    """
    Plot training and test losses side by side for each loss type.
    
    Parameters
    ----------
    train_losses : list
        Total loss values for training data
    test_losses : list
        Total loss values for test data
    recon_train_losses : list
        Reconstruction loss values for training data
    recon_test_losses : list
        Reconstruction loss values for test data
    gmm_train_losses : list
        GMM error values for training data
    gmm_test_losses : list
        GMM error values for test data
    title : str, optional
        Custom title for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if not train_losses:
        logger.warning("No training losses provided")
        return None
        
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot 1: Reconstruction Loss
    ax = axes[0]
    ax.plot(epochs, recon_train_losses, 'b-', linewidth=2, label='Train')
    ax.plot(epochs, recon_test_losses, 'r-', linewidth=2, label='Test')
    
    ax.set_title('Reconstruction Loss', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
    # Plot 2: GMM Error
    ax = axes[1]
    ax.plot(epochs, gmm_train_losses, 'b-', linewidth=2, label='Train')
    ax.plot(epochs, gmm_test_losses, 'r-', linewidth=2, label='Test')
    
    ax.set_title('GMM Error', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
    # Plot 3: Total Loss
    ax = axes[2]
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train')
    ax.plot(epochs, test_losses, 'r-', linewidth=2, label='Test')
    
    ax.set_title('Total Loss', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add main title if provided
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    epoch = title.split()[-1] if title and "Epoch" in title else "final"
    filename = f"losses_epoch_{epoch}"
    save_figure(fig, filename, subdir="losses")
    
    return fig


def plot_images(images: Union[torch.Tensor, np.ndarray], 
                labels: Union[torch.Tensor, np.ndarray], 
                title: str, 
                epoch: Optional[int] = None, 
                cmap: str = 'viridis',
                label_map: Optional[Dict[int, str]] = None) -> None:
    """
    Plots a grid of images with 2 images for each label stacked on top of each other,
    and the label displayed at the bottom of the second row.

    Parameters
    ----------
    images : torch.Tensor or numpy.ndarray
        Tensor of images to plot.
    labels : torch.Tensor or numpy.ndarray
        Tensor of labels corresponding to the images.
    title : str
        Title for the plot.
    epoch : int, optional
        Epoch number to include in the title.
    cmap : str, optional
        Colormap to use for the images (default is 'viridis').
    label_map : dict, optional
        Mapping from numeric labels to string names. If None, uses Fashion-MNIST labels.
    """
    # Default Fashion-MNIST label mapping
    if label_map is None:
        label_map = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }

    # Convert to numpy if needed
    if hasattr(images, 'detach'):
        images = images.detach().cpu().numpy()
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()
    
    # Determine unique labels
    unique_labels = sorted(set(labels.flatten()))
    max_images_per_label = 2

    # Group images by labels
    grouped_images = {label: [] for label in unique_labels}
    for img, label in zip(images, labels):
        label_val = int(label)
        if len(grouped_images[label_val]) < max_images_per_label:
            grouped_images[label_val].append(img)

    # Flatten the grouped images and labels
    stacked_images = []
    stacked_labels = []
    for label, imgs in grouped_images.items():
        if len(imgs) == max_images_per_label:  # Ensure we have exactly max_images_per_label images per label
            stacked_images.extend(imgs)
            stacked_labels.append(label_map.get(label, f"Label {label}"))  # Add label once per column

    n_labels = len(stacked_labels)
    if n_labels == 0:
        logger.warning("No images to plot.")
        return

    n_cols = n_labels  # One column per label
    n_rows = max_images_per_label  # Two rows per label (stacked images)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    
    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, label in enumerate(stacked_labels):
        for row in range(n_rows):
            ax = axes[row, col] if n_cols > 1 else axes[row]
            img = stacked_images[col * n_rows + row].squeeze()
            ax.imshow(img, cmap=cmap, interpolation='nearest')
            ax.axis('off')
        # Add label below the last row
        bottom_ax = axes[-1, col] if n_cols > 1 else axes[-1]
        bottom_ax.text(
            0.5, -0.2, label, fontsize=12, ha='center', va='top', transform=bottom_ax.transAxes
        )
    if epoch is not None:
        title = f'Epoch {epoch} - {title}'
    else:
        title = f'{title}'
    plt.suptitle(title)
    plt.tight_layout()
    clean_title = title.replace(" ", "_").replace("/", "_").lower()
    filename = f"{clean_title}_epoch_{epoch if epoch is not None else 'final'}"
    save_figure(fig, filename, subdir="images")


def plot_gmm_images(model, gmm, title, epoch=None, cmap='viridis', top_n=10, device='cuda'):
    """
    Plots reconstructions of the GMM means with the largest mixing coefficients.
    
    Parameters
    ----------
    model : torch.nn.Module
        Decoder model
    gmm : GaussianMixture
        Fitted GMM model
    title : str
        Plot title
    epoch : int, optional
        Current epoch number for title
    cmap : str, optional
        Colormap to use for images
    top_n : int, optional
        Number of top components to show (by weight)
    device : str, optional
        Device to use for tensor operations
    """
    with torch.no_grad():
        # Get weights and means from GMM
        weights = gmm.weights_.detach().cpu()
        means = gmm.means_.detach()
        
        # Sort indices by weights (descending)
        sorted_indices = torch.argsort(weights, descending=True)
        
        # Take top N components
        top_indices = sorted_indices[:top_n]
        top_weights = weights[top_indices]
        top_means = means[top_indices]
        
        # Generate reconstructions
        reconstructions = model(top_means)
        reconstructions = reconstructions.cpu()
        
        # Create a figure with top_n columns and 1 row
        n_cols = min(5, top_n)  # Limit to 5 columns per row for readability
        n_rows = (top_n + n_cols - 1) // n_cols  # Ceiling division for number of rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        
        # Handle single row/column case
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])
        
        # Plot each component
        for i in range(top_n):
            row, col = i // n_cols, i % n_cols
            ax = axes[row][col]
            img = reconstructions[i].squeeze()
            ax.imshow(img, cmap=cmap, interpolation='nearest')
            
            # Format weight as percentage
            weight_pct = top_weights[i].item() * 100
            ax.set_title(f"Weight: {weight_pct:.1f}%")
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(top_n, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row][col].axis('off')
        
        if epoch is not None:
            plt.suptitle(f'Epoch {epoch} - {title}')
        else:
            plt.suptitle(title)
            
        plt.tight_layout()
        clean_title = title.replace(" ", "_").replace("/", "_").lower()
        filename = f"gmm_means_{clean_title}_epoch_{epoch if epoch is not None else 'final'}"
        save_figure(fig, filename, subdir="gmm")

def plot_gmm_samples(model, gmm, title, n_samples=20, epoch=None, cmap='viridis', device='cuda'):
    """
    Plots images generated from random samples drawn from the GMM distribution.
    
    Parameters
    ----------
    model : torch.nn.Module
        Decoder model
    gmm : GaussianMixture
        Fitted GMM model
    title : str
        Plot title
    n_samples : int
        Number of samples to generate and display
    epoch : int, optional
        Current epoch number for title
    cmap : str, optional
        Colormap to use for images
    device : str, optional
        Device to use for tensor operations
    """
    with torch.no_grad():
        # Sample from the GMM
        samples, _ = gmm.sample(n_samples)  # Returns samples and their component labels
        
        # Generate images from the samples
        generated_images = model(samples)
        generated_images = generated_images.cpu()
        
        # Create a grid layout for displaying the images
        n_cols = min(5, n_samples)  # Max 5 columns
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        
        # Handle single row/column case
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])
        
        # Plot each generated image
        for i in range(n_samples):
            row, col = i // n_cols, i % n_cols
            ax = axes[row][col]
            img = generated_images[i].squeeze()
            ax.imshow(img, cmap=cmap, interpolation='nearest')
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row][col].axis('off')
        
        if epoch is not None:
            plt.suptitle(f'Epoch {epoch} - {title}')
        else:
            plt.suptitle(title)
            
        plt.tight_layout()
        clean_title = title.replace(" ", "_").replace("/", "_").lower()
        filename = f"gmm_samples_{clean_title}_epoch_{epoch if epoch is not None else 'final'}"
        save_figure(fig, filename, subdir="gmm")
import math
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from umap import UMAP
import torch

class LatentSpaceVisualizer:
    """
    A unified class for visualizing latent spaces with different dimensionality reduction techniques.
    Provides visualization for PCA, UMAP, Kernel PCA, and t-SNE with GMM integration.
    """
    
    def __init__(self, custom_colors=None):
        """
        Initialize the visualizer with common settings.
        
        Parameters
        ----------
        custom_colors : list, optional
            Custom color palette for visualization
        """
        # Default color palette
        self.custom_colors = custom_colors or [
            '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
            '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'
        ]
        
    def _prepare_data(self, z_train, labels_train, z_test, labels_test, label_names=None):
        """
        Prepare data for visualization by converting tensors to numpy arrays.
        
        Parameters
        ----------
        z_train, z_test : torch.Tensor
            Latent representations for training and testing data
        labels_train, labels_test : torch.Tensor
            Labels for training and testing data
        label_names : list, optional
            Names for the class labels
            
        Returns
        -------
        tuple
            Processed numpy arrays and label names
        """
        # Convert tensors to numpy arrays
        z_train_np = z_train.detach().cpu().numpy() if hasattr(z_train, 'detach') else z_train
        z_test_np = z_test.detach().cpu().numpy() if hasattr(z_test, 'detach') else z_test
        labels_train_np = labels_train.cpu().numpy() if hasattr(labels_train, 'cpu') else labels_train
        labels_test_np = labels_test.cpu().numpy() if hasattr(labels_test, 'cpu') else labels_test
        
        # Set default label names if not provided
        if label_names is None:
            label_names = [str(i) for i in range(max(np.max(labels_train_np), np.max(labels_test_np)) + 1)]
            
        return z_train_np, labels_train_np, z_test_np, labels_test_np, label_names
        
    def _setup_figure(self, title, include_gmm_samples=True):
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
            fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
        fig.suptitle(title, fontsize=16, y=0.98)
        return fig, axes
    
    def _plot_points_and_means(self, ax, data, labels, means, label_names, title_prefix, 
                             x_label, y_label):
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
            ax.scatter(data[mask, 0], data[mask, 1], 
                      s=10, color=self.custom_colors[i % len(self.custom_colors)], 
                      label=f"{name}", alpha=0.7)
        
        # Plot GMM means with black stars
        ax.scatter(means[:, 0], means[:, 1], 
                  s=100, c='black', marker='*', label="GMM Means")
        
        # Add labels and title
        ax.set_title(f"{title_prefix} Data", fontsize=14)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)
    
    def _plot_gmm_samples(self, ax, samples, means, label_names, transform_fn, x_label, y_label):
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
        
        ax.scatter(samples_reduced[:, 0], samples_reduced[:, 1], s=10, c='grey', alpha=0.5, label="GMM Samples")
        ax.scatter(means[:, 0], means[:, 1], s=100, c='black', marker='*', label="GMM Means")
        
        # Add labels and title
        ax.set_title("GMM Samples", fontsize=14)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)
    
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
        # Early return if no transform components are provided
        if transform_components is None:
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

        
    def visualize(self, z_train, labels_train, z_test, labels_test, gmm, 
                  method='pca', title=None, label_names=None, random_state=42, 
                  **kwargs):
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
        random_state : int
            Random seed for reproducibility
        **kwargs : dict
            Additional parameters for specific methods:
            - PCA: None
            - UMAP: 'n_neighbors', 'min_dist', etc.
            - Kernel PCA: 'kernel', 'gamma'
            - t-SNE: 'perplexity', 'n_iter'
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        # Prepare data
        z_train_np, labels_train_np, z_test_np, labels_test_np, label_names = \
            self._prepare_data(z_train, labels_train, z_test, labels_test, label_names)
        
        # Combine datasets for joint projection
        z_all = np.vstack([z_train_np, z_test_np])
        split = len(z_train_np)
        
        # Extract GMM parameters
        means = gmm.means_.detach().cpu().numpy() if hasattr(gmm.means_, 'detach') else gmm.means_
        if hasattr(gmm, 'covariances_'):
            covariances = gmm.covariances_.detach().cpu().numpy() if hasattr(gmm.covariances_, 'detach') else gmm.covariances_
            covariance_type = gmm.covariance_type if hasattr(gmm, 'covariance_type') else 'full'
        else:
            covariances = None
            covariance_type = None
            
        # Determine if we should include GMM samples
        include_gmm_samples = gmm is not None
        
        # Generate samples from GMM if available
        gmm_samples = None
        if include_gmm_samples:
            try:
                # Try to sample from the GMM (method differs between implementations)
                if hasattr(gmm, 'sample'):
                    gmm_samples, _ = gmm.sample(2000)
                    if hasattr(gmm_samples, 'detach'):
                        gmm_samples = gmm_samples.detach().cpu().numpy()
                elif hasattr(gmm, 'sample_n'):
                    gmm_samples = gmm.sample_n(2000)
                    if hasattr(gmm_samples, 'detach'):
                        gmm_samples = gmm_samples.detach().cpu().numpy()
                else:
                    # If no sampling method is available, don't include GMM samples plot
                    include_gmm_samples = False
            except:
                # If sampling fails, don't include GMM samples plot
                include_gmm_samples = False
        
        # Set default title if not provided
        if title is None:
            title = f"Latent Space - {method.upper()}"
        
        # Add epoch to title if provided
        if 'epoch' in kwargs and kwargs['epoch'] is not None:
            title = f"Epoch {kwargs['epoch']} - {title}"
        
        # Create figure
        fig, axes = self._setup_figure(title, include_gmm_samples)
        
        # Apply dimensionality reduction based on the specified method
        if method.lower() == 'pca':
            # PCA
            pca = PCA(n_components=2, random_state=random_state)
            pca.fit(z_all)
            
            z_train_reduced = pca.transform(z_train_np)
            z_test_reduced = pca.transform(z_test_np)
            means_reduced = pca.transform(means)
            
            # Labels for axes
            x_label = kwargs.get('xlabel', "Principal Component 1")
            y_label = kwargs.get('ylabel', "Principal Component 2")
            
            # Plot data points and means
            self._plot_points_and_means(axes[0], z_train_reduced, labels_train_np, 
                                      means_reduced, label_names, "Training", x_label, y_label)
            self._plot_points_and_means(axes[1], z_test_reduced, labels_test_np, 
                                      means_reduced, label_names, "Testing", x_label, y_label)
            
            # Add ellipses for covariance
            if covariances is not None:
                self._add_ellipses(axes[0], means_reduced, covariances, pca.components_, covariance_type)
                self._add_ellipses(axes[1], means_reduced, covariances, pca.components_, covariance_type)
            
            # Add GMM samples if available
            if include_gmm_samples and gmm_samples is not None:
                self._plot_gmm_samples(axes[2], gmm_samples, means_reduced, label_names, 
                                     pca.transform, x_label, y_label)
                
        elif method.lower() == 'umap':
            # UMAP
            umap_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['n_neighbors', 'min_dist', 'metric', 'n_components']}
            umap_kwargs.setdefault('n_components', 2)
            umap_kwargs.setdefault('random_state', random_state)
            
            reducer = UMAP(**umap_kwargs)
            z_all_reduced = reducer.fit_transform(z_all)
            
            z_train_reduced = z_all_reduced[:split]
            z_test_reduced = z_all_reduced[split:]
            means_reduced = reducer.transform(means)
            
            # Labels for axes
            x_label = "UMAP Dimension 1"
            y_label = "UMAP Dimension 2"
            
            # Plot data points and means
            self._plot_points_and_means(axes[0], z_train_reduced, labels_train_np, 
                                      means_reduced, label_names, "Training", x_label, y_label)
            self._plot_points_and_means(axes[1], z_test_reduced, labels_test_np, 
                                      means_reduced, label_names, "Testing", x_label, y_label)
            
            # Add GMM samples if available
            if include_gmm_samples and gmm_samples is not None:
                self._plot_gmm_samples(axes[2], gmm_samples, means_reduced, label_names, 
                                     reducer.transform, x_label, y_label)
                
        elif method.lower() == 'kpca':
            # Kernel PCA
            kernel = kwargs.get('kernel', 'rbf')
            gamma = kwargs.get('gamma', None)
            
            kpca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma, random_state=random_state)
            kpca.fit(z_all)
            
            z_train_reduced = kpca.transform(z_train_np)
            z_test_reduced = kpca.transform(z_test_np)
            means_reduced = kpca.transform(means)
            
            # Labels for axes
            x_label = f"Kernel PCA Dimension 1 ({kernel})"
            y_label = f"Kernel PCA Dimension 2 ({kernel})"
            
            # Plot data points and means
            self._plot_points_and_means(axes[0], z_train_reduced, labels_train_np, 
                                      means_reduced, label_names, "Training", x_label, y_label)
            self._plot_points_and_means(axes[1], z_test_reduced, labels_test_np, 
                                      means_reduced, label_names, "Testing", x_label, y_label)
            
            # Add GMM samples if available
            if include_gmm_samples and gmm_samples is not None:
                self._plot_gmm_samples(axes[2], gmm_samples, means_reduced, label_names, 
                                     kpca.transform, x_label, y_label)
                
        elif method.lower() == 'tsne':
            # t-SNE
            perplexity = kwargs.get('perplexity', 30)
            n_iter = kwargs.get('n_iter', 1000)
            
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                       random_state=random_state)
            z_all_reduced = tsne.fit_transform(z_all)
            
            z_train_reduced = z_all_reduced[:split]
            z_test_reduced = z_all_reduced[split:]
            
            # Project GMM means to t-SNE space (approximate)
            means_reduced = []
            for mean in means:
                distances = np.sum((z_all - mean) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                means_reduced.append(z_all_reduced[closest_idx])
                
            means_reduced = np.array(means_reduced)
            
            # Labels for axes
            x_label = "t-SNE Dimension 1"
            y_label = "t-SNE Dimension 2"
            
            # Plot data points and means
            self._plot_points_and_means(axes[0], z_train_reduced, labels_train_np, 
                                      means_reduced, label_names, "Training", x_label, y_label)
            self._plot_points_and_means(axes[1], z_test_reduced, labels_test_np, 
                                      means_reduced, label_names, "Testing", x_label, y_label)
            
            # For t-SNE, we need a special approach for GMM samples since t-SNE doesn't have transform
            if include_gmm_samples and gmm_samples is not None:
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
            raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'umap', 'kpca', or 'tsne'.")
        
        # Finalize and show the plot
        self._finalize_plot(fig, axes)
        plt.show()
        
        return fig
    


def plot_training_losses(train_losses, test_losses, recon_train_losses, recon_test_losses, 
                         gmm_train_losses, gmm_test_losses, title=None):
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
    """
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
    plt.show()
    
    return fig


def plot_images(images, labels, title, epoch=None, cmap='viridis'):
    """
    Plots a grid of images with 2 images for each label stacked on top of each other,
    and the label displayed at the bottom of the second row.

    Parameters
    ----------
    images : torch.Tensor
        Tensor of images to plot.
    labels : torch.Tensor
        Tensor of labels corresponding to the images.
    title : str
        Title for the plot.
    epoch : int
        Epoch number to include in the title.
    cmap : str, optional
        Colormap to use for the images (default is 'viridis').
    """
    # Map numeric labels to item types
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

    # Group images by labels
    grouped_images = {label: [] for label in range(10)}
    for img, label in zip(images, labels):
        if len(grouped_images[label.item()]) < 2:  # Select only 2 images per label
            grouped_images[label.item()].append(img)

    # Flatten the grouped images and labels
    stacked_images = []
    stacked_labels = []
    for label, imgs in grouped_images.items():
        if len(imgs) == 2:  # Ensure we have exactly 2 images per label
            stacked_images.extend(imgs)
            stacked_labels.append(label_map[label])  # Add label once per column

    print(f"Plotting {len(stacked_images)} images.")
    n_labels = len(stacked_labels)
    if n_labels == 0:
        print("No images to plot.")
        return

    n_cols = n_labels  # One column per label
    n_rows = 2  # Two rows per label (stacked images)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))

    for col, label in enumerate(stacked_labels):
        for row in range(n_rows):
            ax = axes[row, col]
            img = stacked_images[col * n_rows + row].squeeze()
            ax.imshow(img, cmap=cmap, interpolation='nearest')
            ax.axis('off')
        # Add label below the second row
        axes[1, col].text(
            0.5, -0.2, label, fontsize=12, ha='center', va='top', transform=axes[1, col].transAxes
        )
    if epoch is not None:
        title = f'Epoch {epoch} - {title}'
    else:
        title = f'{title}'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


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
        plt.show()

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
        plt.show()
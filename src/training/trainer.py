"""
Training utilities and trainer class for ImageDGD.
"""

import torch
import torch.nn.functional as F
from clearml import Task
import time
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Any
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

# Additional optimizers
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    Lion = None

try:
    from sophia import SophiaG
    SOPHIA_AVAILABLE = True
except ImportError:
    SOPHIA_AVAILABLE = False
    SophiaG = None

from ..models import RepresentationLayer, DGD, ConvDecoder, GaussianMixture
from ..visualization import LatentSpaceVisualizer, plot_training_losses, plot_images, plot_gmm_images, plot_gmm_samples


class DGDTrainer:
    """
    Trainer class for DGD model with ClearML integration.
    """
    
    def __init__(self, config: DictConfig, device: torch.device, verbose: bool = True):
        """
        Initialize the trainer.
        
        Parameters:
        ----------
        config: Training configuration
        device: Device to run training on
        verbose: Whether to print detailed training information
        """
        self.config = config
        self.device = device
        self.training_config = config.training
        self.verbose = verbose
        
        # Initialize ClearML task
        self.task = Task.current_task()
        if self.task is None:
            # If no task is already initialized, create one
            self.task = Task.init(project_name="ImageDGD", task_name="DGD Training")
        
        # Initialize tracking lists
        self.train_losses = []
        self.test_losses = []
        self.gmm_train_losses = []
        self.gmm_test_losses = []
        self.recon_train_losses = []
        self.recon_test_losses = []
        
        # Initialize visualizer
        self.visualizer = LatentSpaceVisualizer()
        
        # Timing
        self.epoch_times = []
        
    def _calc_improvement(self, loss_list: List[float]) -> float:
        """Calculate percentage improvement compared to previous epoch."""
        if len(loss_list) < 2:
            return 0.0
        previous = loss_list[-2]
        current = loss_list[-1]
        return ((previous - current) / previous) * 100 if previous != 0 else 0.0
    
    def _create_model_components(self, train_loader, test_loader) -> Tuple:
        """Create model components based on configuration."""
        model_config = self.config.model
        
        # Get dataset sizes
        nsample_train = len(train_loader.dataset)
        nsample_test = len(test_loader.dataset)
        
        # Create representation layers
        # Prepare distribution parameters based on the distribution type
        dist_params = {}
        if hasattr(model_config.representation, 'radius'):
            dist_params['radius'] = model_config.representation.radius
        
        # Handle other potential distribution parameters
        for param in ['mean', 'cov', 'low', 'high', 'loc', 'scale', 'scale_matrix', 
                     'rate', 'df', 'mu', 'alpha', 'beta', 'delta']:
            if hasattr(model_config.representation, param):
                dist_params[param] = getattr(model_config.representation, param)
        
        rep = RepresentationLayer(
            dim=model_config.representation.n_features,
            n_samples=nsample_train,
            dist=model_config.representation.distribution,
            dist_params=dist_params,
            device=self.device
        )
        
        test_rep = RepresentationLayer(
            dim=model_config.representation.n_features,
            n_samples=nsample_test,
            dist=model_config.representation.distribution,
            dist_params=dist_params,
            device=self.device
        )
        
        # Create decoder
        decoder = ConvDecoder(
            latent_dim=model_config.representation.n_features,
            hidden_dims=model_config.decoder.hidden_dims,
            output_channels=model_config.decoder.output_channels,
            output_size=model_config.decoder.output_size,
            use_batch_norm=model_config.decoder.use_batch_norm,
            activation=model_config.decoder.activation,
            final_activation=model_config.decoder.final_activation,
            dropout_rate=model_config.decoder.dropout_rate,
            init_size=model_config.decoder.init_size
        ).to(self.device)
        
        # Create GMM
        gmm = GaussianMixture(
            n_features=model_config.representation.n_features,
            n_components=model_config.gmm.n_components,
            covariance_type=model_config.gmm.covariance_type,
            init_params=model_config.gmm.init_params,
            device=self.device,
            random_state=self.config.random_seed,
            verbose=model_config.gmm.verbose,
            max_iter=model_config.gmm.max_iter,
            tol=model_config.gmm.tol,
            n_init=model_config.gmm.n_init,
            warm_start=model_config.gmm.warm_start
        )
        
        # Create full model
        model = DGD(decoder, rep, gmm)
        
        return model, rep, test_rep, gmm
    
    def _get_optimizer_class(self, optimizer_name: str):
        """Get optimizer class by name with support for PyTorch, Lion, and Sophia optimizers."""
        optimizer_name = optimizer_name.lower()
        
        # PyTorch optimizers
        pytorch_optimizers = {
            'adadelta': torch.optim.Adadelta,
            'adagrad': torch.optim.Adagrad,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sparseadam': torch.optim.SparseAdam,
            'adamax': torch.optim.Adamax,
            'asgd': torch.optim.ASGD,
            'lbfgs': torch.optim.LBFGS,
            'nadam': torch.optim.NAdam,
            'radam': torch.optim.RAdam,
            'rmsprop': torch.optim.RMSprop,
            'rprop': torch.optim.Rprop,
            'sgd': torch.optim.SGD,
        }
        
        # Additional optimizers
        additional_optimizers = {}
        if LION_AVAILABLE:
            additional_optimizers['lion'] = Lion
        if SOPHIA_AVAILABLE:
            additional_optimizers['sophia'] = SophiaG
            additional_optimizers['sophiag'] = SophiaG
        
        all_optimizers = {**pytorch_optimizers, **additional_optimizers}
        
        if optimizer_name in all_optimizers:
            return all_optimizers[optimizer_name]
        
        # If not found, try to get from torch.optim dynamically
        if hasattr(torch.optim, optimizer_name.upper()):
            return getattr(torch.optim, optimizer_name.upper())
        elif hasattr(torch.optim, optimizer_name.capitalize()):
            return getattr(torch.optim, optimizer_name.capitalize())
        
        available_opts = list(all_optimizers.keys())
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available optimizers: {available_opts}")
    
    def _create_optimizer(self, optimizer_class, parameters, config):
        """Create optimizer instance with proper parameter filtering."""
        # Get optimizer signature to filter supported parameters
        import inspect
        sig = inspect.signature(optimizer_class.__init__)
        supported_params = set(sig.parameters.keys()) - {'self', 'params'}
        
        # Filter config to only include supported parameters
        filtered_config = {}
        for key, value in config.items():
            if key == 'type':
                continue
            if key in supported_params:
                filtered_config[key] = value
            else:
                print(f"Warning: Parameter '{key}' not supported by {optimizer_class.__name__}, ignoring.")
        
        return optimizer_class(parameters, **filtered_config)

    def _create_optimizers(self, model, rep, test_rep) -> List:
        """Create optimizers based on configuration with support for all PyTorch optimizers plus Lion and Sophia."""
        training_config = self.training_config
        
        # Decoder optimizer
        decoder_config = training_config.optimizer.decoder
        decoder_optimizer_class = self._get_optimizer_class(decoder_config.type)
        decoder_optimizer = self._create_optimizer(
            decoder_optimizer_class, 
            model.decoder.parameters(), 
            decoder_config
        )
        
        # Representation optimizers
        rep_config = training_config.optimizer.representation
        rep_optimizer_class = self._get_optimizer_class(rep_config.type)
        
        trainrep_optimizer = self._create_optimizer(
            rep_optimizer_class,
            rep.parameters(),
            rep_config
        )
        
        testrep_optimizer = self._create_optimizer(
            rep_optimizer_class,
            test_rep.parameters(),
            rep_config
        )
        
        return [decoder_optimizer, trainrep_optimizer, testrep_optimizer]
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics to ClearML."""
        logger = self.task.get_logger()
        for metric_name, value in metrics.items():
            logger.report_scalar("metrics", metric_name, value, iteration=epoch)
    
    def _should_plot(self, epoch: int) -> bool:
        """Check if should plot at current epoch."""
        plot_interval = self.training_config.logging.plot_interval
        return (epoch % plot_interval == 0 or 
                epoch == self.training_config.epochs or 
                epoch == 1)
    
    def _plot_visualizations(self, epoch: int, model, rep, test_rep, gmm, 
                           sample_data, class_names):
        """Generate and save visualizations."""
        if not self._should_plot(epoch):
            return
            
        indices_train, images_train, labels_train, indices_test, images_test, labels_test = sample_data
        
        with torch.no_grad():
            # Plot reconstructed images
            z_train = rep(indices_train)
            reconstructions_train = model.decoder(z_train)
            plot_images(reconstructions_train.cpu(), labels_train.cpu(), 
                       "Reconstructed Train Images", epoch=epoch)
            
            z_test = test_rep(indices_test)
            reconstructions_test = model.decoder(z_test)
            plot_images(reconstructions_test.cpu(), labels_test.cpu(), 
                       "Reconstructed Test Images", epoch=epoch)
            
            # Plot GMM visualizations (only after GMM is fitted)
            first_epoch_gmm = self.training_config.first_epoch_gmm
            if epoch >= first_epoch_gmm and gmm is not None:
                plot_gmm_images(
                    model.decoder, gmm, "GMM Component Means (by weight)",
                    epoch=epoch, top_n=self.config.model.gmm.n_components, 
                    device=self.device
                )
                plot_gmm_samples(
                    model.decoder, gmm, "Generated Images from GMM Samples",
                    n_samples=self.config.model.gmm.n_components, 
                    epoch=epoch, device=self.device
                )
                
            # Plot latent space visualizations
            if gmm is not None and (epoch == 1 or epoch == first_epoch_gmm or self._should_plot(epoch)):
                z_train_all = rep.z.detach()
                z_test_all = test_rep.z.detach()
                
                # Get all labels for visualization
                labels_train_all = torch.tensor([label for _, _, label in self.train_dataset])
                labels_test_all = torch.tensor([label for _, _, label in self.test_dataset])
                
                # Plot different dimensionality reduction techniques
                self.visualizer.visualize(
                    z_train_all, labels_train_all, z_test_all, labels_test_all, gmm,
                    method='pca', title="Latent Space - PCA",
                    label_names=class_names, epoch=epoch
                )
                
                self.visualizer.visualize(
                    z_train_all, labels_train_all, z_test_all, labels_test_all, gmm,
                    method='tsne', perplexity=30, n_iter=1000,
                    title="Latent Space - t-SNE",
                    random_state=self.config.random_seed,
                    label_names=class_names, epoch=epoch
                )
                
                self.visualizer.visualize(
                    z_train_all, labels_train_all, z_test_all, labels_test_all, gmm,
                    method='umap', n_neighbors=20, n_components=2, min_dist=0.01,
                    title="Latent Space - UMAP",
                    random_state=self.config.random_seed,
                    label_names=class_names, epoch=epoch
                )
    
    def train(self, train_loader, test_loader, sample_data, class_names) -> Dict[str, Any]:
        """
        Main training loop.
        
        Parameters:
        ----------
        train_loader: Training data loader
        test_loader: Test data loader
        sample_data: Sample data for visualization
        class_names: List of class names
        
        Returns:
        -------
        Dictionary containing training results
        """
        # Store datasets for visualization
        self.train_dataset = train_loader.dataset
        self.test_dataset = test_loader.dataset
        
        # Create model components
        model, rep, test_rep, gmm = self._create_model_components(train_loader, test_loader)
        
        # Create optimizers
        optimizers = self._create_optimizers(model, rep, test_rep)
        decoder_optimizer, trainrep_optimizer, testrep_optimizer = optimizers
        
        # Log model parameters
        decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        rep_params = sum(p.numel() for p in rep.parameters() if p.requires_grad)
        test_rep_params = sum(p.numel() for p in test_rep.parameters() if p.requires_grad)
        
        if self.verbose:
            print(f"Decoder parameters: {decoder_params:,} ({decoder_params / 1e6:.2f}M)")
            print(f"Train representation parameters: {rep_params:,} ({rep_params / 1e6:.2f}M)")
            print(f"Test representation parameters: {test_rep_params:,} ({test_rep_params / 1e6:.2f}M)")
            print(f"Total trainable parameters: {decoder_params + rep_params + test_rep_params:,} "
                  f"({(decoder_params + rep_params + test_rep_params) / 1e6:.2f}M)")
            
            # Print training configuration info
            print(f"Training for {self.training_config.epochs} epochs")
            print(f"Using device: {self.device}")
            print(f"Batch size: {self.config.data.batch_size}")
            if self.config.data.use_subset:
                print(f"Using data subset: {self.config.data.subset_fraction:.1%}")
        
        # ClearML logging
        self.task.connect(self.config)  # Connect the configuration
        self.task.set_parameters({
            "decoder_params": decoder_params,
            "rep_params": rep_params,
            "test_rep_params": test_rep_params,
            "total_params": decoder_params + rep_params + test_rep_params
        })
        
        # Training loop
        start_time = time.time()
        
        for epoch in tqdm(range(1, self.training_config.epochs + 1), desc="Training"):
            epoch_start_time = time.time()
            
            # Initialize loss tracking
            train_loss = 0.0
            test_loss = 0.0
            gmm_train_loss = 0.0
            gmm_test_loss = 0.0
            recon_train_loss = 0.0
            recon_test_loss = 0.0
            
            # Initialize or refit GMM
            first_epoch_gmm = self.training_config.first_epoch_gmm
            refit_gmm_interval = self.training_config.refit_gmm_interval
            
            if epoch == first_epoch_gmm or (refit_gmm_interval and epoch % refit_gmm_interval == 0):
                if self.verbose:
                    print(f"Fitting GMM at epoch {epoch}...")
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=1000 if epoch == first_epoch_gmm else 100)
            elif epoch > first_epoch_gmm:
                if self.verbose and epoch % (refit_gmm_interval or 10) == 0:
                    print(f"Updating GMM at epoch {epoch}...")
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=100, warm_start=True)
            
            # Training phase
            model.decoder.train()
            trainrep_optimizer.zero_grad()
            
            for i, (index, x, labels_batch) in enumerate(train_loader):
                decoder_optimizer.zero_grad()
                
                x, index = x.to(self.device), index.to(self.device)
                
                # Forward pass
                z = rep(index)
                y = model.decoder(z)
                recon_loss = F.mse_loss(y, x, reduction='sum')
                
                # Add GMM loss if applicable
                if epoch >= first_epoch_gmm and gmm is not None:
                    gmm_error = -self.training_config.lambda_gmm * torch.sum(gmm.score_samples(z))
                    loss = recon_loss + gmm_error
                else:
                    loss = recon_loss
                    gmm_error = torch.tensor(0.0).to(self.device)
                
                # Backward pass
                loss.backward()
                decoder_optimizer.step()
                
                # Track losses
                train_loss += loss.item()
                recon_train_loss += recon_loss.item()
                if epoch >= first_epoch_gmm:
                    gmm_train_loss += gmm_error.item()
            
            trainrep_optimizer.step()
            
            # Testing phase - update test representations only
            model.decoder.eval()
            testrep_optimizer.zero_grad()
            
            # Disable gradients for decoder parameters during test phase
            for param in model.decoder.parameters():
                param.requires_grad = False
            
            for i, (index, x, _) in enumerate(test_loader):
                x, index = x.to(self.device), index.to(self.device)
                
                # Forward pass
                z = test_rep(index)
                y = model.decoder(z)
                recon_loss = F.mse_loss(y, x, reduction='sum')
                
                # Add GMM loss if applicable
                if epoch >= first_epoch_gmm and gmm is not None:
                    gmm_error = -self.training_config.lambda_gmm * torch.sum(gmm.score_samples(z))
                    loss = recon_loss + gmm_error
                else:
                    loss = recon_loss
                    gmm_error = torch.tensor(0.0).to(self.device)
                
                # Backward pass for test representations only
                loss.backward()
                
                # Track losses
                test_loss += loss.item()
                recon_test_loss += recon_loss.item()
                if epoch >= first_epoch_gmm:
                    gmm_test_loss += gmm_error.item()
            
            # Re-enable gradients for decoder parameters
            for param in model.decoder.parameters():
                param.requires_grad = True
            
            testrep_optimizer.step()
            
            # Normalize losses
            train_loss /= len(train_loader.dataset)
            test_loss /= len(test_loader.dataset)
            recon_train_loss /= len(train_loader.dataset)
            recon_test_loss /= len(test_loader.dataset)
            gmm_train_loss /= len(train_loader.dataset)
            gmm_test_loss /= len(test_loader.dataset)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.recon_train_losses.append(recon_train_loss)
            self.recon_test_losses.append(recon_test_loss)
            self.gmm_train_losses.append(gmm_train_loss)
            self.gmm_test_losses.append(gmm_test_loss)
            
            # Calculate timing
            epoch_duration = time.time() - epoch_start_time
            self.epoch_times.append(epoch_duration)
            
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.training_config.epochs - epoch
            estimated_time_remaining = remaining_epochs * avg_epoch_time
            
            # Log metrics
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "recon_train_loss": recon_train_loss,
                "recon_test_loss": recon_test_loss,
                "gmm_train_loss": gmm_train_loss,
                "gmm_test_loss": gmm_test_loss,
                "epoch_time": epoch_duration,
                "estimated_time_remaining": estimated_time_remaining
            }
            self._log_metrics(epoch, metrics)
            
            # Print progress
            if (self.verbose and epoch % self.training_config.logging.log_interval == 0) or \
               (not self.verbose and epoch % (self.training_config.logging.log_interval * 5) == 0):
                epoch_time_str = str(timedelta(seconds=int(epoch_duration)))
                remaining_time_str = str(timedelta(seconds=int(estimated_time_remaining)))
                
                if self.verbose:
                    print(f"Epoch {epoch}/{self.training_config.epochs} "
                          f"[TPE: {epoch_time_str}, RT: {remaining_time_str}]; "
                          f"Train Loss: {train_loss:.4f} ({self._calc_improvement(self.train_losses):.2f}%), "
                          f"Test Loss: {test_loss:.4f} ({self._calc_improvement(self.test_losses):.2f}%)")
                    print(f"  Reconstruction - Train: {recon_train_loss:.4f}, Test: {recon_test_loss:.4f}")
                    if epoch >= first_epoch_gmm:
                        print(f"  GMM Loss - Train: {gmm_train_loss:.4f}, Test: {gmm_test_loss:.4f}")
                else:
                    # Compact output for non-verbose mode
                    print(f"Epoch {epoch}/{self.training_config.epochs}: "
                          f"Train={train_loss:.4f}, Test={test_loss:.4f}, "
                          f"Time={epoch_time_str}")
            
            # Plot visualizations
            if self.training_config.logging.save_figures:
                self._plot_visualizations(epoch, model, rep, test_rep, gmm, sample_data, class_names)
                
                # Plot loss curves
                if self._should_plot(epoch):
                    plot_training_losses(
                        self.train_losses, self.test_losses,
                        self.recon_train_losses, self.recon_test_losses,
                        self.gmm_train_losses, self.gmm_test_losses,
                        title=f"Training Losses at Epoch {epoch}"
                    )
        
        # Training complete
        total_time = time.time() - start_time
        if self.verbose:
            print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
            print(f"Final training loss: {self.train_losses[-1]:.4f}")
            print(f"Final test loss: {self.test_losses[-1]:.4f}")
        else:
            print(f"Training completed: {self.train_losses[-1]:.4f} train, {self.test_losses[-1]:.4f} test")
        
        # Return results
        return {
            "model": model,
            "rep": rep,
            "test_rep": test_rep,
            "gmm": gmm,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "total_time": total_time,
            "final_train_loss": self.train_losses[-1],
            "final_test_loss": self.test_losses[-1]
        }

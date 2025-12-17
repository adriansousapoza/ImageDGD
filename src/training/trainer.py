"""
Training utilities and trainer class for ImageDGD.
"""

import torch
import torch.nn.functional as F
import time
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Any
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

# Import tgmm package for Gaussian Mixture Model
from tgmm import GaussianMixture

from ..models import RepresentationLayer, DGD, ConvDecoder
from ..visualization import plot_latent_space, plot_images_by_class
import matplotlib.pyplot as plt


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
        
        # Initialize tracking lists
        self.train_losses = []
        self.test_losses = []
        self.gmm_train_losses = []
        self.gmm_test_losses = []
        self.recon_train_losses = []
        self.recon_test_losses = []
        
        # Timing
        self.epoch_times = []
        
        # Learning rate tracking
        self.learning_rates = []
        self.momentum_betas = []  # Track beta_1 (momentum) values
        
        # Best loss tracking
        self.best_train_loss = float('inf')
        self.best_test_loss = float('inf')
        self.best_recon_train = float('inf')
        self.best_recon_test = float('inf')
        self.best_gmm_train = float('inf')
        self.best_gmm_test = float('inf')
        
        # Early stopping (based on training loss)
        # Note: Early stopping only starts after GMM is fitted to avoid false triggers
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.early_stopping_active = False  # Only activate after first GMM fit
    
    def _create_model_components(self, train_loader, test_loader) -> Tuple:
        """Create model components based on configuration."""
        model_config = self.config.model
        
        # Get dataset sizes
        nsample_train = len(train_loader.dataset)
        nsample_test = len(test_loader.dataset)
        
        # Create representation layers
        distribution = model_config.representation.distribution
        
        # Prepare distribution parameters based on the distribution type
        dist_params = {}
        
        # Special handling for PCA initialization
        if distribution == 'pca':
            # Collect all training data
            train_data = []
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[1]
                else:
                    images = batch
                train_data.append(images)
            train_data = torch.cat(train_data, dim=0).to(self.device)
            
            # Flatten images for PCA
            n_samples = train_data.shape[0]
            train_data_flat = train_data.reshape(n_samples, -1)
            
            # Ensure float dtype for PCA
            if train_data_flat.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                train_data_flat = train_data_flat.float()
            
            # Add PCA-specific parameters
            dist_params['data'] = train_data_flat
            # Get PCA params from config if present, otherwise use defaults
            config_dist_params = model_config.representation.get('dist_params', {})
            dist_params['whiten'] = config_dist_params.get('whiten', False)
            dist_params['svd_solver'] = config_dist_params.get('svd_solver', 'auto')
        else:
            # Handle standard distribution parameters
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
            dist=distribution,
            dist_params=dist_params,
            device=self.device
        )
        
        if distribution == 'pca' and self.verbose:
            explained_var = rep._options.get('explained_variance_ratio', None)
            if explained_var:
                total_var = sum(explained_var) * 100
                print(f"   PCA explained variance: {total_var:.2f}% (top {len(explained_var)} components)")
                print(f"   Per component: {[f'{v*100:.2f}%' for v in explained_var[:5]]}")
        
        # Create test representation layer
        # For PCA initialization, we use normal distribution for test set
        # since we can't apply the same PCA transform (different sample size)
        if distribution == 'pca':
            if self.verbose:
                print("   Initializing test representations with normal distribution...")
            test_rep = RepresentationLayer(
                dim=model_config.representation.n_features,
                n_samples=nsample_test,
                dist='normal',  # Use normal for test set
                dist_params={},
                device=self.device
            )
        else:
            test_rep = RepresentationLayer(
                dim=model_config.representation.n_features,
                n_samples=nsample_test,
                dist=distribution,
                dist_params=dist_params,
                device=self.device
            )
        
        # Create decoder
        decoder = ConvDecoder(
            latent_dim=model_config.representation.n_features,
            hidden_dims=model_config.decoder.hidden_dims,
            output_channels=model_config.decoder.output_channels,
            output_size=model_config.decoder.output_size,
            activation=model_config.decoder.activation,
            final_activation=model_config.decoder.final_activation,
            dropout_rate=model_config.decoder.dropout_rate,
            init_size=model_config.decoder.init_size
        ).to(self.device)
        
        # Create GMM
        gmm = GaussianMixture(
            n_components=model_config.gmm.n_components,
            n_features=model_config.representation.n_features,
            covariance_type=model_config.gmm.covariance_type,
            max_iter=model_config.gmm.max_iter,
            tol=model_config.gmm.tol,
            reg_covar=model_config.gmm.reg_covar,
            n_init=model_config.gmm.n_init,
            init_means=model_config.gmm.init_means,
            init_weights=model_config.gmm.init_weights,
            init_covariances=model_config.gmm.init_covariances,
            random_state=self.config.random_seed,
            warm_start=model_config.gmm.warm_start,
            cem=model_config.gmm.cem,
            weight_concentration_prior=model_config.gmm.weight_concentration_prior,
            mean_prior=model_config.gmm.mean_prior,
            mean_precision_prior=model_config.gmm.mean_precision_prior,
            covariance_prior=model_config.gmm.covariance_prior,
            degrees_of_freedom_prior=model_config.gmm.degrees_of_freedom_prior,
            verbose=model_config.gmm.verbose,
            verbose_interval=model_config.gmm.verbose_interval,
            device=self.device,
        )
        
        # Create full model
        model = DGD(decoder, rep, gmm)
        
        return model, rep, test_rep, gmm
    
    def _create_optimizers(self, model, rep, test_rep) -> List:
        """Create AdamW optimizers based on configuration."""
        training_config = self.training_config
        
        # Decoder optimizer (AdamW)
        decoder_config = training_config.optimizer.decoder
        decoder_optimizer = torch.optim.AdamW(
            model.decoder.parameters(),
            lr=decoder_config.lr,
            betas=tuple(decoder_config.betas),
            eps=decoder_config.eps,
            weight_decay=decoder_config.weight_decay,
            amsgrad=decoder_config.get('amsgrad', False)
        )
        
        # Representation optimizers (AdamW)
        rep_config = training_config.optimizer.representation
        trainrep_optimizer = torch.optim.AdamW(
            rep.parameters(),
            lr=rep_config.lr,
            betas=tuple(rep_config.betas),
            eps=rep_config.eps,
            weight_decay=rep_config.weight_decay,
            amsgrad=rep_config.get('amsgrad', False)
        )
        
        testrep_optimizer = torch.optim.AdamW(
            test_rep.parameters(),
            lr=rep_config.lr,
            betas=tuple(rep_config.betas),
            eps=rep_config.eps,
            weight_decay=rep_config.weight_decay,
            amsgrad=rep_config.get('amsgrad', False)
        )
        
        return [decoder_optimizer, trainrep_optimizer, testrep_optimizer]
    
    def _plot_reconstructions(self, model, rep, test_rep, epoch: int) -> None:
        """Generate and plot reconstructions for sample data."""
        model.decoder.eval()
        with torch.no_grad():
            # Unpack sample data
            indices_train, images_train, labels_train, indices_test, images_test, labels_test = self.sample_data
            
            # Move to device
            indices_train = indices_train.to(self.device)
            images_train = images_train.to(self.device)
            indices_test = indices_test.to(self.device)
            images_test = images_test.to(self.device)
            
            # Generate reconstructions
            z_train = rep(indices_train)
            recon_train = model.decoder(z_train)
            
            z_test = test_rep(indices_test)
            recon_test = model.decoder(z_test)
            
            # Plot train reconstructions
            plot_images_by_class(
                images=recon_train,
                labels=labels_train,
                class_names=self.class_names,
                title=f'Train: Reconstructed Images by Class - Epoch {epoch}',
                n_per_class=5,
                cmap='viridis'
            )
            plt.show()
            
            # Plot test reconstructions
            plot_images_by_class(
                images=recon_test,
                labels=labels_test,
                class_names=self.class_names,
                title=f'Test: Reconstructed Images by Class - Epoch {epoch}',
                n_per_class=5,
                cmap='viridis'
            )
            plt.show()
    
    def _create_schedulers(self, optimizers, total_epochs: int, steps_per_epoch: int) -> List:
        """Create learning rate schedulers using OneCycleLR.
        
        Note: Decoder steps per batch, representations step per epoch.
        Need different total_steps for each optimizer.
        """
        lr_config = self.training_config.lr_scheduler
        
        if not lr_config.get('enabled', False):
            return [None, None, None]
        
        # OneCycleLR parameters
        pct_start = lr_config.get('pct_start', 0.3)
        div_factor = lr_config.get('div_factor', 25.0)
        final_div_factor = lr_config.get('final_div_factor', 10000.0)
        anneal_strategy = lr_config.get('anneal_strategy', 'cos')
        cycle_momentum = lr_config.get('cycle_momentum', True)
        base_momentum = lr_config.get('base_momentum', 0.85)
        max_momentum = lr_config.get('max_momentum', 0.95)
        three_phase = lr_config.get('three_phase', False)
        
        schedulers = []
        for i, optimizer in enumerate(optimizers):
            # Get max learning rate from scheduler config or optimizer
            if i == 0:  # Decoder optimizer
                max_lr = lr_config.get('max_lr_decoder', None)
                if max_lr is None:
                    max_lr = optimizer.param_groups[0]['lr']
            else:  # Representation optimizers (trainrep and testrep)
                max_lr = lr_config.get('max_lr_representation', None)
                if max_lr is None:
                    max_lr = optimizer.param_groups[0]['lr']
            
            # Decoder (index 0) steps per batch, representations (indices 1, 2) step per epoch
            if i == 0:  # Decoder optimizer
                total_steps = total_epochs * steps_per_epoch
            else:  # Representation optimizers (trainrep and testrep)
                total_steps = total_epochs
            
            # Create OneCycleLR scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
                three_phase=three_phase
            )
            
            schedulers.append(scheduler)
        
        return schedulers
    
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
        self.sample_data = sample_data
        self.class_names = class_names
        
        # Create model components
        model, rep, test_rep, gmm = self._create_model_components(train_loader, test_loader)
        
        # Collect training labels in correct index order for visualization
        # We need labels aligned with representation indices (0 to n_samples-1)
        n_train_samples = len(train_loader.dataset)
        train_labels = torch.zeros(n_train_samples, dtype=torch.long)
        
        # Collect labels indexed correctly (not in batch order!)
        for index, _, labels_batch in train_loader:
            train_labels[index] = labels_batch
        
        plot_latent_space(
            representations=rep.z.detach(),
            labels=train_labels,
            gmm=None,  # No GMM fitted yet
            class_names=class_names,
            title="Initial Latent Space (Before Training)",
            save_path=None,
            show=True,
            verbose=self.verbose
        )
        
        # Plot initial reconstructions
        self._plot_reconstructions(model, rep, test_rep, epoch=0)
        
        # Create optimizers
        optimizers = self._create_optimizers(model, rep, test_rep)
        decoder_optimizer, trainrep_optimizer, testrep_optimizer = optimizers
        
        # Create learning rate schedulers
        steps_per_epoch = len(train_loader)
        schedulers = self._create_schedulers(optimizers, self.training_config.epochs, steps_per_epoch)
        decoder_scheduler, trainrep_scheduler, testrep_scheduler = schedulers
        
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
        
        # Training loop
        start_time = time.time()

        for epoch in range(1, self.training_config.epochs + 1):
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
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=1000 if epoch == first_epoch_gmm else 100)
                
                # Plot latent space with GMM overlay
                plot_latent_space(
                    representations=representations,
                    labels=train_labels,
                    gmm=gmm,
                    class_names=class_names,
                    title=f"Latent Space - Epoch {epoch} (GMM Fitted)",
                    save_path=None,
                    show=True,
                    verbose=self.verbose
                )
                
                # Plot reconstructions after GMM fit
                self._plot_reconstructions(model, rep, test_rep, epoch=epoch)
                
                # Activate early stopping after first GMM fit
                if epoch == first_epoch_gmm:
                    self.early_stopping_active = True
                    self.best_train_loss = float('inf')  # Reset best loss
                    self.best_test_loss = float('inf')  # Reset test loss too (GMM adds error term)
                    self.epochs_without_improvement = 0
            elif epoch > first_epoch_gmm:
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

                # Latent Space Noise Injection (regularization during training)
                if model.decoder.training and self.training_config.latent_noise_scale > 0:
                    noise = torch.randn_like(z) * self.training_config.latent_noise_scale
                    z = z + noise

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
                
                # Step learning rate schedulers (OneCycleLR steps per batch)
                if decoder_scheduler is not None:
                    decoder_scheduler.step()
                
                # Track losses
                train_loss += loss.item()
                recon_train_loss += recon_loss.item()
                if epoch >= first_epoch_gmm:
                    gmm_train_loss += gmm_error.item()
            
            trainrep_optimizer.step()
            
            # Step trainrep scheduler
            if trainrep_scheduler is not None:
                trainrep_scheduler.step()
            
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
            
            # Step testrep scheduler  
            if testrep_scheduler is not None:
                testrep_scheduler.step()
            
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
            
            # Track learning rates (decoder LR)
            self.learning_rates.append(decoder_optimizer.param_groups[0]['lr'])
            # Track momentum (beta_1)
            self.momentum_betas.append(decoder_optimizer.param_groups[0]['betas'][0])
            
            # Update best losses and early stopping
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.best_epoch = epoch
                # Reset early stopping counter when test loss improves
                if self.early_stopping_active:
                    self.epochs_without_improvement = 0
            elif self.early_stopping_active:
                # Increment counter only if test loss didn't improve
                self.epochs_without_improvement += 1
            if recon_train_loss < self.best_recon_train:
                self.best_recon_train = recon_train_loss
            if recon_test_loss < self.best_recon_test:
                self.best_recon_test = recon_test_loss
            if epoch >= first_epoch_gmm:
                if gmm_train_loss < self.best_gmm_train:
                    self.best_gmm_train = gmm_train_loss
                if gmm_test_loss < self.best_gmm_test:
                    self.best_gmm_test = gmm_test_loss
            
            # Calculate timing
            epoch_duration = time.time() - epoch_start_time
            self.epoch_times.append(epoch_duration)
            
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.training_config.epochs - epoch
            estimated_time_remaining = remaining_epochs * avg_epoch_time
            
            epoch_time_str = str(timedelta(seconds=int(epoch_duration)))
            remaining_time_str = str(timedelta(seconds=int(estimated_time_remaining)))
            
            # Get current learning rates
            lr_decoder = decoder_optimizer.param_groups[0]['lr']
            lr_rep = trainrep_optimizer.param_groups[0]['lr']
            
            # Format GMM losses for display
            gmm_train_str = f"{gmm_train_loss:.4f} (B: {self.best_gmm_train:.4f})" if epoch >= first_epoch_gmm else "0.0000"
            gmm_test_str = f"{gmm_test_loss:.4f} (B: {self.best_gmm_test:.4f})" if epoch >= first_epoch_gmm else "0.0000"
            
            print(f"Epoch {epoch}/{self.training_config.epochs} [Time per Epoch: {epoch_time_str}, Remaining Time: {remaining_time_str}, LR: Dec={lr_decoder:.2e}, Rep={lr_rep:.2e}]")
            print(f"       - Train Loss: {train_loss:.4f} (B: {self.best_train_loss:.4f}), Recon: {recon_train_loss:.4f} (B: {self.best_recon_train:.4f}), GMM: {gmm_train_str}")
            print(f"       - Test  Loss: {test_loss:.4f} (B: {self.best_test_loss:.4f}), Recon: {recon_test_loss:.4f} (B: {self.best_recon_test:.4f}), GMM: {gmm_test_str}")

            # Check for early stopping (only if active)
            early_stopping_patience = getattr(self.training_config, 'early_stopping_patience', None)
            if self.early_stopping_active and early_stopping_patience is not None and self.epochs_without_improvement >= early_stopping_patience:
                if self.verbose:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"   No improvement in test loss for {early_stopping_patience} consecutive epochs (since GMM activation)")
                    print(f"   Best test loss: {self.best_test_loss:.4f} at epoch {self.best_epoch}")
                break
        
        # Final GMM fit after training completes (for best generative model)
        final_gmm_fit = getattr(self.training_config, 'final_gmm_fit', False)
        if final_gmm_fit:
            if self.verbose:
                print(f"\nPerforming final GMM fit for optimal generative model...")
            with torch.no_grad():
                representations = rep.z.detach()
                gmm.fit(representations, max_iter=self.config.model.gmm.max_iter)
            if self.verbose:
                print(f"   Final GMM converged: {gmm.converged_} (iterations: {gmm.n_iter_})")
        
        # Training complete
        total_time = time.time() - start_time
        if self.verbose:
            print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
            print(f"Final training loss: {self.train_losses[-1]:.4f}")
            print(f"Final test loss: {self.test_losses[-1]:.4f}")
            print(f"Best train loss: {self.best_train_loss:.4f} at epoch {self.best_epoch}")
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
            "final_test_loss": self.test_losses[-1],
            "best_train_loss": self.best_train_loss,
            "best_epoch": self.best_epoch,
            "stopped_early": self.epochs_without_improvement >= getattr(self.training_config, 'early_stopping_patience', float('inf'))
        }

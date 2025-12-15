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

# Import tgmm package for Gaussian Mixture Model
from tgmm import GaussianMixture

from ..models import RepresentationLayer, DGD, ConvDecoder


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
        
        # Initialize ClearML task only if enabled in config
        self.task = None
        if hasattr(config, 'clearml') and config.clearml.enabled:
            self.task = Task.current_task()
            if self.task is None:
                # If no task is already initialized, create one
                project_name = config.clearml.get('project_name', 'ImageDGD')
                task_name = config.clearml.get('task_name', 'DGD Training')
                self.task = Task.init(project_name=project_name, task_name=task_name)
        
        # Initialize tracking lists
        self.train_losses = []
        self.test_losses = []
        self.gmm_train_losses = []
        self.gmm_test_losses = []
        self.recon_train_losses = []
        self.recon_test_losses = []
        
        # Timing
        self.epoch_times = []
        
        # Early stopping (based on training loss)
        # Note: Early stopping only starts after GMM is fitted to avoid false triggers
        self.best_train_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.early_stopping_active = False  # Only activate after first GMM fit
        
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
    
    def _create_schedulers(self, optimizers, total_epochs: int) -> List:
        """Create learning rate schedulers with warmup and cosine annealing."""
        lr_config = self.training_config.lr_scheduler
        
        if not lr_config.get('enabled', False):
            return [None, None, None]
        
        warmup_epochs = lr_config.get('warmup_epochs', 0)
        eta_min = lr_config.get('eta_min', 0)
        
        schedulers = []
        for optimizer in optimizers:
            # Get initial learning rate
            initial_lr = optimizer.param_groups[0]['lr']
            
            # Create cosine annealing scheduler for epochs after warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=eta_min
            )
            
            if warmup_epochs > 0:
                # Create linear warmup scheduler
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,  # Start at 1% of initial LR
                    end_factor=1.0,     # End at 100% of initial LR
                    total_iters=warmup_epochs
                )
                
                # Chain warmup then cosine annealing
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = cosine_scheduler
            
            schedulers.append(scheduler)
        
        return schedulers
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics to ClearML."""
        if self.task is None:
            return
        logger = self.task.get_logger()
        for metric_name, value in metrics.items():
            logger.report_scalar("metrics", metric_name, value, iteration=epoch)
    
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
        
        # Create learning rate schedulers
        schedulers = self._create_schedulers(optimizers, self.training_config.epochs)
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
        
        # ClearML logging
        if self.task is not None:
            self.task.connect(self.config)  # Connect the configuration
            self.task.set_parameters({
                "decoder_params": decoder_params,
                "rep_params": rep_params,
                "test_rep_params": test_rep_params,
                "total_params": decoder_params + rep_params + test_rep_params
            })
        
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
                print(f"Fitting GMM at epoch {epoch}...")
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=1000 if epoch == first_epoch_gmm else 100)
                
                # Activate early stopping after first GMM fit
                if epoch == first_epoch_gmm:
                    self.early_stopping_active = True
                    self.best_train_loss = float('inf')  # Reset best loss
                    self.epochs_without_improvement = 0
                    if self.verbose:
                        print(f"   Early stopping activated from epoch {epoch} onwards")
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

                # TODO: Add noise here

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
            
            # Step learning rate schedulers
            if decoder_scheduler is not None:
                decoder_scheduler.step()
            if trainrep_scheduler is not None:
                trainrep_scheduler.step()
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
            
            # Early stopping check (based on training loss)
            # Only check if early stopping is active (after first GMM fit)
            if self.early_stopping_active:
                if train_loss < self.best_train_loss:
                    self.best_train_loss = train_loss
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
            
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
                "estimated_time_remaining": estimated_time_remaining,
                "lr_decoder": decoder_optimizer.param_groups[0]['lr'],
                "lr_representation": trainrep_optimizer.param_groups[0]['lr']
            }
            self._log_metrics(epoch, metrics)
            
            epoch_time_str = str(timedelta(seconds=int(epoch_duration)))
            remaining_time_str = str(timedelta(seconds=int(estimated_time_remaining)))
            train_loss_improv = self._calc_improvement(self.train_losses)
            test_loss_improv = self._calc_improvement(self.test_losses)
            train_recon_improv = self._calc_improvement(self.recon_train_losses)
            test_recon_improv = self._calc_improvement(self.recon_test_losses)
            train_gmm_improv = self._calc_improvement(self.gmm_train_losses) if epoch >= first_epoch_gmm else 0.0
            test_gmm_improv = self._calc_improvement(self.gmm_test_losses) if epoch >= first_epoch_gmm else 0.0
            
            # Get current learning rates
            lr_decoder = decoder_optimizer.param_groups[0]['lr']
            lr_rep = trainrep_optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch}/{self.training_config.epochs} [Time per Epoch: {epoch_time_str}, Remaining Time: {remaining_time_str}, LR: Dec={lr_decoder:.2e}, Rep={lr_rep:.2e}]")
            print(f"       - Train Loss: {train_loss:.4f} ({train_loss_improv:+.2f}%), Recon: {recon_train_loss:.4f} ({train_recon_improv:+.2f}%), GMM: {gmm_train_loss:.4f} ({train_gmm_improv:+.2f}%)")
            print(f"       - Test  Loss: {test_loss:.4f} ({test_loss_improv:+.2f}%), Recon: {recon_test_loss:.4f} ({test_recon_improv:+.2f}%), GMM: {gmm_test_loss:.4f} ({test_gmm_improv:+.2f}%)")

            # Check for early stopping (only if active)
            early_stopping_patience = getattr(self.training_config, 'early_stopping_patience', None)
            if self.early_stopping_active and early_stopping_patience is not None and self.epochs_without_improvement >= early_stopping_patience:
                if self.verbose:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"   No improvement in train loss for {early_stopping_patience} consecutive epochs (since GMM activation)")
                    print(f"   Best train loss: {self.best_train_loss:.4f} at epoch {self.best_epoch}")
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

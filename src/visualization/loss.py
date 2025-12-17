"""
Visualization functions for training loss and dynamics analysis.
"""
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from omegaconf import DictConfig


def plot_training_analysis(
    train_losses: List[float],
    test_losses: List[float],
    trainer: Any,
    config: DictConfig,
    skip_first_epoch: bool = True
) -> None:
    """
    Plot comprehensive training analysis including total losses, reconstruction losses, and GMM losses.
    
    Parameters
    ----------
    train_losses : List[float]
        Total training losses per epoch
    test_losses : List[float]
        Total test losses per epoch
    trainer : DGDTrainer
        Trainer object containing detailed loss tracking
    config : DictConfig
        Configuration object containing training parameters
    skip_first_epoch : bool, default=True
        Whether to skip the first epoch in plots (often has initialization artifacts)
    """
    # Plot comprehensive training analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('DGD Training Analysis', fontsize=16, fontweight='bold')
    
    # Determine starting epoch for plotting
    start_epoch = 2 if skip_first_epoch else 1
    start_idx = 1 if skip_first_epoch else 0
    
    # 1. Training and Test Loss
    axes[0].plot(
        range(start_epoch, len(train_losses) + 1), 
        train_losses[start_idx:], 
        'b-', 
        label='Train Loss', 
        linewidth=2
    )
    axes[0].plot(
        range(start_epoch, len(test_losses) + 1), 
        test_losses[start_idx:], 
        'r-', 
        label='Test Loss', 
        linewidth=2
    )
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Reconstruction Losses
    recon_train_losses = trainer.recon_train_losses
    recon_test_losses = trainer.recon_test_losses
    axes[1].plot(
        range(start_epoch, len(recon_train_losses) + 1), 
        recon_train_losses[start_idx:], 
        'g-', 
        label='Train Reconstruction', 
        linewidth=2
    )
    axes[1].plot(
        range(start_epoch, len(recon_test_losses) + 1), 
        recon_test_losses[start_idx:], 
        'orange', 
        label='Test Reconstruction', 
        linewidth=2
    )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. GMM Losses (after GMM starts)
    gmm_train_losses = trainer.gmm_train_losses
    gmm_test_losses = trainer.gmm_test_losses
    gmm_start_epoch = config.training.first_epoch_gmm
    
    if len(gmm_train_losses) > 0 and any(x != 0 for x in gmm_train_losses):
        non_zero_epochs = [i+1 for i, x in enumerate(gmm_train_losses) if x != 0 and i >= start_idx]
        non_zero_train = [x for i, x in enumerate(gmm_train_losses) if x != 0 and i >= start_idx]
        non_zero_test = [gmm_test_losses[i] for i, x in enumerate(gmm_train_losses) if x != 0 and i >= start_idx]
        
        axes[2].plot(non_zero_epochs, non_zero_train, 'purple', label='Train GMM Loss', linewidth=2)
        axes[2].plot(non_zero_epochs, non_zero_test, 'brown', label='Test GMM Loss', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('GMM Loss')
        axes[2].set_title(f'GMM Loss (starts epoch {gmm_start_epoch})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'GMM not fitted yet\nor no GMM loss', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('GMM Loss')
    
    plt.tight_layout()
    plt.show()


def plot_training_dynamics(
    trainer: Any,
    skip_first_epoch: bool = True
) -> None:
    """
    Plot training dynamics including learning rate schedule, momentum, and epoch timing.
    
    Parameters
    ----------
    trainer : DGDTrainer
        Trainer object containing dynamics tracking (learning_rates, momentum_betas, epoch_times)
    skip_first_epoch : bool, default=True
        Whether to skip the first epoch in plots (often has initialization artifacts)
    """
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 5))
    fig2.suptitle('Training Dynamics', fontsize=16, fontweight='bold')
    
    # Determine starting epoch for plotting
    start_epoch = 2 if skip_first_epoch else 1
    start_idx = 1 if skip_first_epoch else 0
    
    # 1. Learning Rate
    learning_rates = trainer.learning_rates
    axes2[0].plot(
        range(start_epoch, len(learning_rates) + 1), 
        learning_rates[start_idx:], 
        'c-', 
        label='Learning Rate', 
        linewidth=2
    )
    axes2[0].set_xlabel('Epoch')
    axes2[0].set_ylabel('Learning Rate')
    axes2[0].set_title('Learning Rate Schedule')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_yscale('log')
    
    # 2. Momentum (beta_1)
    if hasattr(trainer, 'momentum_betas') and len(trainer.momentum_betas) > 1:
        momentum_betas = trainer.momentum_betas
        axes2[1].plot(
            range(start_epoch, len(momentum_betas) + 1), 
            momentum_betas[start_idx:], 
            'g-', 
            label='Beta_1 (Momentum)', 
            linewidth=2
        )
        axes2[1].set_xlabel('Epoch')
        axes2[1].set_ylabel('Beta_1')
        axes2[1].set_title('Momentum Schedule (OneCycleLR)')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
    else:
        axes2[1].text(0.5, 0.5, 'Momentum tracking\nnot available', ha='center', va='center', transform=axes2[1].transAxes)
        axes2[1].set_title('Momentum Schedule')
    
    # 3. Time per Epoch
    epoch_times = trainer.epoch_times
    axes2[2].plot(
        range(start_epoch, len(epoch_times) + 1), 
        epoch_times[start_idx:], 
        'm-', 
        label='Time per Epoch', 
        linewidth=2
    )
    axes2[2].set_xlabel('Epoch')
    axes2[2].set_ylabel('Time (seconds)')
    axes2[2].set_title('Time per Epoch')
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

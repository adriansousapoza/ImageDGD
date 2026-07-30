"""
Visualization functions for training loss and dynamics analysis.
"""
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from omegaconf import DictConfig


def plot_training_analysis(
    train_losses: List[float],
    val_losses: List[float],
    trainer: Any,
    config: DictConfig,
    skip_first_epoch: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comprehensive training analysis including total losses, reconstruction losses, GMM losses, and clustering metrics.

    Parameters
    ----------
    train_losses : List[float]
        Total training losses per epoch
    val_losses : List[float]
        Total validation losses per epoch
    trainer : DGDTrainer
        Trainer object containing detailed loss tracking
    config : DictConfig
        Configuration object containing training parameters
    skip_first_epoch : bool, default=True
        Whether to skip the first epoch in plots (often has initialization artifacts)
    save_path : Optional path to save the figure
    show : Whether to display the figure
    """
    # Plot comprehensive training analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DGD Training Analysis', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Determine starting epoch for plotting
    start_epoch = 2 if skip_first_epoch else 1
    start_idx = 1 if skip_first_epoch else 0

    # Get GMM start epoch for vertical line
    gmm_start_epoch = config.training.first_epoch_gmm

    # 1. Training and Validation Loss
    axes[0].plot(
        range(start_epoch, len(train_losses) + 1),
        train_losses[start_idx:],
        'b-',
        label='Train Loss',
        linewidth=2
    )
    axes[0].plot(
        range(start_epoch, len(val_losses) + 1),
        val_losses[start_idx:],
        'r-',
        label='Val Loss',
        linewidth=2
    )
    # Add vertical line at GMM start epoch
    axes[0].axvline(x=gmm_start_epoch, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'GMM starts (epoch {gmm_start_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Reconstruction Losses
    recon_train_losses = trainer.recon_train_losses
    recon_val_losses = trainer.recon_val_losses
    axes[1].plot(
        range(start_epoch, len(recon_train_losses) + 1),
        recon_train_losses[start_idx:],
        'g-',
        label='Train Reconstruction',
        linewidth=2
    )
    axes[1].plot(
        range(start_epoch, len(recon_val_losses) + 1),
        recon_val_losses[start_idx:],
        'orange',
        label='Val Reconstruction',
        linewidth=2
    )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. GMM Losses (after GMM starts)
    gmm_train_losses = trainer.gmm_train_losses
    gmm_val_losses = trainer.gmm_val_losses

    if len(gmm_train_losses) > 0 and any(x != 0 for x in gmm_train_losses):
        non_zero_epochs = [i+1 for i, x in enumerate(gmm_train_losses) if x != 0 and i >= start_idx]
        non_zero_train = [x for i, x in enumerate(gmm_train_losses) if x != 0 and i >= start_idx]
        non_zero_val = [gmm_val_losses[i] for i, x in enumerate(gmm_train_losses) if x != 0 and i >= start_idx]

        axes[2].plot(non_zero_epochs, non_zero_train, 'purple', label='Train GMM Loss', linewidth=2)
        axes[2].plot(non_zero_epochs, non_zero_val, 'brown', label='Val GMM Loss', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('GMM Loss')
        axes[2].set_title(f'GMM Loss (starts epoch {gmm_start_epoch})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'GMM not fitted yet\nor no GMM loss', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('GMM Loss')

    # 4. Clustering Quality (AMI & ARI)
    if hasattr(trainer, 'ami_scores') and len(trainer.ami_scores) > 0:
        ami_scores = trainer.ami_scores
        val_ami_scores = trainer.val_ami_scores if hasattr(trainer, 'val_ami_scores') else []
        ari_scores = trainer.ari_scores if hasattr(trainer, 'ari_scores') else []
        val_ari_scores = trainer.val_ari_scores if hasattr(trainer, 'val_ari_scores') else []

        # Find epochs where metrics were computed (non-zero GMM epochs).
        # NOT filtered by start_idx here: ami_scores/ari_scores gain one
        # entry per GMM-active epoch unconditionally (trainer.py appends them
        # every time, regardless of skip_first_epoch), so their natural x-axis
        # is unfiltered too. Filtering here would desync the two whenever the
        # GMM is active starting at epoch 1 itself (metric_epochs would drop
        # epoch 1 while the score lists still include its entry).
        metric_epochs = [i+1 for i, x in enumerate(gmm_train_losses) if x != 0]

        def _align(epochs, values):
            """Trim both to the same length, keeping the most recent entries.

            metric_epochs and the *_scores lists are built from independent
            conditions in trainer.py and can end up a handful of entries
            apart at the start (e.g. a GMM refit that fires before
            first_epoch_gmm). Aligning from the end keeps every value
            correctly paired with its epoch instead of crashing or silently
            mis-pairing when lengths differ.
            """
            n = min(len(epochs), len(values))
            return epochs[-n:], values[-n:]

        ax_metrics = axes[3]

        if len(ami_scores) > 0:
            x, y = _align(metric_epochs, ami_scores)
            ax_metrics.plot(x, y, 'b-', label='Train AMI', linewidth=2, marker='o')
        if len(val_ami_scores) > 0:
            x, y = _align(metric_epochs, val_ami_scores)
            ax_metrics.plot(x, y, 'r-', label='Val AMI', linewidth=2, marker='o')
        if len(ari_scores) > 0:
            x, y = _align(metric_epochs, ari_scores)
            ax_metrics.plot(x, y, 'g--', label='Train ARI', linewidth=2, marker='s')
        if len(val_ari_scores) > 0:
            x, y = _align(metric_epochs, val_ari_scores)
            ax_metrics.plot(x, y, 'orange', linestyle='--', label='Val ARI', linewidth=2, marker='s')

        ax_metrics.set_xlabel('Epoch')
        ax_metrics.set_ylabel('Score')
        ax_metrics.set_title('Clustering Quality (AMI & ARI)')
        ax_metrics.legend(loc='best')
        ax_metrics.grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'No clustering metrics\navailable', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Clustering Quality (AMI & ARI)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_training_dynamics(
    trainer: Any,
    skip_first_epoch: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training dynamics including learning rate schedule, momentum, noise level, and epoch timing.

    Parameters
    ----------
    trainer : DGDTrainer
        Trainer object containing dynamics tracking (learning_rates, momentum_betas, epoch_times)
    skip_first_epoch : bool, default=True
        Whether to skip the first epoch in plots (often has initialization artifacts)
    save_path : Optional path to save the figure
    show : Whether to display the figure
    """
    import math
    import numpy as np

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Training Dynamics', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    axes2 = axes2.flatten()

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

    # 3. Noise Level (Cosine Annealing)
    if hasattr(trainer.training_config, 'latent_noise_scale') and trainer.training_config.latent_noise_scale > 0:
        noise_start = trainer.training_config.get('latent_noise_start', 1.0)
        noise_end = trainer.training_config.get('latent_noise_end', 0.01)
        total_epochs = len(learning_rates)

        # Calculate noise schedule for all epochs
        noise_schedule = []
        for epoch in range(1, total_epochs + 1):
            progress = (epoch - 1) / max(total_epochs - 1, 1)
            noise_scale = noise_end + (noise_start - noise_end) * 0.5 * (1 + math.cos(math.pi * progress))
            noise_schedule.append(noise_scale)

        axes2[2].plot(
            range(start_epoch, len(noise_schedule) + 1),
            noise_schedule[start_idx:],
            'orange',
            label='Noise Level',
            linewidth=2
        )
        axes2[2].set_xlabel('Epoch')
        axes2[2].set_ylabel('Noise Scale')
        axes2[2].set_title(f'Noise Schedule (Cosine Annealing: {noise_start:.2f} → {noise_end:.4f})')
        axes2[2].legend()
        axes2[2].grid(True, alpha=0.3)
    else:
        axes2[2].text(0.5, 0.5, 'No noise injection\nenabled', ha='center', va='center', transform=axes2[2].transAxes)
        axes2[2].set_title('Noise Schedule')

    # 4. Time per Epoch
    epoch_times = trainer.epoch_times
    axes2[3].plot(
        range(start_epoch, len(epoch_times) + 1),
        epoch_times[start_idx:],
        'm-',
        label='Time per Epoch',
        linewidth=2
    )
    axes2[3].set_xlabel('Epoch')
    axes2[3].set_ylabel('Time (seconds)')
    axes2[3].set_title('Time per Epoch')
    axes2[3].legend()
    axes2[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig2.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig2)

    return fig2


def plot_inference_analysis(
    step_losses: List[float],
    step_recon: List[float],
    step_gmm: List[float],
    step_noise: List[float],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot the M-step inference optimization (Algorithm 2): total loss,
    reconstruction loss, GMM error, and noise scale, each vs. optimization
    step m. Unlike plot_training_analysis there is no train/val split (a
    single representation layer is optimized) and no learning-rate/momentum/
    epoch-timing panel (Algorithm 2 has no LR schedule or per-epoch timing).

    Parameters
    ----------
    step_losses : List[float]
        Total loss per optimization step m
    step_recon : List[float]
        Reconstruction loss per optimization step m
    step_gmm : List[float]
        GMM error per optimization step m
    step_noise : List[float]
        Noise scale per optimization step m
    save_path : Optional path to save the figure
    show : Whether to display the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DGD Inference Analysis (Algorithm 2)', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    steps = range(1, len(step_losses) + 1)

    axes[0].plot(steps, step_losses, 'b-', label='Total Loss', linewidth=2)
    axes[0].set_xlabel('Step (m)')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, step_recon, 'g-', label='Reconstruction Loss', linewidth=2)
    axes[1].set_xlabel('Step (m)')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, step_gmm, 'purple', label='GMM Error', linewidth=2)
    axes[2].set_xlabel('Step (m)')
    axes[2].set_ylabel('GMM Error')
    axes[2].set_title('GMM Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(steps, step_noise, 'orange', label='Noise Scale', linewidth=2)
    axes[3].set_xlabel('Step (m)')
    axes[3].set_ylabel('Noise Scale')
    axes[3].set_title('Noise Schedule')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

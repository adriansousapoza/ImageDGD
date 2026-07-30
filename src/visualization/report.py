"""
Post-hoc figure generation from saved training checkpoints.

Regenerates the same latent-space/reconstruction/GMM-sample figures the old
inline training-loop plotting produced, but from disk after training
completes — training itself stays plot-free (see src/training/trainer.py).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from ..models import ConvDecoder
from ..utils.checkpoint import load_checkpoint
from .latent import plot_latent_space
from .image import plot_images_by_class, plot_generated_samples
from .loss import plot_training_analysis, plot_training_dynamics


def _epoch_sort_key(path: Path) -> int:
    return int(path.name.split('_')[1])


def _build_decoder_factory(model_config):
    def factory():
        return ConvDecoder(
            latent_dim=model_config.representation.n_features,
            hidden_dims=model_config.decoder.hidden_dims,
            output_channels=model_config.decoder.output_channels,
            output_size=model_config.decoder.output_size,
            activation=model_config.decoder.activation,
            final_activation=model_config.decoder.final_activation,
            dropout_rate=model_config.decoder.dropout_rate,
            init_size=model_config.decoder.init_size,
        )
    return factory


def _plot_checkpoint_figures(
    checkpoint, tag: str, class_names: List[str],
    train_labels: torch.Tensor, val_labels: torch.Tensor,
    sample_data: Tuple, figures_dir: Path, device: torch.device
) -> None:
    decoder = checkpoint['decoder']
    decoder.eval()
    rep, val_rep, gmm = checkpoint['rep'], checkpoint['val_rep'], checkpoint['gmm']

    plot_latent_space(
        representations=rep.z.detach(), labels=train_labels, gmm=gmm, class_names=class_names,
        title=f"Train Latent Space ({tag})",
        save_path=str(figures_dir / f"latent_train_{tag}.png"), show=False,
    )
    plot_latent_space(
        representations=val_rep.z.detach(), labels=val_labels, gmm=gmm, class_names=class_names,
        title=f"Val Latent Space ({tag})",
        save_path=str(figures_dir / f"latent_val_{tag}.png"), show=False,
    )

    indices_train, images_train, labels_train, indices_val, images_val, labels_val = sample_data
    with torch.no_grad():
        recon_train = decoder(rep(indices_train.to(device)))
        recon_val = decoder(val_rep(indices_val.to(device)))

    plot_images_by_class(
        images=recon_train, labels=labels_train, class_names=class_names,
        title=f"Train: Reconstructed Images by Class ({tag})", n_per_class=5, cmap='viridis',
        save_path=str(figures_dir / f"recon_train_{tag}.png"), show=False,
    )
    plot_images_by_class(
        images=recon_val, labels=labels_val, class_names=class_names,
        title=f"Val: Reconstructed Images by Class ({tag})", n_per_class=5, cmap='viridis',
        save_path=str(figures_dir / f"recon_val_{tag}.png"), show=False,
    )


def generate_training_figures(
    experiment_dir: Path,
    figures_dir: Path,
    class_names: List[str],
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    sample_data: Tuple,
    device: Optional[torch.device] = None,
) -> None:
    """
    Walk every saved epoch checkpoint under experiment_dir, plus the best/
    checkpoint, and write latent-space/reconstruction/GMM-sample/loss-curve
    PNGs under figures_dir. Never calls plt.show().
    """
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_dir = experiment_dir / "best"
    # The config.yaml copy for a run lives at the run folder's root (one
    # level above experiment_dir, which is that run's models/ subfolder),
    # not inside best/ -- written once by the calling notebook when the run
    # folder is created.
    run_dir = experiment_dir.parent
    config = OmegaConf.load(run_dir / "config.yaml")
    decoder_factory = _build_decoder_factory(config.model)

    # Loss curves and training dynamics, from the persisted history. Plotted
    # first (before the checkpoint walk below) so these PNGs are written and
    # survive even if load_checkpoint raises partway through a malformed
    # checkpoint directory during the walk.
    history = torch.load(best_dir / "training_results.pth", map_location="cpu")

    # Guard against pre-AMI/ARI-swap checkpoints (which have nmi_scores, or
    # even older ari_scores/silhouette_scores)
    if 'ami_scores' not in history:
        raise ValueError(
            f"{best_dir / 'training_results.pth'} predates the AMI+ARI metric swap "
            "(missing ami_scores). Re-run training to regenerate it."
        )

    trainer_view = SimpleNamespace(
        recon_train_losses=history['recon_train_losses'],
        recon_val_losses=history['recon_val_losses'],
        gmm_train_losses=history['gmm_train_losses'],
        gmm_val_losses=history['gmm_val_losses'],
        ami_scores=history['ami_scores'],
        val_ami_scores=history['val_ami_scores'],
        ari_scores=history['ari_scores'],
        val_ari_scores=history['val_ari_scores'],
        learning_rates=history['learning_rates'],
        momentum_betas=history['momentum_betas'],
        epoch_times=history['epoch_times'],
        training_config=config.training,
    )
    plot_training_analysis(
        history['train_losses'], history['val_losses'], trainer_view, config,
        save_path=str(figures_dir / "loss_curves.png"), show=False,
    )
    plot_training_dynamics(
        trainer_view,
        save_path=str(figures_dir / "training_dynamics.png"), show=False,
    )

    checkpoint_dirs = sorted(
        (experiment_dir / "checkpoints").glob("epoch_*"),
        key=_epoch_sort_key,
    )
    for checkpoint_dir in checkpoint_dirs:
        epoch = _epoch_sort_key(checkpoint_dir)
        checkpoint = load_checkpoint(checkpoint_dir, decoder_factory, device=device)
        _plot_checkpoint_figures(
            checkpoint, f"epoch{epoch:04d}", class_names, train_labels, val_labels,
            sample_data, figures_dir, device
        )

    best_checkpoint = load_checkpoint(best_dir, decoder_factory, device=device)
    _plot_checkpoint_figures(
        best_checkpoint, "best", class_names, train_labels, val_labels,
        sample_data, figures_dir, device
    )

    # GMM-component sample grid, best model only
    gmm = best_checkpoint['gmm']
    decoder = best_checkpoint['decoder']
    if gmm is not None:
        weights = gmm.weights_.detach().cpu().numpy()
        sorted_components = np.argsort(weights)[::-1]
        with torch.no_grad():
            for component_idx in sorted_components:
                component_idx = int(component_idx)
                z_samples, component_labels = gmm.sample(32, component=component_idx)
                generated_images = decoder(z_samples)
                plot_generated_samples(
                    generated_images, labels=component_labels,
                    title=f"GMM Component {component_idx} - Weight: {weights[component_idx]:.4f} - Generated Samples",
                    n_cols=8, cmap='viridis', denormalize=True, figsize=(16, 8),
                    save_path=str(figures_dir / f"gmm_component{component_idx:02d}_samples.png"), show=False,
                )

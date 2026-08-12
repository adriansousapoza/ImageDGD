"""
Post-hoc figure generation from saved training checkpoints.

Regenerates the same latent-space/reconstruction/GMM-sample figures the old
inline training-loop plotting produced, but from disk after training
completes -- training itself stays plot-free (see src/training/trainer.py).

Every figures_dir is organized into four subfolders:
  training/        per-epoch-checkpoint progression: latent (label-colored,
                    no GMM) and ground-truth+reconstruction grids for every
                    saved epoch, plus loss_curves.png / training_dynamics.png
  reconstructions/  ground-truth+reconstruction grids for the best/final
                    model only, one file per split (train/val/test)
  latent/<split>/   best/final-model latent-space diagnostics per split:
                    overview.png (label-colored), gmm_overview.png (all GMM
                    clusters), cluster{rank:02d}.png (one per component)
  samples/          GMM-component-conditioned generated samples, best/final
                    model only, ranked by descending component weight
"""

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from tgmm import ClusteringMetrics

from ..models import ConvDecoder
from ..utils.checkpoint import load_checkpoint
from ..utils.schedules import cosine_noise_schedule
from .latent import plot_latent_space, generate_latent_space_figures, plot_noise_comparison
from .image import plot_ground_truth_and_reconstructions_by_class, plot_generated_samples
from .loss import plot_training_analysis, plot_training_dynamics, plot_inference_analysis

# Fixed representation row used for the single-point noise-ball detail panel,
# reused across every checkpoint of a run so the ball is comparable over the
# course of training -- see plot_noise_comparison.
_NOISE_DETAIL_POINT_INDEX = 0


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


def _make_subdirs(figures_dir: Path) -> SimpleNamespace:
    dirs = SimpleNamespace(
        training=figures_dir / "training",
        reconstructions=figures_dir / "reconstructions",
        latent=figures_dir / "latent",
        samples=figures_dir / "samples",
    )
    for d in (dirs.training, dirs.reconstructions, dirs.latent, dirs.samples):
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _plot_epoch_checkpoint_figures(
    checkpoint, tag: str, epoch: int, training_config, class_names: List[str],
    train_labels: torch.Tensor, val_labels: torch.Tensor,
    sample_data: Tuple, training_dir: Path, device: torch.device
) -> None:
    """Per-epoch progression figures: label-colored latent + GT/recon grids
    + (if noise injection is enabled) noise-comparison diagnostics."""
    decoder = checkpoint['decoder']
    decoder.eval()
    rep, val_rep = checkpoint['rep'], checkpoint['val_rep']

    if training_config.latent_noise_enabled:
        noise_start = training_config.get('latent_noise_start', 1.0)
        noise_end = training_config.get('latent_noise_end', 0.01)
        noise_scale = cosine_noise_schedule(epoch, training_config.epochs, noise_start, noise_end)
        plot_noise_comparison(
            representations=rep.z.detach(), noise_scale=noise_scale, point_index=_NOISE_DETAIL_POINT_INDEX,
            title=f"Train Noise Injection ({tag}, σ={noise_scale:.4f})",
            save_path=str(training_dir / f"noise_train_{tag}.png"), show=False,
        )
        plot_noise_comparison(
            representations=val_rep.z.detach(), noise_scale=noise_scale, point_index=_NOISE_DETAIL_POINT_INDEX,
            title=f"Val Noise Injection ({tag}, σ={noise_scale:.4f})",
            save_path=str(training_dir / f"noise_val_{tag}.png"), show=False,
        )

    plot_latent_space(
        representations=rep.z.detach(), labels=train_labels, class_names=class_names,
        title=f"Train Latent Space ({tag})",
        save_path=str(training_dir / f"latent_train_{tag}.png"), show=False,
    )
    plot_latent_space(
        representations=val_rep.z.detach(), labels=val_labels, class_names=class_names,
        title=f"Val Latent Space ({tag})",
        save_path=str(training_dir / f"latent_val_{tag}.png"), show=False,
    )

    indices_train, images_train, labels_train, indices_val, images_val, labels_val = sample_data
    with torch.no_grad():
        recon_train = decoder(rep(indices_train.to(device)))
        recon_val = decoder(val_rep(indices_val.to(device)))

    plot_ground_truth_and_reconstructions_by_class(
        images=images_train, reconstructions=recon_train, labels=labels_train, class_names=class_names,
        title=f"Train: Ground Truth vs Reconstructed ({tag})", n_per_class=5, cmap='viridis',
        save_path=str(training_dir / f"recon_train_{tag}.png"), show=False,
    )
    plot_ground_truth_and_reconstructions_by_class(
        images=images_val, reconstructions=recon_val, labels=labels_val, class_names=class_names,
        title=f"Val: Ground Truth vs Reconstructed ({tag})", n_per_class=5, cmap='viridis',
        save_path=str(training_dir / f"recon_val_{tag}.png"), show=False,
    )


def _plot_best_model_figures(
    checkpoint, class_names: List[str],
    train_labels: torch.Tensor, val_labels: torch.Tensor,
    sample_data: Tuple, reconstructions_dir: Path, latent_dir: Path, device: torch.device
) -> None:
    """Best/final-model figures: rich GT/recon grids + full GMM cluster breakdown, per split."""
    decoder = checkpoint['decoder']
    decoder.eval()
    rep, val_rep, gmm = checkpoint['rep'], checkpoint['val_rep'], checkpoint['gmm']

    (latent_dir / "train").mkdir(parents=True, exist_ok=True)
    (latent_dir / "val").mkdir(parents=True, exist_ok=True)

    if gmm is not None:
        generate_latent_space_figures(
            representations=rep.z.detach(), labels=train_labels, gmm=gmm, class_names=class_names,
            save_dir=latent_dir / "train", title_prefix="Train ",
        )
        generate_latent_space_figures(
            representations=val_rep.z.detach(), labels=val_labels, gmm=gmm, class_names=class_names,
            save_dir=latent_dir / "val", title_prefix="Val ",
        )
    else:
        plot_latent_space(
            representations=rep.z.detach(), labels=train_labels, class_names=class_names,
            title="Train Latent Space (best)",
            save_path=str(latent_dir / "train" / "overview.png"), show=False,
        )
        plot_latent_space(
            representations=val_rep.z.detach(), labels=val_labels, class_names=class_names,
            title="Val Latent Space (best)",
            save_path=str(latent_dir / "val" / "overview.png"), show=False,
        )

    indices_train, images_train, labels_train, indices_val, images_val, labels_val = sample_data
    with torch.no_grad():
        recon_train = decoder(rep(indices_train.to(device)))
        recon_val = decoder(val_rep(indices_val.to(device)))

    plot_ground_truth_and_reconstructions_by_class(
        images=images_train, reconstructions=recon_train, labels=labels_train, class_names=class_names,
        title="Train: Ground Truth vs Reconstructed (best)", n_per_class=5, cmap='viridis',
        save_path=str(reconstructions_dir / "recon_train.png"), show=False,
    )
    plot_ground_truth_and_reconstructions_by_class(
        images=images_val, reconstructions=recon_val, labels=labels_val, class_names=class_names,
        title="Val: Ground Truth vs Reconstructed (best)", n_per_class=5, cmap='viridis',
        save_path=str(reconstructions_dir / "recon_val.png"), show=False,
    )


def _plot_gmm_component_samples(gmm, decoder, samples_dir: Path, device: torch.device) -> None:
    """Write one GMM-component sample grid PNG per component, ranked by descending weight."""
    if gmm is None:
        return
    weights = gmm.weights_.detach().cpu().numpy()
    rank_order = np.argsort(weights)[::-1]
    with torch.no_grad():
        for rank, component_idx in enumerate(rank_order, start=1):
            component_idx = int(component_idx)
            z_samples, component_labels = gmm.sample(32, component=component_idx)
            generated_images = decoder(z_samples)
            plot_generated_samples(
                generated_images, labels=component_labels,
                title=f"Cluster {rank} (raw idx {component_idx}, weight={weights[component_idx]:.4f}) - Generated Samples",
                n_cols=8, cmap='viridis', denormalize=False, figsize=(16, 8),
                save_path=str(samples_dir / f"component_rank{rank:02d}_samples.png"), show=False,
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
    PNGs under figures_dir (see module docstring for the subfolder layout).
    Never calls plt.show().
    """
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir)
    dirs = _make_subdirs(figures_dir)
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
        save_path=str(dirs.training / "loss_curves.png"), show=False,
    )
    plot_training_dynamics(
        trainer_view,
        save_path=str(dirs.training / "training_dynamics.png"), show=False,
    )

    checkpoint_dirs = sorted(
        (experiment_dir / "checkpoints").glob("epoch_*"),
        key=_epoch_sort_key,
    )
    for checkpoint_dir in checkpoint_dirs:
        epoch = _epoch_sort_key(checkpoint_dir)
        checkpoint = load_checkpoint(checkpoint_dir, decoder_factory, device=device)
        _plot_epoch_checkpoint_figures(
            checkpoint, f"epoch{epoch:04d}", epoch, config.training, class_names, train_labels, val_labels,
            sample_data, dirs.training, device
        )

    best_checkpoint = load_checkpoint(best_dir, decoder_factory, device=device)
    _plot_best_model_figures(
        best_checkpoint, class_names, train_labels, val_labels,
        sample_data, dirs.reconstructions, dirs.latent, device
    )

    # GMM-component sample grid, best model only
    _plot_gmm_component_samples(best_checkpoint['gmm'], best_checkpoint['decoder'], dirs.samples, device)


def generate_inference_figures(
    figures_dir: Path,
    decoder,
    gmm,
    test_rep,
    test_labels: torch.Tensor,
    class_names: List[str],
    sample_data: Tuple,
    step_history: dict,
    device: torch.device,
    noise_snapshots: Optional[List[dict]] = None,
) -> Tuple[float, float]:
    """
    Write latent-space, reconstruction, loss-curve, and GMM-component-sample
    figures for one inference run (Algorithm 2), mirroring
    generate_training_figures' output categories but for a single optimized
    representation layer instead of a train/val pair across many
    checkpoints. Never calls plt.show().

    Parameters
    ----------
    figures_dir : Path
        Directory to write PNGs into (subfolders created if missing)
    decoder : ConvDecoder
        Frozen decoder used during inference
    gmm : GaussianMixture
        Frozen GMM used during inference
    test_rep : RepresentationLayer
        The optimized test representation layer
    test_labels : torch.Tensor
        True labels for the test split
    class_names : List[str]
        Class names for plot legends
    sample_data : Tuple
        (indices, images, labels) 3-tuple for the test split, from
        collect_class_samples -- unlike generate_training_figures' 6-tuple,
        there is only one split here.
    step_history : dict
        Dict with keys 'loss', 'recon', 'gmm', 'noise', each a list of
        per-step values collected during the M-step optimization loop.
    device : torch.device
    noise_snapshots : Optional[List[dict]]
        In-memory z snapshots captured during the M-step optimization loop
        (every 50 steps -- inference has no per-step disk checkpointing, so
        these come from the caller's own loop instead of being reloaded from
        disk like generate_training_figures' epoch checkpoints). Each dict
        has keys 'step', 'z', 'noise_scale'. One noise-comparison figure is
        written per snapshot. None/empty skips this figure set.

    Returns
    -------
    Tuple[float, float]
        (test_ami, test_ari), computed from the clean test_rep.z against the
        frozen gmm's predictions -- the same values used in the latent-space
        plot title.
    """
    figures_dir = Path(figures_dir)
    dirs = _make_subdirs(figures_dir)

    with torch.no_grad():
        predicted_labels = gmm.predict(test_rep.z.detach())

    cluster_metrics = ClusteringMetrics()
    test_ami = cluster_metrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    test_ari = cluster_metrics.adjusted_rand_score(test_labels, predicted_labels)

    generate_latent_space_figures(
        representations=test_rep.z.detach(), labels=test_labels, gmm=gmm, class_names=class_names,
        save_dir=dirs.latent / "test",
        title_prefix=f"Test (AMI: {test_ami:.4f}, ARI: {test_ari:.4f}) ",
    )

    indices_test, images_test, labels_test = sample_data
    with torch.no_grad():
        recon_test = decoder(test_rep(indices_test.to(device)))

    plot_ground_truth_and_reconstructions_by_class(
        images=images_test, reconstructions=recon_test, labels=labels_test, class_names=class_names,
        title="Test: Ground Truth vs Reconstructed (Algorithm 2 inference)", n_per_class=5, cmap='viridis',
        save_path=str(dirs.reconstructions / "recon_test.png"), show=False,
    )

    plot_inference_analysis(
        step_history['loss'], step_history['recon'], step_history['gmm'], step_history['noise'],
        save_path=str(dirs.training / "loss_curve.png"), show=False,
    )

    if noise_snapshots:
        for snapshot in noise_snapshots:
            plot_noise_comparison(
                representations=snapshot['z'], noise_scale=snapshot['noise_scale'],
                point_index=_NOISE_DETAIL_POINT_INDEX,
                title=f"Test Noise Injection (step {snapshot['step']:04d}, σ={snapshot['noise_scale']:.4f})",
                save_path=str(dirs.training / f"noise_test_step{snapshot['step']:04d}.png"), show=False,
            )

    _plot_gmm_component_samples(gmm, decoder, dirs.samples, device)

    return test_ami, test_ari

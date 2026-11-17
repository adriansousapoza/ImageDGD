"""
Post-hoc figure generation from saved training checkpoints.

Regenerates the same latent-space/reconstruction/GMM-sample figures the old
inline training-loop plotting produced, but from disk after training
completes -- training itself stays plot-free (see src/training/trainer.py).

Every figures_dir is organized into four subfolders:
  training/        per-epoch-checkpoint progression, one kind-subfolder per
                    figure type -- latent_train/, latent_val/, noise_train/,
                    noise_val/, recon_train/, recon_val/ (plus latent_test/,
                    noise_test/, recon_test/ from generate_inference_figures)
                    -- each holding one PNG per saved epoch/step
                    (epoch0160.png, ...) plus that kind's own
                    progression.gif/progression.mp4, stitched in epoch/step
                    order from those same PNGs (see animate.py). Also
                    directly under training/: loss_curves.png /
                    training_dynamics.png. latent_train/latent_val PNGs
                    overlay GMM component means (all three embeddings) and a
                    95%-confidence ellipse (PCA panel only) once a GMM has
                    been fit -- absent on earlier checkpoints, see
                    plot_latent_space. noise_train/noise_val PNGs fold in the
                    real per-epoch clean->noised->after-optimizer-step shift
                    (see plot_noise_comparison) as a third panel whenever
                    trainer.py wrote a z_trajectory.pth for that checkpoint.
                    Epoch checkpoints only exist if training.plot_interval
                    was set (see trainer.py) -- null (default) means no
                    intermediate checkpoints, so this progression set is
                    empty and only the best/final model gets plotted below.
  reconstructions/  ground-truth+reconstruction grids for the best/final
                    model only, one file per split (train/val/test)
  latent/<split>/   best/final-model latent-space diagnostics per split:
                    overview.png (label-colored), gmm_overview.png (all GMM
                    clusters), cluster{rank:02d}.png (one per component)
  samples/          GMM-component-conditioned generated samples, best/final
                    model only, ranked by descending component weight

model.representation.n_latent_copies (default 1): when > 1, each training
image has that many independently-optimized representation rows instead of
one (see trainer.py) -- rep.z/val_rep.z then have n_images*n_latent_copies
rows, ordered [image0_copy0, image0_copy1, ..., image1_copy0, ...]. Every
plotting call below that consumes a full rep.z/val_rep.z array is given
labels expanded the same way (labels.repeat_interleave(mult)) so per-row
coloring still lines up; anywhere a *specific* image's row is needed (the
fixed noise/zshift track points, the reconstruction-grid sample_data
indices), the row used is that image's canonical copy (index*mult, i.e.
copy 0) -- matching the canonical-only convention trainer.py itself uses for
the GMM fit and AMI/ARI, so those numbers stay comparable across n_latent_copies
values. Test-time inference (generate_inference_figures, Algorithm 2 in
dgd_test_inference.ipynb) always uses a fresh 1:1 representation layer
regardless of the training run's n_latent_copies -- multiplicity is a
training-time diagnostic only.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from tgmm import ClusteringMetrics

from ..models import ConvDecoder
from ..data.dataloader import pick_class_representative_indices
from ..utils.checkpoint import load_checkpoint
from ..utils.schedules import noise_schedule
from .latent import (
    plot_latent_space, generate_latent_space_figures, plot_noise_comparison,
)
from .image import plot_ground_truth_and_reconstructions_by_class, plot_generated_samples
from .loss import plot_training_analysis, plot_training_dynamics, plot_inference_analysis
from .animate import animate_from_pattern


def _epoch_sort_key(path: Path) -> int:
    return int(path.name.split('_')[1])


def _n_latent_copies(config) -> int:
    """model.representation.n_latent_copies, defaulting to 1 for configs
    written before the option existed."""
    return int(config.model.representation.get('n_latent_copies', 1))


def _canonical_rows(image_indices: torch.Tensor, mult: int) -> torch.Tensor:
    """Map per-image indices to their canonical (copy 0) row in a
    RepresentationLayer whose rows are [image0_copy0, image0_copy1, ...,
    image1_copy0, ...] -- a no-op when mult == 1."""
    return image_indices * mult if mult > 1 else image_indices


def _expand_labels(labels: torch.Tensor, mult: int) -> torch.Tensor:
    """Repeat each image's label mult times, matching the row order of a
    multiplicity-expanded RepresentationLayer -- a no-op when mult == 1."""
    return labels.repeat_interleave(mult) if mult > 1 else labels


def _kind_dir(training_dir: Path, kind: str) -> Path:
    """training/<kind>/ subfolder (e.g. latent_train, noise_val, recon_test),
    created on first use -- holds one PNG per checkpoint/step plus that
    kind's own progression.gif/progression.mp4 once animate_from_pattern runs."""
    d = training_dir / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    checkpoint, checkpoint_dir: Path, tag: str, epoch: int, training_config, class_names: List[str],
    train_labels: torch.Tensor, val_labels: torch.Tensor, random_seed: int,
    sample_data: Tuple, training_dir: Path, device: torch.device, mult: int = 1,
) -> None:
    """Per-epoch progression figures: GMM-overlaid latent + GT/recon grids +
    (if noise injection is enabled) noise-comparison diagnostics, folding in
    the real clean/noised/post-optimization shift as a third panel whenever
    z_trajectory.pth exists for this checkpoint.

    train_labels/val_labels are this split's canonical (one-per-image)
    labels -- expanded internally (see _expand_labels) wherever a plot
    consumes the full mult-expanded rep.z/val_rep.z array; track/sample
    indices are mapped to their canonical row (see _canonical_rows) wherever
    a *specific* image's representation is needed. See module docstring.
    """
    decoder = checkpoint['decoder']
    decoder.eval()
    rep, val_rep, gmm = checkpoint['rep'], checkpoint['val_rep'], checkpoint['gmm']
    train_labels_full = _expand_labels(train_labels, mult)
    val_labels_full = _expand_labels(val_labels, mult)

    if training_config.latent_noise_enabled:
        noise_start = training_config.get('latent_noise_start', 1.0)
        noise_end = training_config.get('latent_noise_end', 0.01)
        noise_scale = noise_schedule(
            epoch, training_config.epochs, noise_start, noise_end,
            schedule_type=training_config.get('latent_noise_schedule', 'cosine'),
            warmup_steps=training_config.get('latent_noise_warmup_epochs', 0),
        )
        # Same (deterministic given labels/n_classes/seed) points trainer.py
        # tracked for z_trajectory.pth -- re-derived here rather than read
        # back from that file so this still works even when it's absent
        # (e.g. epoch_0000, or an older checkpoint from before it existed).
        # These are per-image indices; mapped to their canonical row below
        # since rep.z/val_rep.z have mult rows per image (see module docstring).
        n_classes = len(class_names)
        track_indices_train = pick_class_representative_indices(train_labels, n_classes, random_seed)
        track_indices_val = pick_class_representative_indices(val_labels, n_classes, random_seed)

        trajectory_path = checkpoint_dir / "z_trajectory.pth"
        trajectory = torch.load(trajectory_path, map_location="cpu") if trajectory_path.is_file() else None

        plot_noise_comparison(
            representations=rep.z.detach(), noise_scale=noise_scale,
            point_indices=_canonical_rows(track_indices_train, mult).tolist(),
            point_labels=train_labels[track_indices_train].tolist(),
            class_names=class_names,
            title=f"Train Noise Injection ({tag}, σ={noise_scale:.4f})",
            save_path=str(_kind_dir(training_dir, "noise_train") / f"{tag}.png"), show=False,
            labels=train_labels_full if trajectory else None,
            track_labels=trajectory['train']['labels'] if trajectory else None,
            z_clean=trajectory['train']['z_clean'] if trajectory else None,
            z_noised_real=trajectory['train']['z_noised'] if trajectory else None,
            z_after=trajectory['train']['z_after'] if trajectory else None,
        )
        plot_noise_comparison(
            representations=val_rep.z.detach(), noise_scale=noise_scale,
            point_indices=_canonical_rows(track_indices_val, mult).tolist(),
            point_labels=val_labels[track_indices_val].tolist(),
            class_names=class_names,
            title=f"Val Noise Injection ({tag}, σ={noise_scale:.4f})",
            save_path=str(_kind_dir(training_dir, "noise_val") / f"{tag}.png"), show=False,
            labels=val_labels_full if trajectory else None,
            track_labels=trajectory['val']['labels'] if trajectory else None,
            z_clean=trajectory['val']['z_clean'] if trajectory else None,
            z_noised_real=trajectory['val']['z_noised'] if trajectory else None,
            z_after=trajectory['val']['z_after'] if trajectory else None,
        )

    plot_latent_space(
        representations=rep.z.detach(), labels=train_labels_full, class_names=class_names,
        title=f"Train Latent Space ({tag})", gmm=gmm,
        save_path=str(_kind_dir(training_dir, "latent_train") / f"{tag}.png"), show=False,
    )
    plot_latent_space(
        representations=val_rep.z.detach(), labels=val_labels_full, class_names=class_names,
        title=f"Val Latent Space ({tag})", gmm=gmm,
        save_path=str(_kind_dir(training_dir, "latent_val") / f"{tag}.png"), show=False,
    )

    indices_train, images_train, labels_train, indices_val, images_val, labels_val = sample_data
    with torch.no_grad():
        recon_train = decoder(rep(_canonical_rows(indices_train, mult).to(device)))
        recon_val = decoder(val_rep(_canonical_rows(indices_val, mult).to(device)))

    plot_ground_truth_and_reconstructions_by_class(
        images=images_train, reconstructions=recon_train, labels=labels_train, class_names=class_names,
        title=f"Train: Ground Truth vs Reconstructed ({tag})", n_per_class=5, cmap='viridis',
        save_path=str(_kind_dir(training_dir, "recon_train") / f"{tag}.png"), show=False,
    )
    plot_ground_truth_and_reconstructions_by_class(
        images=images_val, reconstructions=recon_val, labels=labels_val, class_names=class_names,
        title=f"Val: Ground Truth vs Reconstructed ({tag})", n_per_class=5, cmap='viridis',
        save_path=str(_kind_dir(training_dir, "recon_val") / f"{tag}.png"), show=False,
    )


def _plot_best_model_figures(
    checkpoint, class_names: List[str],
    train_labels: torch.Tensor, val_labels: torch.Tensor,
    sample_data: Tuple, reconstructions_dir: Path, latent_dir: Path, device: torch.device, mult: int = 1,
) -> None:
    """Best/final-model figures: rich GT/recon grids + full GMM cluster breakdown, per split.

    train_labels/val_labels are canonical (one-per-image) -- see module
    docstring and _plot_epoch_checkpoint_figures for the mult convention."""
    decoder = checkpoint['decoder']
    decoder.eval()
    rep, val_rep, gmm = checkpoint['rep'], checkpoint['val_rep'], checkpoint['gmm']
    train_labels_full = _expand_labels(train_labels, mult)
    val_labels_full = _expand_labels(val_labels, mult)

    (latent_dir / "train").mkdir(parents=True, exist_ok=True)
    (latent_dir / "val").mkdir(parents=True, exist_ok=True)

    if gmm is not None:
        generate_latent_space_figures(
            representations=rep.z.detach(), labels=train_labels_full, gmm=gmm, class_names=class_names,
            save_dir=latent_dir / "train", title_prefix="Train ",
        )
        generate_latent_space_figures(
            representations=val_rep.z.detach(), labels=val_labels_full, gmm=gmm, class_names=class_names,
            save_dir=latent_dir / "val", title_prefix="Val ",
        )
    else:
        plot_latent_space(
            representations=rep.z.detach(), labels=train_labels_full, class_names=class_names,
            title="Train Latent Space (best)",
            save_path=str(latent_dir / "train" / "overview.png"), show=False,
        )
        plot_latent_space(
            representations=val_rep.z.detach(), labels=val_labels_full, class_names=class_names,
            title="Val Latent Space (best)",
            save_path=str(latent_dir / "val" / "overview.png"), show=False,
        )

    indices_train, images_train, labels_train, indices_val, images_val, labels_val = sample_data
    with torch.no_grad():
        recon_train = decoder(rep(_canonical_rows(indices_train, mult).to(device)))
        recon_val = decoder(val_rep(_canonical_rows(indices_val, mult).to(device)))

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
    mult = _n_latent_copies(config)

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
            checkpoint, checkpoint_dir, f"epoch{epoch:04d}", epoch, config.training, class_names,
            train_labels, val_labels, config.random_seed,
            sample_data, dirs.training, device, mult=mult,
        )

    # Stitch each per-epoch figure kind into a GIF + MP4, in epoch order,
    # inside that kind's own subfolder (training/<kind>/progression.gif/.mp4)
    # -- a no-op (returns []) for any kind with fewer than 2 frames, e.g. when
    # training.plot_interval was left at its null default. The latent_* kinds
    # refit PCA/UMAP/t-SNE independently per checkpoint (see plot_latent_space),
    # so their PCA panel can visibly flip/rotate and the UMAP/t-SNE panels can
    # look discontinuous frame to frame -- a real limitation of animating them
    # as-is, not a bug in the stitching itself. recon_*/noise_* have no such
    # issue: reconstructions and noise balls are directly comparable as drawn.
    if checkpoint_dirs:
        for kind in ("latent_train", "latent_val", "recon_train", "recon_val", "noise_train", "noise_val"):
            kind_dir = _kind_dir(dirs.training, kind)
            animate_from_pattern(kind_dir, "epoch*.png", kind_dir / "progression")

    best_checkpoint = load_checkpoint(best_dir, decoder_factory, device=device)
    _plot_best_model_figures(
        best_checkpoint, class_names, train_labels, val_labels,
        sample_data, dirs.reconstructions, dirs.latent, device, mult=mult,
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
    random_seed: int = 42,
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
        In-memory z snapshots captured during the M-step optimization loop at
        training.inference.plot_interval cadence (inference has no per-step
        disk checkpointing, so these come from the caller's own loop instead
        of being reloaded from disk like generate_training_figures' epoch
        checkpoints). Each dict has keys 'step', 'z', 'noise_scale'. One
        noise-comparison, one latent-space, and one reconstruction-grid
        figure are written per snapshot, then stitched into a GIF+MP4 per
        kind inside that kind's own training/<kind>/ subfolder. None/empty
        skips all of this.

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
        track_indices_test = pick_class_representative_indices(test_labels, len(class_names), random_seed)
        for snapshot in noise_snapshots:
            tag = f"step{snapshot['step']:04d}"

            plot_noise_comparison(
                representations=snapshot['z'], noise_scale=snapshot['noise_scale'],
                point_indices=track_indices_test.tolist(), point_labels=test_labels[track_indices_test].tolist(),
                class_names=class_names,
                title=f"Test Noise Injection ({tag}, σ={snapshot['noise_scale']:.4f})",
                save_path=str(_kind_dir(dirs.training, "noise_test") / f"{tag}.png"), show=False,
            )

            # Per-step progression figures, mirroring generate_training_figures'
            # per-epoch ones: latent (GMM overlay -- frozen throughout
            # inference, so it's the same anchor every frame, unlike training)
            # and a ground-truth/reconstruction grid from this step's snapshot.
            plot_latent_space(
                representations=snapshot['z'], labels=test_labels, class_names=class_names,
                title=f"Test Latent Space ({tag})", gmm=gmm,
                save_path=str(_kind_dir(dirs.training, "latent_test") / f"{tag}.png"), show=False,
            )
            with torch.no_grad():
                recon_step = decoder(snapshot['z'][indices_test.to(device)])
            plot_ground_truth_and_reconstructions_by_class(
                images=images_test, reconstructions=recon_step, labels=labels_test, class_names=class_names,
                title=f"Test: Ground Truth vs Reconstructed ({tag})", n_per_class=5, cmap='viridis',
                save_path=str(_kind_dir(dirs.training, "recon_test") / f"{tag}.png"), show=False,
            )

        # Stitch each per-step figure kind into a GIF + MP4, in step order,
        # inside that kind's own subfolder -- a no-op for any kind with fewer
        # than 2 frames. Same PCA/UMAP/t-SNE per-frame-refit caveat as
        # generate_training_figures' latent_* animations applies to
        # latent_test here (see that function's comment).
        for kind in ("latent_test", "recon_test", "noise_test"):
            kind_dir = _kind_dir(dirs.training, kind)
            animate_from_pattern(kind_dir, "step*.png", kind_dir / "progression")

    _plot_gmm_component_samples(gmm, decoder, dirs.samples, device)

    return test_ami, test_ari

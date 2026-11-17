"""
Training utilities and trainer class for ImageDGD.
"""

import copy
import math
import torch
import torch.nn.functional as F
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from omegaconf import DictConfig, OmegaConf

# Import tgmm package for Gaussian Mixture Model
from tgmm import GaussianMixture, ClusteringMetrics

from ..models import RepresentationLayer, DGD, ConvDecoder
from ..data.dataloader import collect_all_labels, pick_class_representative_indices
from ..utils.checkpoint import save_checkpoint
from ..utils.schedules import noise_schedule, noise_std_for_dim


def reduction_scale(loss_reduction: str, n_images: float) -> float:
    """Multiplier applied to every batch loss term (recon, GMM).

    'sum' (default, the original behaviour): summed over the batch's images.
    'mean': divided by the number of images in the batch (still summed over
    pixels), so the loss no longer grows with batch size. Applied to every
    term alike, so the recon:GMM ratio -- the effective lambda_gmm -- is the
    same under both.
    """
    if loss_reduction == 'sum':
        return 1.0
    if loss_reduction == 'mean':
        return 1.0 / max(float(n_images), 1.0)
    raise ValueError(f"training.loss_reduction must be 'sum' or 'mean', got {loss_reduction!r}")


class DecoderEMA:
    """Exponential moving average of the decoder's parameters and BatchNorm buffers.

    `module` is a frozen eval-mode copy. It is what the validation phase optimises the
    validation latents against and what is saved as the best decoder. The effective decay
    warms up as min(decay, (1 + n) / (10 + n)) so the first steps are not dominated by the
    random initialisation.
    """

    def __init__(self, decoder: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.n = 0
        self.module = copy.deepcopy(decoder).eval()
        for q in self.module.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def update(self, decoder: torch.nn.Module):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for pe, q in zip(self.module.parameters(), decoder.parameters()):
            pe.mul_(d).add_(q.detach(), alpha=1.0 - d)
        for be, b in zip(self.module.buffers(), decoder.buffers()):
            if be.dtype.is_floating_point:
                be.mul_(d).add_(b.detach(), alpha=1.0 - d)
            else:
                be.copy_(b)


class MultiOptimizer(torch.optim.Optimizer):
    """Several optimizers behind one Optimizer interface (zero_grad/step/param_groups),
    so one LR scheduler drives them all. Used for Muon (conv/linear weights) + AdamW (the rest)."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        super().__init__([g for o in self.optimizers for g in o.param_groups], {})
        # share the very same group dicts so scheduler edits reach the inner optimizers
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.optimizers:
            o.step()
        return loss


@torch.no_grad()
def grow_means(means: torch.Tensor, z: torch.Tensor, k_new: int) -> torch.Tensor:
    """Coarse-to-fine component means: repeatedly split the component with the
    largest within-component sum of squares along its principal axis
    (mu +- 0.5 * sd_axis * v) until there are `k_new` means. Starting from one
    component this separates the biggest differences in the latents first."""
    means = means.clone()
    while means.shape[0] < k_new:
        a = torch.cdist(z, means).argmin(dim=1)
        sse = torch.zeros(means.shape[0], device=z.device, dtype=z.dtype)
        sse.scatter_add_(0, a, (z - means[a]).pow(2).sum(dim=1))
        sse[torch.bincount(a, minlength=means.shape[0]) < 2] = -1.0
        c = int(sse.argmax())
        if sse[c] <= 0:    # nothing left to split: seed with a random latent
            pick = torch.randint(0, z.shape[0], (1,), device=z.device)
            means = torch.cat([means, z[pick]], dim=0)
            continue
        pts = z[a == c] - means[c]
        evals, evecs = torch.linalg.eigh(pts.T @ pts / pts.shape[0])
        shift = 0.5 * evals[-1].clamp_min(1e-12).sqrt() * evecs[:, -1]
        new = means[c] - shift
        means[c] = means[c] + shift
        means = torch.cat([means, new[None]], dim=0)
    return means


@torch.no_grad()
def propose_cluster_jumps(decoder, rep, gmm, loader, device, mult: int, lambda_gmm: float,
                          optimizer=None, n_candidates: int = 1, fraction: float = 1.0,
                          margin: float = 0.0, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    """Greedy move of each training latent into another GMM component.

    For every (sampled) latent z_i, draw `n_candidates` points z' from
    components k != current(i), score the clean per-image objective
        c(z) = ||x_i - decoder(z)||^2 - lambda_gmm * log p_gmm(z)
    (the training loss without noise, per row) and overwrite z_i with the best
    candidate only if c(z') < c(z) - margin; otherwise z_i is left untouched.
    The Adam moments of moved rows are zeroed so stale momentum does not drag
    them straight back. Returns acceptance statistics.
    """
    n_comp = int(gmm.n_components)
    zero = {'n_tried': 0, 'n_accepted': 0, 'accept_rate': 0.0, 'mean_gain': 0.0}
    if n_comp < 2:
        return zero
    z_all = rep.z
    tried = accepted = 0
    gain_sum = 0.0
    moved_rows = []
    decoder_was_training = decoder.training
    decoder.eval()
    for index, x, _ in loader:
        x, index = x.to(device), index.to(device)
        if mult > 1:
            rows = (index.unsqueeze(1) * mult + torch.arange(mult, device=device)).reshape(-1)
            x_t = x.repeat_interleave(mult, dim=0)
        else:
            rows, x_t = index, x
        if fraction < 1.0:
            keep = torch.rand(rows.shape[0], device=device, generator=generator) < fraction
            rows, x_t = rows[keep], x_t[keep]
        if rows.numel() == 0:
            continue
        z = z_all[rows].detach()

        def cost(zz):
            sq_err = (decoder(zz) - x_t).pow(2).flatten(start_dim=1).sum(dim=1)
            return sq_err - lambda_gmm * gmm.score_samples(zz)

        cur_cost = cost(z)
        cur_comp = gmm.predict(z)
        best_cost, best_z = cur_cost.clone(), z.clone()
        for _ in range(n_candidates):
            target = (cur_comp + torch.randint(1, n_comp, cur_comp.shape, device=device,
                                               generator=generator)) % n_comp
            z_new = z.clone()
            for k in target.unique().tolist():
                sel = target == k
                z_new[sel] = gmm.sample(int(sel.sum()), component=int(k))[0].to(z.dtype)
            c_new = cost(z_new)
            better = c_new < best_cost
            best_cost = torch.where(better, c_new, best_cost)
            best_z[better] = z_new[better]
        accept = best_cost < cur_cost - margin
        tried += rows.numel()
        accepted += int(accept.sum())
        gain_sum += float((cur_cost - best_cost)[accept].sum())
        if accept.any():
            z_all.data[rows[accept]] = best_z[accept]
            moved_rows.append(rows[accept])
    if decoder_was_training:
        decoder.train()
    if optimizer is not None and moved_rows:
        moved = torch.cat(moved_rows)
        state = optimizer.state.get(z_all, {})
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in state:
                state[key][moved] = 0.0
    return {'n_tried': tried, 'n_accepted': accepted,
            'accept_rate': accepted / max(tried, 1), 'mean_gain': gain_sum / max(accepted, 1)}


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
        self.val_losses = []
        self.gmm_train_losses = []
        self.gmm_val_losses = []
        self.recon_train_losses = []
        self.recon_val_losses = []

        # Timing
        self.epoch_times = []

        # Learning rate tracking
        self.learning_rates = []
        self.momentum_betas = []  # Track beta_1 (momentum) values

        # Clustering metrics tracking
        self.ami_scores = []
        self.val_ami_scores = []
        self.ari_scores = []
        self.val_ari_scores = []

        # Best loss tracking
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_recon_train = float('inf')
        self.best_recon_val = float('inf')
        self.best_gmm_train = float('inf')
        self.best_gmm_val = float('inf')

        # Early stopping (based on training loss)
        # Note: Early stopping only starts after GMM is fitted to avoid false triggers
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.early_stopping_active = False  # Only activate after first GMM fit

        # Model checkpointing for best model
        self.best_model_state = None

    def _create_model_components(self, train_loader, val_loader) -> Tuple:
        """Create model components based on configuration."""
        model_config = self.config.model

        # n_latent_copies (default 1): each training/val image gets this many
        # independent representation rows instead of one -- see config.yaml's
        # comment on model.representation.n_latent_copies. Row layout is
        # [image0_copy0, image0_copy1, ..., image1_copy0, ...], i.e. image i's
        # canonical (copy 0) row is i * n_latent_copies -- see
        # self._rep_index/_canonical_rows below and report.py's module
        # docstring, which both rely on this exact layout.
        self.n_latent_copies = int(model_config.representation.get('n_latent_copies', 1))
        mult = self.n_latent_copies

        # Get dataset sizes
        nsample_train = len(train_loader.dataset) * mult
        nsample_val = len(val_loader.dataset) * mult

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

            # Each of an image's mult copies starts from that same image's
            # PCA projection (row order matches the canonical-row layout
            # above) -- noise/optimization differentiate them from there.
            if mult > 1:
                train_data_flat = train_data_flat.repeat_interleave(mult, dim=0)

            # Add PCA-specific parameters
            dist_params['data'] = train_data_flat
            # Get PCA params from config if present, otherwise use defaults
            config_dist_params = model_config.representation.get('dist_params', {})
            dist_params['whiten'] = config_dist_params.get('whiten', False)
            dist_params['svd_solver'] = config_dist_params.get('svd_solver', 'auto')
        else:
            # Distribution-specific parameters (radius, mean, cov, low, high, ...)
            # live nested under representation.dist_params in config, not as flat
            # siblings of `distribution` -- read them from there so config.yaml's
            # documented dist_params values (e.g. uniform_ball's radius) are
            # actually applied instead of silently falling back to
            # RepresentationLayer's own built-in defaults.
            dist_params = OmegaConf.to_container(
                model_config.representation.get('dist_params', {}), resolve=True
            )

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

        # Create val representation layer
        # For PCA initialization, we use normal distribution for val set
        # since we can't apply the same PCA transform (different sample size)
        if distribution == 'pca':
            if self.verbose:
                print("   Initializing val representations with normal distribution...")
            val_rep = RepresentationLayer(
                dim=model_config.representation.n_features,
                n_samples=nsample_val,
                dist='normal',  # Use normal for val set
                dist_params={},
                device=self.device
            )
        else:
            val_rep = RepresentationLayer(
                dim=model_config.representation.n_features,
                n_samples=nsample_val,
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

        # Create GMM. mean_prior/covariance_prior must already be
        # torch.Tensor -- tgmm's GaussianMixture reads `.shape` off them
        # directly with no auto-conversion (unlike weight_concentration_prior,
        # which it does wrap itself) -- but config.yaml can only hold plain
        # YAML scalars/lists, so those two need converting here first.
        def _prior_tensor(value):
            if value is None:
                return None
            container = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
            return torch.tensor(container, dtype=torch.get_default_dtype(), device=self.device)

        def make_gmm(n_components=None, init_means=None):
            return GaussianMixture(
                n_components=n_components or model_config.gmm.n_components,
                n_features=model_config.representation.n_features,
                covariance_type=model_config.gmm.covariance_type,
                max_iter=model_config.gmm.max_iter,
                tol=model_config.gmm.tol,
                reg_covar=model_config.gmm.reg_covar,
                n_init=model_config.gmm.n_init,
                init_means=init_means if init_means is not None else model_config.gmm.init_means,
                init_weights=model_config.gmm.init_weights,
                init_covariances=model_config.gmm.init_covariances,
                random_state=self.config.random_seed,
                warm_start=model_config.gmm.warm_start,
                cem=model_config.gmm.cem,
                weight_concentration_prior=model_config.gmm.weight_concentration_prior,
                mean_prior=_prior_tensor(model_config.gmm.mean_prior),
                mean_precision_prior=model_config.gmm.mean_precision_prior,
                covariance_prior=_prior_tensor(model_config.gmm.covariance_prior),
                degrees_of_freedom_prior=model_config.gmm.degrees_of_freedom_prior,
                verbose=model_config.gmm.verbose,
                verbose_interval=model_config.gmm.verbose_interval,
                device=self.device,
            )

        gmm = make_gmm()
        self._make_gmm = make_gmm

        # Create full model
        model = DGD(decoder, rep, gmm)

        return model, rep, val_rep, gmm

    def _create_optimizers(self, model, rep, val_rep) -> List:
        """Create AdamW optimizers based on configuration.

        When the cosine lr_scheduler is enabled, its base_lr_* overrides the
        optimizer's own lr at construction time -- CosineAnnealingLR has no
        base_lr argument of its own, it reads its starting LR straight off
        the optimizer, so this is the only place that value can be set.
        """
        training_config = self.training_config
        lr_config = training_config.lr_scheduler

        # Decoder optimizer (AdamW)
        decoder_config = training_config.optimizer.decoder
        decoder_lr = decoder_config.lr
        if lr_config.get('enabled', False) and lr_config.get('base_lr_decoder', None) is not None:
            decoder_lr = lr_config.base_lr_decoder
        decoder_type = decoder_config.get('type', 'adamw')
        if decoder_type == 'muon':
            # Muon orthogonalises the update of 2-D weight matrices. Muon only
            # accepts 2-D parameters, so conv kernels (4-D) are *not* given to
            # it here -- they stay on AdamW with biases and norm parameters.
            # With ConvDecoder that leaves only the initial Linear layer (if
            # any) on Muon, so check the effect before relying on it for images.
            mats = [q for q in model.decoder.parameters() if q.ndim == 2]
            rest = [q for q in model.decoder.parameters() if q.ndim != 2]
            if not mats:
                raise ValueError("optimizer.decoder.type='muon' but the decoder has no 2-D weight matrices.")
            muon = torch.optim.Muon(
                mats,
                lr=decoder_lr * decoder_config.get('muon_lr_mult', 1.0),
                weight_decay=decoder_config.weight_decay,
                momentum=decoder_config.get('muon_momentum', 0.95),
                adjust_lr_fn=decoder_config.get('muon_adjust_lr_fn', 'match_rms_adamw'),
            )
            adamw_rest = torch.optim.AdamW(
                rest, lr=decoder_lr, betas=tuple(decoder_config.betas), eps=decoder_config.eps,
                weight_decay=decoder_config.weight_decay, amsgrad=decoder_config.get('amsgrad', False))
            decoder_optimizer = MultiOptimizer([adamw_rest, muon])   # AdamW group first: logging reads its lr/betas
        elif decoder_type == 'adamw':
            decoder_optimizer = torch.optim.AdamW(
                model.decoder.parameters(),
                lr=decoder_lr,
                betas=tuple(decoder_config.betas),
                eps=decoder_config.eps,
                weight_decay=decoder_config.weight_decay,
                amsgrad=decoder_config.get('amsgrad', False)
            )
        else:
            raise ValueError(f"training.optimizer.decoder.type must be 'adamw' or 'muon', got {decoder_type!r}")

        # Representation optimizers (AdamW)
        rep_config = training_config.optimizer.representation
        rep_lr = rep_config.lr
        if lr_config.get('enabled', False) and lr_config.get('base_lr_representation', None) is not None:
            rep_lr = lr_config.base_lr_representation
        trainrep_optimizer = torch.optim.AdamW(
            rep.parameters(),
            lr=rep_lr,
            betas=tuple(rep_config.betas),
            eps=rep_config.eps,
            weight_decay=rep_config.weight_decay,
            amsgrad=rep_config.get('amsgrad', False)
        )

        valrep_optimizer = torch.optim.AdamW(
            val_rep.parameters(),
            lr=rep_lr,
            betas=tuple(rep_config.betas),
            eps=rep_config.eps,
            weight_decay=rep_config.weight_decay,
            amsgrad=rep_config.get('amsgrad', False)
        )

        return [decoder_optimizer, trainrep_optimizer, valrep_optimizer]

    def _create_schedulers(self, optimizers, total_epochs: int, steps_per_epoch: int) -> List:
        """Create learning rate schedulers using plain cosine annealing
        (base_lr -> final_lr, no warm-up, no momentum cycling).

        Note: Decoder steps per batch, representations step per epoch.
        Need different total_steps for each optimizer.
        """
        lr_config = self.training_config.lr_scheduler

        if not lr_config.get('enabled', False):
            return [None, None, None]

        schedulers = []
        for i, optimizer in enumerate(optimizers):
            # Decoder (index 0) steps per batch, representations (indices 1, 2) step per epoch
            if i == 0:  # Decoder optimizer
                total_steps = total_epochs * steps_per_epoch
                final_lr = lr_config.final_lr_decoder
            else:  # Representation optimizers (trainrep and valrep)
                total_steps = total_epochs
                final_lr = lr_config.final_lr_representation

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=final_lr,
            )

            schedulers.append(scheduler)

        return schedulers

    def train(self, train_loader, val_loader, sample_data, class_names) -> Dict[str, Any]:
        """
        Main training loop.

        Parameters:
        ----------
        train_loader: Training data loader
        val_loader: Validation data loader
        sample_data: Sample data for visualization
        class_names: List of class names

        Returns:
        -------
        Dictionary containing training results
        """
        # Store datasets for later use
        self.train_dataset = train_loader.dataset
        self.val_dataset = val_loader.dataset
        self.sample_data = sample_data
        self.class_names = class_names

        # config.paths.models_dir is already a unique, timestamped path
        # (resolved by the calling notebook, e.g.
        # experiments/<timestamp>_<experiment_name>/models) -- no
        # experiment_name join needed here.
        experiment_dir = Path(self.config.paths.models_dir)
        checkpoint_root = experiment_dir / "checkpoints"

        # How often (in epochs) to checkpoint purely for post-hoc progression
        # plotting (see generate_training_figures / report.py) -- independent
        # of refit_gmm_interval, which controls the GMM's own refit cadence,
        # not plotting. None (default): no intermediate checkpoints are
        # written during training, so generate_training_figures has nothing
        # to walk and only the best model gets plotted. Set to e.g. 10 to
        # checkpoint (and later plot) every 10th epoch.
        plot_interval = self.training_config.get('plot_interval', None)

        # Create model components
        model, rep, val_rep, gmm = self._create_model_components(train_loader, val_loader)
        # n_latent_copies (default 1) -- see _create_model_components. mult
        # copies of image i live at rep rows [i*mult, i*mult + mult), so
        # image-indexed values (dataloader indices, train_labels, ...) need
        # `* mult` to become a row index into rep.z/val_rep.z; row `i*mult`
        # (copy 0) is that image's "canonical" row, used wherever exactly one
        # representative value per image is needed (GMM fit, AMI/ARI, the
        # noise/zshift tracked points) so those stay comparable across
        # n_latent_copies values -- see the module-level comment in
        # src/visualization/report.py for the plotting side of this same
        # convention.
        mult = self.n_latent_copies

        # Collect training and val labels, indexed to match representation rows
        train_labels = collect_all_labels(train_loader)
        val_labels = collect_all_labels(val_loader)

        # Fixed set of one random point per class, per split -- tracked
        # across the whole run for the clean/noised/post-optimization
        # trajectory diagnostic (z_trajectory.pth, written alongside every
        # plot_interval checkpoint below). Same indices every epoch so the
        # per-epoch snapshots are directly comparable across the run.
        # pick_class_representative_indices is deterministic given
        # (labels, n_classes, seed) -- report.py re-derives this same set
        # independently for the noise-comparison figures rather than reading
        # it back out of z_trajectory.pth, so both stay in sync automatically.
        n_classes = len(class_names)
        self.track_indices_train = pick_class_representative_indices(train_labels, n_classes, self.config.random_seed)
        self.track_labels_train = train_labels[self.track_indices_train]
        self.track_indices_val = pick_class_representative_indices(val_labels, n_classes, self.config.random_seed)
        self.track_labels_val = val_labels[self.track_indices_val]
        # Keyed by canonical row (image_idx * mult), not raw image index --
        # see the mult comment above.
        track_pos_train = {idx * mult: pos for pos, idx in enumerate(self.track_indices_train.tolist())}
        track_pos_val = {idx * mult: pos for pos, idx in enumerate(self.track_indices_val.tolist())}

        if plot_interval is not None:
            # Persist the initial (pre-training) state as the epoch-0
            # checkpoint -- the first frame of the post-hoc progression plot.
            save_checkpoint(
                checkpoint_root / "epoch_0000",
                model.decoder, rep, val_rep, gmm,
                metadata={'epoch': 0}
            )

        # Create optimizers
        optimizers = self._create_optimizers(model, rep, val_rep)
        decoder_optimizer, trainrep_optimizer, valrep_optimizer = optimizers

        # Create learning rate schedulers
        steps_per_epoch = len(train_loader)
        schedulers = self._create_schedulers(optimizers, self.training_config.epochs, steps_per_epoch)
        decoder_scheduler, trainrep_scheduler, valrep_scheduler = schedulers

        # Log model parameters
        decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        rep_params = sum(p.numel() for p in rep.parameters() if p.requires_grad)
        val_rep_params = sum(p.numel() for p in val_rep.parameters() if p.requires_grad)

        if self.verbose:
            print(f"Decoder parameters: {decoder_params:,} ({decoder_params / 1e6:.2f}M)")
            print(f"Train representation parameters: {rep_params:,} ({rep_params / 1e6:.2f}M)")
            print(f"Val representation parameters: {val_rep_params:,} ({val_rep_params / 1e6:.2f}M)")
            print(f"Total trainable parameters: {decoder_params + rep_params + val_rep_params:,} "
                  f"({(decoder_params + rep_params + val_rep_params) / 1e6:.2f}M)")

            # Print training configuration info
            print(f"Training for {self.training_config.epochs} epochs")
            print(f"Using device: {self.device}")
            print(f"Batch size: {self.config.data.batch_size}")

        # Loss reduction ('sum' = original; 'mean' = per image), see reduction_scale.
        loss_reduction = self.training_config.get('loss_reduction', 'sum')
        reduction_scale(loss_reduction, 1)  # validate early

        # Cluster jumps (off by default): periodically try moving latents into
        # other GMM components and keep the move only if the loss improves.
        jump_cfg = self.training_config.get('cluster_jump', None)
        jump_on = bool(jump_cfg is not None and jump_cfg.get('enabled', False))
        self.jump_history = []

        # Decoder weight EMA (off by default): validation latents are optimised against the
        # averaged decoder and the best checkpoint stores it.
        ema_cfg = self.training_config.get('decoder_ema', None)
        ema = DecoderEMA(model.decoder, ema_cfg.get('decay', 0.99)) \
            if (ema_cfg is not None and ema_cfg.get('enabled', False)) else None

        # Growing GMM (off by default): start with schedule[0] components and
        # move to the next entry at every scheduled refit epoch (>= first_epoch_gmm),
        # splitting the widest component each time (see grow_means).
        growth_cfg = self.training_config.get('gmm_growth', None)
        growth_on = bool(growth_cfg is not None and growth_cfg.get('enabled', False))
        final_K = int(self.config.model.gmm.n_components)
        growth_sched = [int(k) for k in growth_cfg.get('schedule', [1, final_K])] if growth_on else []
        growth_stage = 0
        self.gmm_growth_history = []
        if growth_on:
            gmm = self._make_gmm(n_components=min(growth_sched[0], final_K))
            model.gmm = gmm
        walk_cfg = self.training_config.get('latent_walk', None)
        self.walk_scales = []

        # Training loop
        start_time = time.time()

        for epoch in range(1, self.training_config.epochs + 1):
            epoch_start_time = time.time()

            # Initialize loss tracking
            train_loss = 0.0
            val_loss = 0.0
            gmm_train_loss = 0.0
            gmm_val_loss = 0.0
            recon_train_loss = 0.0
            recon_val_loss = 0.0

            # Initialize or refit GMM
            first_epoch_gmm = self.training_config.first_epoch_gmm
            refit_gmm_interval = self.training_config.refit_gmm_interval

            # Initialize clustering metrics calculator
            cluster_metrics = ClusteringMetrics()
            current_train_ami = 0.0
            current_val_ami = 0.0
            current_train_ari = 0.0
            current_val_ari = 0.0

            # Calculate scheduled noise scale (shape selected by
            # latent_noise_schedule -- constant/linear/cosine/warmup_cosine,
            # see noise_schedule in src/utils/schedules.py). noise_scale is
            # the target *expected displacement norm* in latent space,
            # independent of dim_z; noise_std is the actual per-coordinate
            # std drawn from below, corrected by 1/sqrt(dim) so that expected
            # displacement stays ~noise_scale regardless of representation.n_features
            # (see noise_std_for_dim's docstring for why this correction is needed).
            # Computed up front (before the GMM fit below) so the GMM's own
            # fit can also draw noised representations, not just the decoder.
            if self.training_config.latent_noise_enabled:
                noise_start = self.training_config.get('latent_noise_start', 1.0)
                noise_end = self.training_config.get('latent_noise_end', 0.01)
                noise_schedule_type = self.training_config.get('latent_noise_schedule', 'cosine')
                noise_warmup_epochs = self.training_config.get('latent_noise_warmup_epochs', 0)
                noise_scale = noise_schedule(
                    epoch, self.training_config.epochs, noise_start, noise_end,
                    schedule_type=noise_schedule_type, warmup_steps=noise_warmup_epochs,
                )
                noise_std = noise_std_for_dim(noise_scale, rep.dim)
            else:
                noise_scale = 0.0
                noise_std = 0.0

            # GMM schedule. Defaults reproduce the original behaviour (refit
            # every refit_gmm_interval epochs forever, cheap warm-started
            # update on every other epoch). gmm_refit_until freezes the GMM
            # after that epoch; gmm_update_between_refits=False keeps it
            # constant between refits instead of the per-epoch warm update.
            gmm_refit_until = self.training_config.get('gmm_refit_until', None)
            gmm_update_between = self.training_config.get('gmm_update_between_refits', True)
            if gmm_refit_until is None:
                is_gmm_refit_epoch = epoch == first_epoch_gmm or (refit_gmm_interval and epoch % refit_gmm_interval == 0)
            else:
                is_gmm_refit_epoch = epoch == first_epoch_gmm or bool(
                    refit_gmm_interval and epoch % refit_gmm_interval == 0
                    and first_epoch_gmm <= epoch <= gmm_refit_until)
            gmm_frozen = gmm_refit_until is not None and epoch > gmm_refit_until
            do_gmm_fit = is_gmm_refit_epoch or (epoch > first_epoch_gmm and gmm_update_between and not gmm_frozen)

            grew_this_epoch = False
            if growth_on and is_gmm_refit_epoch and epoch >= first_epoch_gmm:
                k_target = min(growth_sched[min(growth_stage, len(growth_sched) - 1)], final_K)
                growth_stage += 1
                if k_target > gmm.n_components:
                    with torch.no_grad():
                        z_clean = rep.z.detach()
                        if mult > 1:
                            z_clean = z_clean.view(len(train_labels), mult, rep.dim)[:, 0, :]
                        base = gmm.means_ if (gmm.fitted_ and gmm.means_ is not None) else z_clean.mean(0, keepdim=True)
                        init = grow_means(base.to(z_clean.device, z_clean.dtype), z_clean, k_target)
                    gmm = self._make_gmm(n_components=k_target, init_means=init)
                    model.gmm = gmm
                    grew_this_epoch = True
                self.gmm_growth_history.append({'epoch': epoch, 'n_components': int(gmm.n_components)})
                if self.verbose:
                    print(f"[gmm growth @ epoch {epoch}] {gmm.n_components} components")

            if is_gmm_refit_epoch or epoch > first_epoch_gmm:
                with torch.no_grad():
                    # Canonical (copy 0) row per image only -- with
                    # n_latent_copies > 1, fitting on every copy would treat
                    # each image's mult rows as mult independent samples,
                    # inflating the GMM's effective N (and so the pooled
                    # scatter each M-step's reg_covar has to compete with)
                    # purely as a side effect of this setting, not a real
                    # change in how much data the model has. See mult comment
                    # above.
                    representations = rep.z.detach()
                    if mult > 1:
                        representations = representations.view(len(train_labels), mult, rep.dim)[:, 0, :]
                    # Fit on a noised copy, not the clean representations:
                    # noise inflates the empirical spread the GMM sees, which
                    # raises its effective covariance floor for this fit and
                    # makes it harder for a component to collapse onto a
                    # single tight point. Prediction/metrics below still use
                    # the clean representations -- we want real cluster
                    # quality, not the noised-for-fitting copy.
                    if noise_std > 0:
                        fit_representations = representations + torch.randn_like(representations) * noise_std
                    else:
                        fit_representations = representations
                    if not do_gmm_fit:
                        pass  # GMM held constant this epoch (metrics below still run)
                    elif is_gmm_refit_epoch:
                        gmm.fit(fit_representations, max_iter=1000 if (epoch == first_epoch_gmm or grew_this_epoch) else 100)
                    else:
                        # Between refits: cheap warm-started update only.
                        gmm.fit(fit_representations, max_iter=100, warm_start=True)

                    # Calculate AMI and ARI for training data
                    predicted_labels = gmm.predict(representations)
                    current_train_ami = cluster_metrics.adjusted_mutual_info_score(train_labels, predicted_labels)
                    self.ami_scores.append(current_train_ami)
                    current_train_ari = cluster_metrics.adjusted_rand_score(train_labels, predicted_labels)
                    self.ari_scores.append(current_train_ari)

                    # Calculate AMI and ARI for val data (canonical rows only -- same reasoning as above)
                    val_representations = val_rep.z.detach()
                    if mult > 1:
                        val_representations = val_representations.view(len(val_labels), mult, rep.dim)[:, 0, :]
                    val_predicted_labels = gmm.predict(val_representations)
                    current_val_ami = cluster_metrics.adjusted_mutual_info_score(val_labels, val_predicted_labels)
                    self.val_ami_scores.append(current_val_ami)
                    current_val_ari = cluster_metrics.adjusted_rand_score(val_labels, val_predicted_labels)
                    self.val_ari_scores.append(current_val_ari)

            # Checkpoint purely for post-hoc progression plotting -- its own
            # cadence (plot_interval), independent of the GMM refit cadence
            # above. See the plot_interval comment near the epoch-0 checkpoint.
            if plot_interval is not None and epoch % plot_interval == 0:
                save_checkpoint(
                    checkpoint_root / f"epoch_{epoch:04d}",
                    model.decoder, rep, val_rep, gmm,
                    metadata={
                        'epoch': epoch,
                        'train_ami': current_train_ami,
                        'val_ami': current_val_ami,
                        'train_ari': current_train_ari,
                        'val_ari': current_val_ari,
                    }
                )

            if is_gmm_refit_epoch:
                # Activate early stopping after first GMM fit
                if epoch == first_epoch_gmm:
                    self.early_stopping_active = True
                    self.best_train_loss = float('inf')  # Reset best loss
                    self.best_val_loss = float('inf')  # Reset val loss too (GMM adds error term)
                    self.epochs_without_improvement = 0

            # z_trajectory tracking: clean/noised/post-optimization snapshots
            # of the fixed track_indices_*, only on plot_interval epochs
            # (same cadence as the checkpoint above). z_clean is rep.z as of
            # right now -- the end of the *previous* epoch's update, i.e.
            # exactly what this epoch's noise gets added to below.
            track_this_epoch = plot_interval is not None and epoch % plot_interval == 0
            if track_this_epoch:
                # Canonical row per tracked image (idx * mult) -- see mult comment above.
                track_rows_train = self.track_indices_train * mult
                track_rows_val = self.track_indices_val * mult
                z_track_clean_train = rep.z[track_rows_train].detach().clone()
                z_track_noised_train = torch.zeros_like(z_track_clean_train)
                track_idx_train_device = track_rows_train.to(self.device)
                z_track_clean_val = val_rep.z[track_rows_val].detach().clone()
                z_track_noised_val = torch.zeros_like(z_track_clean_val)
                track_idx_val_device = track_rows_val.to(self.device)

            # Cluster-jump step (training.cluster_jump): clean, no-noise,
            # greedy accept/reject against the same loss as training.
            if jump_on and gmm.fitted_ and epoch >= jump_cfg.get('start_epoch', first_epoch_gmm) \
                    and (epoch - jump_cfg.get('start_epoch', first_epoch_gmm)) % jump_cfg.get('interval', 25) == 0 \
                    and (jump_cfg.get('until', None) is None or epoch <= jump_cfg.get('until')):
                stats = propose_cluster_jumps(
                    model.decoder, rep, gmm, train_loader, self.device, mult,
                    self.training_config.lambda_gmm, optimizer=trainrep_optimizer,
                    n_candidates=jump_cfg.get('n_candidates', 3),
                    fraction=jump_cfg.get('fraction', 1.0),
                    margin=jump_cfg.get('margin', 0.0))
                stats['epoch'] = epoch
                self.jump_history.append(stats)
                if self.verbose:
                    print(f"[cluster jump @ epoch {epoch}] accepted {stats['n_accepted']}/{stats['n_tried']} "
                          f"({100 * stats['accept_rate']:.1f}%), mean loss gain {stats['mean_gain']:.3f}")

            # Training phase
            model.decoder.train()
            trainrep_optimizer.zero_grad()

            for index, x, _ in train_loader:
                decoder_optimizer.zero_grad()

                x, index = x.to(self.device), index.to(self.device)

                # Expand each image index into its mult representation rows
                # (see mult comment above) and repeat its target image to
                # match, so every copy gets its own independently-drawn noise
                # below but is trained against the same reconstruction target.
                # A no-op reshape when mult == 1.
                if mult > 1:
                    rep_index = (index.unsqueeze(1) * mult + torch.arange(mult, device=self.device)).reshape(-1)
                    x_target = x.repeat_interleave(mult, dim=0)
                else:
                    rep_index = index
                    x_target = x

                # Forward pass
                z = rep(rep_index)

                # Latent Space Noise Injection (regularization, applied in
                # both train and val phases so noise robustness generalizes)
                if noise_std > 0:
                    noise = torch.randn_like(z) * noise_std
                    z = z + noise

                if track_this_epoch:
                    track_mask = torch.isin(rep_index, track_idx_train_device)
                    if track_mask.any():
                        for di, zi in zip(rep_index[track_mask].tolist(), z[track_mask].detach().cpu()):
                            z_track_noised_train[track_pos_train[di]] = zi

                y = model.decoder(z)
                # Divided by mult so recon_loss stays the same scale (average
                # over each image's mult copies) regardless of n_latent_copies
                # -- a no-op at mult=1. Without this, both loss terms below
                # would scale with mult purely from summing over more rows,
                # confounding "more copies" with "a stronger loss".
                red = reduction_scale(loss_reduction, x_target.shape[0] / mult)
                recon_loss = F.mse_loss(y, x_target, reduction='sum') / mult * red

                # Add GMM loss if applicable
                if epoch >= first_epoch_gmm and gmm is not None:
                    gmm_error = -self.training_config.lambda_gmm * torch.sum(gmm.score_samples(z)) / mult * red
                    loss = recon_loss + gmm_error
                else:
                    loss = recon_loss
                    gmm_error = torch.tensor(0.0).to(self.device)

                # Backward pass
                loss.backward()
                decoder_optimizer.step()
                if ema is not None:
                    ema.update(model.decoder)

                # Step decoder's cosine scheduler (steps per batch)
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

            # Persistent latent walk (training.latent_walk): unlike the per-batch
            # noise above (used only to evaluate the loss), this writes noise
            # into the stored latents so they explore the space. Own schedule,
            # same displacement-norm convention as latent_noise_*.
            if walk_cfg is not None and walk_cfg.get('enabled', False):
                walk_scale = noise_schedule(
                    epoch, self.training_config.epochs, walk_cfg.get('start', 0.2), walk_cfg.get('end', 0.0),
                    schedule_type=walk_cfg.get('schedule', 'cosine'), warmup_steps=0)
                if walk_scale > 0:
                    with torch.no_grad():
                        rep.z.add_(torch.randn_like(rep.z) * noise_std_for_dim(walk_scale, rep.dim))
                self.walk_scales.append(walk_scale)

            if track_this_epoch:
                z_track_after_train = rep.z[track_rows_train].detach().clone()

            # Validation phase - update val representations only
            model.decoder.eval()
            valrep_optimizer.zero_grad()

            val_decoder = ema.module if ema is not None else model.decoder

            # Disable gradients for decoder parameters during val phase
            for param in model.decoder.parameters():
                param.requires_grad = False

            for index, x, _ in val_loader:
                x, index = x.to(self.device), index.to(self.device)

                if mult > 1:
                    rep_index = (index.unsqueeze(1) * mult + torch.arange(mult, device=self.device)).reshape(-1)
                    x_target = x.repeat_interleave(mult, dim=0)
                else:
                    rep_index = index
                    x_target = x

                # Forward pass
                z = val_rep(rep_index)

                # Latent Space Noise Injection (same per-epoch schedule as
                # the train phase above — no separate/independent schedule)
                if noise_std > 0:
                    noise = torch.randn_like(z) * noise_std
                    z = z + noise

                if track_this_epoch:
                    track_mask = torch.isin(rep_index, track_idx_val_device)
                    if track_mask.any():
                        for di, zi in zip(rep_index[track_mask].tolist(), z[track_mask].detach().cpu()):
                            z_track_noised_val[track_pos_val[di]] = zi

                y = val_decoder(z)
                red = reduction_scale(loss_reduction, x_target.shape[0] / mult)
                recon_loss = F.mse_loss(y, x_target, reduction='sum') / mult * red

                # Add GMM loss if applicable
                if epoch >= first_epoch_gmm and gmm is not None:
                    gmm_error = -self.training_config.lambda_gmm * torch.sum(gmm.score_samples(z)) / mult * red
                    loss = recon_loss + gmm_error
                else:
                    loss = recon_loss
                    gmm_error = torch.tensor(0.0).to(self.device)

                # Backward pass for val representations only
                loss.backward()

                # Track losses
                val_loss += loss.item()
                recon_val_loss += recon_loss.item()
                if epoch >= first_epoch_gmm:
                    gmm_val_loss += gmm_error.item()

            # Re-enable gradients for decoder parameters
            for param in model.decoder.parameters():
                param.requires_grad = True

            valrep_optimizer.step()

            # Step valrep scheduler
            if valrep_scheduler is not None:
                valrep_scheduler.step()

            if track_this_epoch:
                z_track_after_val = val_rep.z[track_rows_val].detach().clone()
                # checkpoint_root / epoch_XXXX already exists -- created by
                # the save_checkpoint call earlier this epoch, gated by the
                # same plot_interval condition as track_this_epoch.
                torch.save(
                    {
                        'epoch': epoch,
                        'train': {
                            'indices': self.track_indices_train,
                            'labels': self.track_labels_train,
                            'z_clean': z_track_clean_train,
                            'z_noised': z_track_noised_train,
                            'z_after': z_track_after_train,
                        },
                        'val': {
                            'indices': self.track_indices_val,
                            'labels': self.track_labels_val,
                            'z_clean': z_track_clean_val,
                            'z_noised': z_track_noised_val,
                            'z_after': z_track_after_val,
                        },
                    },
                    checkpoint_root / f"epoch_{epoch:04d}" / "z_trajectory.pth",
                )

            # Normalize losses
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            recon_train_loss /= len(train_loader.dataset)
            recon_val_loss /= len(val_loader.dataset)
            gmm_train_loss /= len(train_loader.dataset)
            gmm_val_loss /= len(val_loader.dataset)

            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.recon_train_losses.append(recon_train_loss)
            self.recon_val_losses.append(recon_val_loss)
            self.gmm_train_losses.append(gmm_train_loss)
            self.gmm_val_losses.append(gmm_val_loss)

            # Track learning rates (decoder LR)
            self.learning_rates.append(decoder_optimizer.param_groups[0]['lr'])
            # Track momentum (beta_1)
            self.momentum_betas.append(decoder_optimizer.param_groups[0]['betas'][0])

            # Update best losses and early stopping
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
            # Model selection is keyed on recon_val_loss, not combined
            # val_loss: gmm_error is unbounded below as the GMM's covariance
            # sharpens (log-density -> +inf as covariance -> 0), so combined
            # val_loss keeps "improving" indefinitely once the GMM starts
            # sharpening even while reconstruction quality is actively
            # getting worse -- selecting by combined val_loss would then
            # always just end up picking the last epoch, regardless of
            # reconstruction quality. Before first_epoch_gmm, recon_val_loss
            # == val_loss (gmm_error is 0), so this is a strict
            # generalization of the old criterion, not a behavior change,
            # during the pre-GMM phase.
            if recon_val_loss < self.best_recon_val:
                self.best_recon_val = recon_val_loss
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                # Save best model checkpoint (decoder and representations only)
                self.best_model_state = {
                    'decoder': {k: v.detach().clone() for k, v in
                                (ema.module if ema is not None else model.decoder).state_dict().items()},
                    'rep': {k: v.detach().clone() for k, v in rep.state_dict().items()},
                    'val_rep': {k: v.detach().clone() for k, v in val_rep.state_dict().items()}
                }
                # Reset early stopping counter when recon_val_loss improves
                if self.early_stopping_active:
                    self.epochs_without_improvement = 0
            elif self.early_stopping_active:
                # Increment counter only if recon_val_loss didn't improve
                self.epochs_without_improvement += 1
            if recon_train_loss < self.best_recon_train:
                self.best_recon_train = recon_train_loss
            if epoch >= first_epoch_gmm:
                if gmm_train_loss < self.best_gmm_train:
                    self.best_gmm_train = gmm_train_loss
                if gmm_val_loss < self.best_gmm_val:
                    self.best_gmm_val = gmm_val_loss

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
            gmm_val_str = f"{gmm_val_loss:.4f} (B: {self.best_gmm_val:.4f})" if epoch >= first_epoch_gmm else "0.0000"
            train_ami_ari_str = f", AMI={current_train_ami:.4f}, ARI={current_train_ari:.4f}" if epoch >= first_epoch_gmm else ""
            val_ami_ari_str = f", AMI={current_val_ami:.4f}, ARI={current_val_ari:.4f}" if epoch >= first_epoch_gmm else ""

            print(f"Epoch {epoch}/{self.training_config.epochs} [Time per Epoch: {epoch_time_str}, Remaining Time: {remaining_time_str}, LR: Dec={lr_decoder:.2e}, Rep={lr_rep:.2e}, Noise={noise_scale:.4f}]")
            print(f"       - Train Loss: {train_loss:.4f} (B: {self.best_train_loss:.4f}), Recon: {recon_train_loss:.4f} (B: {self.best_recon_train:.4f}), GMM: {gmm_train_str}{train_ami_ari_str}")
            print(f"       - Val   Loss: {val_loss:.4f} (B: {self.best_val_loss:.4f}), Recon: {recon_val_loss:.4f} (B: {self.best_recon_val:.4f}), GMM: {gmm_val_str}{val_ami_ari_str}")

            # Check for early stopping (only if active)
            early_stopping_patience = getattr(self.training_config, 'early_stopping_patience', None)
            if self.early_stopping_active and early_stopping_patience is not None and self.epochs_without_improvement >= early_stopping_patience:
                if self.verbose:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"   No improvement in recon val loss for {early_stopping_patience} consecutive epochs (since GMM activation)")
                    print(f"   Best recon val loss: {self.best_recon_val:.4f} at epoch {self.best_epoch}")
                break

        if plot_interval is not None:
            # Persist the final-epoch state (whether training completed or
            # stopped early) -- always included as the last progression
            # frame, even when `epoch` isn't itself a multiple of plot_interval.
            save_checkpoint(
                checkpoint_root / f"epoch_{epoch:04d}",
                model.decoder, rep, val_rep, gmm,
                metadata={'epoch': epoch, 'train_loss': self.train_losses[-1], 'val_loss': self.val_losses[-1]}
            )

        # Restore best model checkpoint
        if self.best_model_state is not None:
            if self.verbose:
                print(f"\nRestoring best model from epoch {self.best_epoch} (recon val loss: {self.best_recon_val:.4f}, val loss: {self.best_val_loss:.4f})")
            model.decoder.load_state_dict(self.best_model_state['decoder'])
            rep.load_state_dict(self.best_model_state['rep'])
            val_rep.load_state_dict(self.best_model_state['val_rep'])
            # Refit GMM to the best representations (canonical rows only -- see mult comment above)
            if growth_on and gmm.n_components < final_K:
                with torch.no_grad():
                    z_best = rep.z.detach()
                    if mult > 1:
                        z_best = z_best.view(len(train_labels), mult, rep.dim)[:, 0, :]
                    base = gmm.means_ if (gmm.fitted_ and gmm.means_ is not None) else z_best.mean(0, keepdim=True)
                    gmm = self._make_gmm(n_components=final_K,
                                         init_means=grow_means(base.to(z_best.device, z_best.dtype), z_best, final_K))
                    model.gmm = gmm
            if self.verbose:
                print("   Refitting GMM to best representations...")
            with torch.no_grad():
                representations = rep.z.detach()
                if mult > 1:
                    representations = representations.view(len(train_labels), mult, rep.dim)[:, 0, :]
                gmm.fit(representations, max_iter=self.config.model.gmm.max_iter)
            if self.verbose:
                print(f"   GMM converged: {gmm.converged_} (iterations: {gmm.n_iter_})")

        # Final GMM fit after training completes (for best generative model)
        final_gmm_fit = getattr(self.training_config, 'final_gmm_fit', False)
        if final_gmm_fit:
            if self.verbose:
                print("\nPerforming final GMM fit for optimal generative model...")
            with torch.no_grad():
                representations = rep.z.detach()
                if mult > 1:
                    representations = representations.view(len(train_labels), mult, rep.dim)[:, 0, :]
                gmm.fit(representations, max_iter=self.config.model.gmm.max_iter)
            if self.verbose:
                print(f"   Final GMM converged: {gmm.converged_} (iterations: {gmm.n_iter_})")

        # Optional descriptive GMM (training.final_gmm.*): a separate, richer
        # mixture (e.g. covariance_type 'full') fitted once on the final
        # latents and saved next to the training GMM as best/gmm_final.pkl.
        # The training GMM (gmm.pkl) is untouched -- it is what inference's
        # prior is tied to.
        final_cfg = self.training_config.get('final_gmm', None)
        gmm_final = None
        if final_cfg is not None and final_cfg.get('enabled', False):
            gm = self.config.model.gmm
            with torch.no_grad():
                representations = rep.z.detach()
                if mult > 1:
                    representations = representations.view(len(train_labels), mult, rep.dim)[:, 0, :]
            # Full covariances with many components in a high-dimensional latent
            # space easily go singular (a component with fewer images than
            # dimensions), so escalate reg_covar and finally fall back to a
            # diagonal covariance. A failure here must never abort the run: the
            # model below still gets saved.
            cov_type = final_cfg.get('covariance_type', 'full')
            reg0 = float(final_cfg.get('reg_covar', 1e-3))
            attempts = [(cov_type, reg0 * 10 ** i) for i in range(4)] + [('diag', 1e-3)]
            for cov, reg in attempts:
                try:
                    cand = GaussianMixture(
                        n_components=final_cfg.get('n_components', None) or gm.n_components,
                        n_features=self.config.model.representation.n_features,
                        covariance_type=cov,
                        max_iter=final_cfg.get('max_iter', gm.max_iter),
                        tol=gm.tol, reg_covar=reg,
                        n_init=final_cfg.get('n_init', 3),
                        init_means=gm.init_means,
                        random_state=self.config.random_seed,
                        device=self.device,
                    )
                    cand.fit(representations)
                    gmm_final = cand
                    break
                except Exception as exc:  # singular covariance etc.
                    if self.verbose:
                        print(f"   final GMM ({cov}, reg_covar={reg:g}) failed: {str(exc)[:90]} -- trying next setting")
            if self.verbose and gmm_final is not None:
                print(f"\nFinal descriptive GMM ({gmm_final.covariance_type}, K={gmm_final.n_components}, "
                      f"reg_covar={gmm_final.reg_covar:g}): converged={gmm_final.converged_}, "
                      f"mean log-lik={float(gmm_final.score_samples(representations).mean()):.3f} "
                      f"(training GMM: {float(gmm.score_samples(representations).mean()):.3f})")
            elif gmm_final is None:
                print("   WARNING: final descriptive GMM could not be fitted; skipped (training GMM is unaffected).")

        # Persist the best model: decoder, train/val representations, GMM,
        # and loss history. The config.yaml copy for this run lives at the
        # run folder's root (written once by the calling notebook when the
        # run folder is created), not duplicated here.
        best_dir = experiment_dir / "best"
        save_checkpoint(
            best_dir,
            model.decoder, rep, val_rep, gmm,
            metadata={
                'best_epoch': self.best_epoch,
                'best_val_loss': self.best_val_loss,
                'best_recon_val_loss': self.best_recon_val,
            }
        )
        if gmm_final is not None and gmm_final.fitted_:
            gmm_final.save(str(best_dir / "gmm_final.pkl"))
        torch.save({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_train_losses': self.recon_train_losses,
            'recon_val_losses': self.recon_val_losses,
            'gmm_train_losses': self.gmm_train_losses,
            'gmm_val_losses': self.gmm_val_losses,
            'ami_scores': self.ami_scores,
            'val_ami_scores': self.val_ami_scores,
            'ari_scores': self.ari_scores,
            'val_ari_scores': self.val_ari_scores,
            'learning_rates': self.learning_rates,
            'momentum_betas': self.momentum_betas,
            'epoch_times': self.epoch_times,
            'cluster_jump_history': self.jump_history,
            'latent_walk_scales': self.walk_scales,
            'gmm_growth_history': self.gmm_growth_history,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_recon_val_loss': self.best_recon_val,
        }, best_dir / "training_results.pth")

        # Training complete
        total_time = time.time() - start_time
        if self.verbose:
            print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
            print(f"Final training loss: {self.train_losses[-1]:.4f}")
            print(f"Final val loss: {self.val_losses[-1]:.4f}")
            print(f"Best recon val loss: {self.best_recon_val:.4f} at epoch {self.best_epoch}")
        else:
            print(f"Training completed: {self.train_losses[-1]:.4f} train, {self.val_losses[-1]:.4f} val")

        # Return results
        early_stopping_patience = getattr(self.training_config, 'early_stopping_patience', None)
        stopped_early = (
            early_stopping_patience is not None
            and self.epochs_without_improvement >= early_stopping_patience
        )
        return {
            "model": model,
            "rep": rep,
            "val_rep": val_rep,
            "gmm": gmm,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "total_time": total_time,
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "best_train_loss": self.best_train_loss,
            "best_recon_val_loss": self.best_recon_val,
            "best_epoch": self.best_epoch,
            "stopped_early": stopped_early,
        }

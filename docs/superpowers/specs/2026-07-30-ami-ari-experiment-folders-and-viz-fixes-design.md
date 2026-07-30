# AMI+ARI Metric Swap, Timestamped Experiment Folders, Inference Figure Parity, and Two Visualization Bug Fixes — Design

## Problem

Four related changes, requested together:

1. **NMI was the wrong call.** The previous plan swapped ARI+silhouette for a single NMI metric. That was a mistake — Adjusted Mutual Information (AMI) and Adjusted Rand Index (ARI) should be tracked side by side instead, so the two can be compared directly. Silhouette stays removed.
2. **Every training/inference run overwrites the last one's models and figures.** `models/{experiment_name}/` and `figures/{experiment_name}/` are fixed paths — running the training notebook twice destroys the first run's checkpoints and figures. There's no record of which config produced which results.
3. **The inference notebook produces fewer figures than training does**, and currently reuses training's noise schedule verbatim rather than having its own tunable knobs.
4. **Two visualization bugs**: the GMM-component overlay on the PCA latent-space plot crashes for `spherical`/`tied_spherical` covariance types, and cuML's `TSNE` emits a spurious "nearest neighbors" warning on every call.

## Goals

1. `tgmm.ClusteringMetrics.adjusted_mutual_info_score` and `.adjusted_rand_score` replace `.normalized_mutual_info_score` everywhere the latter was just introduced — training-time tracking, checkpoint metadata, the `training_results.pth` schema, the loss-curve panel, and the inference notebook's final metric.
2. Every training run writes to its own timestamped folder (`experiments/<timestamp>_<experiment_name>/`), containing `config.yaml` (the exact config used), `models/` (checkpoints + best), and `figures/`. Nothing gets overwritten by a later run.
3. Every inference run against a trained model writes to its own timestamped subfolder nested under that training run (`experiments/<training-run>/inference/<timestamp>/`), containing its own `config.yaml` copy, `test_representation.pt`, and `figures/`.
4. The inference notebook produces the same category of figures the training notebook does: latent space, reconstructions, a loss-curve-style analysis of its own optimization, and GMM-component sample grids.
5. `training.inference:` gains its own `latent_noise_scale`/`latent_noise_start`/`latent_noise_end`, independent of the top-level training noise settings, so inference-time noise (and epoch count `M`) can be tuned without touching training's config. No new file — `config.yaml` stays the single source of truth, and it's what gets copied into each run folder.
6. `_add_gmm_overlay_pca` (in `src/visualization/latent.py`) renders correctly for every covariance type, including `spherical`/`tied_spherical`.
7. `TSNE(...)` calls in `plot_latent_space` no longer trigger cuML's nearest-neighbors warning.

## Non-goals

- No migration of the checkpoint currently on disk at `models/ImageDGD_Default/` — it's abandoned under the new scheme, same as any pre-this-plan run. The user is actively retraining; no backward-compatibility shim is added.
- No change to the checkpoint file format itself (`save_checkpoint`/`load_checkpoint` in `src/utils/checkpoint.py`) — only where callers point those functions.
- No pruning/retention policy for old run folders. Every run keeps its folder forever; cleanup is a manual/future concern.
- No change to how NMI's replacement metrics are computed *from* — same clean, stored `rep.z`/`val_rep.z`/`test_rep.z` as before, never a transient noised copy.
- No change to GMM refit cadence, checkpoint-write cadence, or the noise-injection mechanism itself (train/val/inference all still noise, per the prior plan) — only the metric names and the folder layout change.

## 1. Metric swap: NMI → AMI + ARI

### `src/training/trainer.py`

- `__init__` tracking lists: `self.nmi_scores`, `self.val_nmi_scores` → `self.ami_scores`, `self.val_ami_scores`, `self.ari_scores`, `self.val_ari_scores` (four lists).
- Both GMM-refit branches (`is_gmm_refit_epoch` and `elif epoch > first_epoch_gmm`): each `cluster_metrics.normalized_mutual_info_score(...)` call becomes two calls — `cluster_metrics.adjusted_mutual_info_score(labels_true, labels_pred)` and `cluster_metrics.adjusted_rand_score(labels_true, labels_pred)` — for both train and val, appended to the four lists above.
- Checkpoint metadata at GMM-refit epochs: `train_nmi`/`val_nmi` → `train_ami`/`val_ami`/`train_ari`/`val_ari`.
- Per-epoch log line: `NMI=0.0800` → `AMI=0.0800, ARI=0.0003` (both train and val strings).
- `training_results.pth` schema: `nmi_scores`/`val_nmi_scores` → `ami_scores`/`val_ami_scores`/`ari_scores`/`val_ari_scores`.

### `src/visualization/loss.py`

`plot_training_analysis` panel 4 becomes single-axis, 4 lines (confirmed: AMI and ARI share a comparable scale, unlike the old ARI+silhouette split): "Train AMI", "Val AMI", "Train ARI", "Val ARI". Title "Clustering Quality (AMI & ARI)", y-label "Score". Same `_align`/`metric_epochs` machinery, now serving four lists instead of two.

### `src/visualization/report.py`

`generate_training_figures`'s schema guard and `SimpleNamespace` construction read `ami_scores`/`val_ami_scores`/`ari_scores`/`val_ari_scores` instead of the two NMI keys; the guard's error message updates to name the new keys.

### `notebooks/dgd_test_inference.ipynb`

Final metrics cell: `test_ami = cluster_metrics.adjusted_mutual_info_score(test_labels, predicted_labels)` and `test_ari = cluster_metrics.adjusted_rand_score(test_labels, predicted_labels)`, both printed and both in the latent-space plot title.

## 2. Timestamped experiment folders

### `config/config.yaml`

`paths:` section replaces `models_dir`/`figures_dir` with a single `experiments_dir: "./experiments"`.

### `src/training/trainer.py`

`train()` currently computes `experiment_dir = Path(self.config.paths.models_dir) / self.config.experiment_name`. The caller (notebook) now resolves `config.paths.models_dir` to an already-unique, already-timestamped path before calling `train()`, so this becomes `experiment_dir = Path(self.config.paths.models_dir)` — no `/ self.config.experiment_name` join. `checkpoint_root`/`best_dir` derivation from `experiment_dir` is unchanged. The `OmegaConf.save(self.config, str(best_dir / "config.yaml"))` line is deleted — the notebook now writes the config copy once, at the run folder's root, when it creates the run folder (before training starts), not at the end of training into `best/`.

### `src/visualization/report.py`

`generate_training_figures` currently loads `config = OmegaConf.load(best_dir / "config.yaml")`. Since the config copy now lives at the run root (one level above `models/`, i.e. `experiment_dir.parent`, since `experiment_dir` is the caller's `models_dir` = `run_dir/models`), this becomes `config = OmegaConf.load(experiment_dir.parent / "config.yaml")`. No signature change — `generate_training_figures(experiment_dir, figures_dir, ...)` still takes the same two path arguments; this is an internal path derivation change only.

### `notebooks/dgd_training_demo.ipynb`

The config-setup cell gains timestamp/run-folder resolution, replacing the current `config.paths.models_dir`/`figures_dir` assignment:

```python
from datetime import datetime

config.data.root_dir = str(project_root / "data")
config.paths.experiments_dir = str(project_root / "experiments")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(config.paths.experiments_dir) / f"{timestamp}_{config.experiment_name}"
run_dir.mkdir(parents=True, exist_ok=True)

config.paths.models_dir = str(run_dir / "models")
config.paths.figures_dir = str(run_dir / "figures")

OmegaConf.save(config, str(run_dir / "config.yaml"))
print(f"Run directory: {run_dir}")
```

Everything downstream in the notebook (the `trainer.train(...)` call, the `generate_training_figures(...)` call) already reads `config.paths.models_dir`/`figures_dir` and needs no further change — both now resolve under `run_dir`.

### `notebooks/dgd_test_inference.ipynb`

The config-setup cell replaces its fixed-path config/best-dir resolution with: set `config.paths.experiments_dir`, scan for the most recent training run matching the current `experiment_name`, then create this run's own timestamped `inference/` subfolder underneath it:

```python
config.data.root_dir = str(project_root / "data")
config.paths.experiments_dir = str(project_root / "experiments")

experiments_dir = Path(config.paths.experiments_dir)
candidates = sorted(
    p for p in experiments_dir.glob(f"*_{config.experiment_name}")
    if (p / "models" / "best").is_dir()
)
assert candidates, (
    f"No completed training runs found under {experiments_dir} matching "
    f"*_{config.experiment_name} (looked for a models/best/ subfolder). "
    "Run dgd_training_demo.ipynb first."
)
run_dir = candidates[-1]  # newest, since the timestamp prefix sorts lexicographically
print(f"Using training run: {run_dir}")

config.paths.models_dir = str(run_dir / "models")

# Guard: make sure this notebook's config matches the config actually used
# during training, so `test_loader` below is exactly the held-out split
# carved out at training time -- not a silently different split produced by
# a config.yaml that has since changed (random_seed, subset fraction, or
# split ratios).
trained_cfg = OmegaConf.load(run_dir / "config.yaml")
assert config.random_seed == trained_cfg.random_seed, (
    f"random_seed differs from the training run ({config.random_seed} vs {trained_cfg.random_seed}); "
    "the test split would not match the one carved out during training."
)
for key in ['total_subset_fraction', 'val_split', 'test_split']:
    assert config.data[key] == trained_cfg.data[key], (
        f"data.{key} differs from the training run ({config.data[key]} vs {trained_cfg.data[key]}); "
        "the test split would not match the one carved out during training."
    )

train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
print(f"Test loader: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
```

`best_dir` (used to load the frozen decoder+GMM) becomes `Path(config.paths.models_dir) / "best"` (no more `/ config.experiment_name` — `models_dir` already points inside the run folder).

This run's own inference output folder is created once, right after `test_rep` finishes optimizing (replacing the current `test_inference_dir` cell):

```python
from datetime import datetime

inference_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
inference_dir = run_dir / "inference" / inference_timestamp
inference_dir.mkdir(parents=True, exist_ok=True)

OmegaConf.save(config, str(inference_dir / "config.yaml"))

test_rep.save(str(inference_dir / "test_representation.pt"))
print(f"Saved optimized test representations to {inference_dir / 'test_representation.pt'}")
```

The final figures cell's `figures_dir` becomes `inference_dir / "figures"` instead of `Path(config.paths.figures_dir) / f"{config.experiment_name}_test_inference"`.

## 3. Inference-specific config (same file, no new keys elsewhere)

### `config/config.yaml`

```yaml
  inference:
    epochs: 200                    # M: total optimization steps
    prior_warmup_steps: 0          # M0: steps before the GMM prior term is added
    latent_noise_scale: 0.1        # Independent of training's latent_noise_scale (0.0 = disabled)
    latent_noise_start: 1.0        # Starting noise scale (step 1)
    latent_noise_end: 0.01         # Ending noise scale (final step M)
```

### `notebooks/dgd_test_inference.ipynb`

The M-step loop cell's noise-schedule setup currently reads `config.training.latent_noise_scale`/`latent_noise_start`/`latent_noise_end` (reusing training's verbatim). It changes to read `config.training.inference.latent_noise_scale`/`.latent_noise_start`/`.latent_noise_end` instead — same cosine-anneal formula, same per-step application before `y = decoder(z)`, just sourced from the new inference-scoped keys. No other change to the loop's structure.

## 4. Inference figure parity

### `src/visualization/loss.py`

New function, sibling to `plot_training_analysis`/`plot_training_dynamics`:

```python
def plot_inference_analysis(
    step_losses: List[float],
    step_recon: List[float],
    step_gmm: List[float],
    step_noise: List[float],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the M-step inference optimization: total loss, reconstruction loss,
    GMM error, and noise scale, each vs. optimization step m. Unlike
    plot_training_analysis, there is no train/val split (a single
    representation layer is optimized) and no learning-rate/momentum/epoch-
    timing panel (Algorithm 2 has no LR schedule or per-epoch timing).
    """
```

2×2 layout: (1) total loss vs. step, (2) reconstruction loss vs. step, (3) GMM error vs. step, (4) noise scale vs. step. Each panel single-line (no train/val distinction — there's only the one optimized `test_rep`), `save_path`/`show` following the existing convention.

### `src/visualization/report.py`

The GMM-component sample grid loop currently lives inline at the bottom of `generate_training_figures` (lines ~155-171). It's extracted into a shared helper:

```python
def _plot_gmm_component_samples(gmm, decoder, figures_dir: Path, device: torch.device) -> None:
    """Write one GMM-component sample grid PNG per component, sorted by weight descending."""
```

`generate_training_figures` calls this helper instead of inlining the loop (behavior unchanged). A new function reuses it for inference:

```python
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
) -> None:
    """
    Write latent-space, reconstruction, loss-curve, and GMM-component-sample
    figures for one inference run, mirroring generate_training_figures'
    output categories but for a single optimized representation layer
    instead of a train/val pair across many checkpoints.
    """
```

Writes: `latent_test.png` (via `plot_latent_space`, same as today), `recon_test.png` (via `plot_images_by_class`, same as today), `loss_curve.png` (via the new `plot_inference_analysis`, fed from `step_history['loss']`/`['recon']`/`['gmm']`/`['noise']`), and the GMM-component grids via `_plot_gmm_component_samples`.

### `notebooks/dgd_test_inference.ipynb`

The M-step loop cell collects per-step history instead of only printing it:

```python
step_history = {'loss': [], 'recon': [], 'gmm': [], 'noise': []}

for m in range(1, M + 1):
    ...
    step_history['loss'].append(total_loss / n_test)
    step_history['recon'].append(total_recon / n_test)
    step_history['gmm'].append(total_gmm / n_test)
    step_history['noise'].append(noise_scale_m)

    if m % max(1, M // 10) == 0 or m == M:
        print(f"Step {m}/{M}: loss={total_loss/n_test:.4f}, recon={total_recon/n_test:.4f}, gmm={total_gmm/n_test:.4f}, noise={noise_scale_m:.4f}")
```

The final figures cell computes `test_ami`/`test_ari`, prints them, then calls `generate_inference_figures(figures_dir, decoder, gmm, test_rep, test_labels, class_names, sample_data, step_history, device)` in place of the current inline `plot_latent_space`/`plot_images_by_class` calls. `sample_data` here is exactly the 3-tuple `(indices_test_sample, images_test_sample, labels_test_sample)` already built by `collect_class_samples` earlier in the cell — deliberately a 3-tuple, not `generate_training_figures`' 6-tuple (which carries a train half and a val half); `generate_inference_figures` has only one split to plot. Internally, `generate_inference_figures` computes the reconstruction the same way `_plot_checkpoint_figures` does — `decoder(test_rep(indices_test_sample.to(device)))` — rather than receiving pre-computed reconstructed images.

## 5. Two visualization bug fixes — `src/visualization/latent.py`

**GMM-PCA overlay.** `_add_gmm_overlay_pca` currently branches on `TGMM_PLOTTING_AVAILABLE` to get the covariance matrix via `tgmm.plotting.get_covariance_matrix(gmm, i)`. That helper is correct for `full`/`diag`/`tied_full`/`tied_diag` (it returns the true `n_features × n_features` matrix for those), but for `spherical`/`tied_spherical` it hardcodes a `2×2` identity-scaled matrix — a design choice specific to tgmm's own already-2D-native plotting, not applicable when the GMM has `n_features > 2` and is being projected through PCA here. The existing fallback branch (`else:` — used only when `tgmm.plotting` fails to import) already builds the correct full-dimensional matrix for every covariance type, spherical/tied_spherical included. Fix: stop calling `get_covariance_matrix` for the covariance matrix specifically — always use the local fallback logic (the `if gmm.covariance_type == 'full': ... elif ...` block) regardless of `TGMM_PLOTTING_AVAILABLE`. Keep using tgmm's `ensure_tensor_on_cpu`/`create_colormap` helpers for means/weights/colors where they're already correct — only the covariance-matrix extraction path changes.

**TSNE perplexity warning.** `TSNE(n_components=n_components, perplexity=30, max_iter=1000)` (both the cuML and sklearn branches) relies on cuML's default `n_neighbors=90`, which sits exactly on `3 * perplexity`. cuML's internal check appears to require strictly more than that boundary (confirmed empirically: the default 90 triggers the warning; an explicit 91 does not). Fix: pass `n_neighbors=3 * 30 + 1` explicitly in both `TSNE(...)` constructions (cuML and sklearn branches, for consistency, even though the warning is cuML-specific — sklearn's TSNE silently ignores an out-of-range perplexity/n_neighbors combination differently and isn't affected either way).

## Testing / verification approach

Consistent with the rest of this codebase: standalone runnable Python scripts against a tiny real-data config, run via `/home/asp/.venvs/rapids_cu13/bin/python`, no pytest suite.

- A tiny end-to-end training run confirming: `ami_scores`/`val_ami_scores`/`ari_scores`/`val_ari_scores` present in `training_results.pth` with expected length; no `nmi_scores`/`ari_scores`-only-partial keys; checkpoint metadata has `train_ami`/`val_ami`/`train_ari`/`val_ari`; the run folder is created under `experiments/<timestamp>_<name>/` with `config.yaml`, `models/`, and (after `generate_training_figures`) `figures/` all present; a second run with the same `experiment_name` produces a *different* timestamped folder and does not touch the first run's contents.
- `generate_training_figures` run against that tiny run's folder, confirming `loss_curves.png` renders (4-line single-axis AMI/ARI panel) without error.
- A tiny end-to-end inference run against that training run: confirms the "most recent run" scan picks the correct folder, the reproducibility assertion passes, `inference/<timestamp>/` is created with its own `config.yaml`/`test_representation.pt`/`figures/`, `loss_curve.png` renders from the collected step history, GMM-component grids are written, and `test_ami`/`test_ari` are both printed and both appear in the latent-space plot title.
- `_add_gmm_overlay_pca` exercised directly against a `tied_spherical` GMM with `n_features > 2` (matching the real config), confirming no exception and that the resulting figure actually has ellipse patches drawn (not just "no crash").
- `plot_latent_space`'s `TSNE` call exercised with output captured, confirming the nearest-neighbors warning no longer appears, for both a large (thousands of samples) and small dataset.

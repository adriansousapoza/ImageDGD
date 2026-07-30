# AMI+ARI, Timestamped Experiment Folders, Inference Figure Parity, and Viz Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap the recently-introduced NMI metric back out for AMI+ARI tracked side by side, give every training/inference run its own timestamped folder so nothing gets overwritten, bring the inference notebook's figures to parity with training's, give inference its own tunable noise/epoch config, and fix two visualization bugs (GMM-PCA overlay crash, TSNE perplexity warning).

**Architecture:** Nine sequential tasks across `config/config.yaml`, `src/training/trainer.py`, `src/visualization/{loss,report,latent}.py`, and both notebooks. Each notebook is edited via a small Python script that rewrites specific cells by index (never by hand), matching this project's established convention.

**Tech Stack:** PyTorch, `tgmm.ClusteringMetrics` (`adjusted_mutual_info_score`, `adjusted_rand_score`), OmegaConf, cuML/sklearn (PCA/UMAP/TSNE), Jupyter notebook JSON.

**Spec:** `docs/superpowers/specs/2026-07-30-ami-ari-experiment-folders-and-viz-fixes-design.md` — read it if any task step below seems to conflict with this plan; this plan is a direct implementation of it.

## Global Constraints

- Use `/home/asp/.venvs/rapids_cu13/bin/python` for every verification command in this plan — the only venv with the full stack (torch, tgmm, cudf, cuml) needed to import this project's `src` package.
- No pytest suite in this project. Verification is standalone runnable Python scripts against a tiny real-data config (`total_subset_fraction` small, a scratch `experiment_name`, paths pointed at `/tmp`), matching every prior plan in this repo.
- `sklearn.manifold.TSNE` has **no** `n_neighbors` parameter (confirmed against its signature). The perplexity-warning fix in Task 1 applies **only** to the cuML branch — adding `n_neighbors` to the sklearn branch would raise `TypeError`.
- `config.paths.models_dir`/`config.paths.figures_dir` are no longer static config.yaml defaults after this plan — they become runtime-resolved paths the calling notebook sets to point inside a timestamped run folder (`config.paths.experiments_dir` is the one static default). Every place that previously did `Path(config.paths.models_dir) / config.experiment_name` must drop the `/ config.experiment_name` join — `models_dir` already points inside a uniquely-named folder. This applies in three places: `src/training/trainer.py`'s `train()`, `notebooks/dgd_training_demo.ipynb`'s figures cell, and `notebooks/dgd_test_inference.ipynb`'s checkpoint-loading cell.
- The `config.yaml` copy for a training run moves from `best/config.yaml` (written by `trainer.py` at the end of training) to the run folder's root, `run_dir/config.yaml` (written by the calling notebook once, when the run folder is created, before training starts). No file keeps a copy in both places.
- NMI is fully removed, not aliased — every `nmi_scores`/`val_nmi_scores`/`current_train_nmi`/`current_val_nmi`/`train_nmi`/`val_nmi` symbol from the prior plan becomes `ami_scores`/`ari_scores` (and their `val_`/`current_`/checkpoint-metadata counterparts). Silhouette stays removed — this plan does not resurrect it.
- Every file this plan touches was last modified by the noise-everywhere-and-NMI plan (commits `295e7c5`..`01a6055`). This plan's edits are written against that content — read the file before editing if picking this plan up cold.

---

### Task 1: Fix two visualization bugs — `src/visualization/latent.py`

**Files:**
- Modify: `src/visualization/latent.py` (`_add_gmm_overlay_pca` covariance extraction, and the `TSNE(...)` construction inside `plot_latent_space`)

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature change to `plot_latent_space` or `_add_gmm_overlay_pca`. Both now work correctly for `spherical`/`tied_spherical` GMMs with `n_features > 2`, and the cuML TSNE branch no longer emits the nearest-neighbors warning.

- [ ] **Step 1: Stop routing the GMM-PCA overlay's covariance matrix through tgmm's 2D-only helper**

`tgmm.plotting.get_covariance_matrix` hardcodes a 2×2 result for `spherical`/`tied_spherical` covariance types (it's built for GMMs already living in 2D, for tgmm's own native plotting). `_add_gmm_overlay_pca` needs the full `n_features × n_features` matrix to project through the `n_components × n_features` PCA components matrix — using the 2×2 result crashes with a matmul dimension mismatch. The function already has a correct fallback (used only when `tgmm.plotting` fails to import) that builds the right-sized matrix for every covariance type. Always use that logic instead.

Replace:
```python
            # Get covariance matrix using tgmm helper if available
            if TGMM_PLOTTING_AVAILABLE:
                cov = get_covariance_matrix(gmm, i).numpy()
            else:
                # Fallback implementation
                covariances = gmm.covariances_.detach().cpu().numpy() if isinstance(gmm.covariances_, torch.Tensor) else gmm.covariances_
                
                # Handle different covariance types
                if gmm.covariance_type == 'full':
                    cov = covariances[i]
                elif gmm.covariance_type == 'diag':
                    cov = np.diag(covariances[i])
                elif gmm.covariance_type == 'spherical':
                    cov = covariances[i] * np.eye(n_features)
                elif gmm.covariance_type == 'tied_full':
                    cov = covariances
                elif gmm.covariance_type == 'tied_diag':
                    cov = np.diag(covariances)
                elif gmm.covariance_type == 'tied_spherical':
                    cov = covariances.item() * np.eye(n_features) if isinstance(covariances, np.ndarray) else covariances * np.eye(n_features)
                else:
                    # Unknown type, skip
                    print(f"Warning: Unknown covariance type {gmm.covariance_type}")
                    continue
```
with:
```python
            # Always build the covariance matrix locally rather than via
            # tgmm.plotting.get_covariance_matrix: that helper hardcodes a
            # 2x2 result for spherical/tied_spherical types (it's designed
            # for GMMs already living in 2D, for tgmm's own native
            # plotting). Here the GMM has n_features dimensions and needs
            # the full n_features x n_features matrix to project through
            # PCA correctly.
            covariances = gmm.covariances_.detach().cpu().numpy() if isinstance(gmm.covariances_, torch.Tensor) else gmm.covariances_

            # Handle different covariance types
            if gmm.covariance_type == 'full':
                cov = covariances[i]
            elif gmm.covariance_type == 'diag':
                cov = np.diag(covariances[i])
            elif gmm.covariance_type == 'spherical':
                cov = covariances[i] * np.eye(n_features)
            elif gmm.covariance_type == 'tied_full':
                cov = covariances
            elif gmm.covariance_type == 'tied_diag':
                cov = np.diag(covariances)
            elif gmm.covariance_type == 'tied_spherical':
                cov = covariances.item() * np.eye(n_features) if isinstance(covariances, np.ndarray) else covariances * np.eye(n_features)
            else:
                # Unknown type, skip
                print(f"Warning: Unknown covariance type {gmm.covariance_type}")
                continue
```

- [ ] **Step 2: Silence the cuML TSNE perplexity warning**

Replace (inside `plot_latent_space`, the t-SNE `try:` block):
```python
    try:
        if using_cuml:
            # cuML TSNE: n_neighbors should be at least 3 * perplexity
            # Don't pass random_state to avoid brute_force_knn warning
            tsne = TSNE(n_components=n_components, perplexity=30, max_iter=1000)
        else:
            tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=30, max_iter=1000)
```
with:
```python
    try:
        if using_cuml:
            # cuML's default n_neighbors (90) sits exactly on 3*perplexity
            # (30), and cuML's internal check requires strictly more than
            # that boundary -- the default always triggers a spurious "# of
            # Nearest Neighbors should be at least 3 * perplexity" warning.
            # One above the boundary silences it with no effect on results.
            # sklearn.manifold.TSNE has no n_neighbors parameter at all, so
            # this only applies to the cuML branch.
            # Don't pass random_state to avoid brute_force_knn warning
            tsne = TSNE(n_components=n_components, perplexity=30, max_iter=1000, n_neighbors=3 * 30 + 1)
        else:
            tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=30, max_iter=1000)
```

- [ ] **Step 3: Verify — GMM overlay renders without crashing, ellipses are actually drawn**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
import matplotlib
matplotlib.use('Agg')
import torch
from tgmm import GaussianMixture
from src.visualization.latent import plot_latent_space

n_features = 8
representations = torch.randn(500, n_features)
labels = torch.randint(0, 5, (500,))

gmm = GaussianMixture(n_components=5, n_features=n_features, covariance_type='tied_spherical', device=torch.device('cpu'))
gmm.fit(representations)

fig = plot_latent_space(
    representations=representations, labels=labels, gmm=gmm,
    class_names=[str(i) for i in range(5)],
    save_path='/tmp/plan_smoke_latent_overlay.png', show=False,
)
ax_pca = fig.axes[0]
n_ellipses = len(ax_pca.patches)
assert n_ellipses == gmm.n_components, f'expected {gmm.n_components} GMM ellipses on the PCA plot, got {n_ellipses}'
print(f'OK: {n_ellipses} GMM ellipses rendered on PCA plot for tied_spherical, no crash')
"
```
Expected: `OK: 5 GMM ellipses rendered on PCA plot for tied_spherical, no crash`.

- [ ] **Step 4: Verify — no perplexity warning from cuML TSNE**

Run as a subprocess so cuML's C++-side logger output is reliably captured regardless of Python-level stdout redirection:

```bash
cat > /tmp/plan_smoke_tsne_check.py << 'PYEOF'
import torch
import matplotlib
matplotlib.use('Agg')
from src.visualization.latent import plot_latent_space

representations = torch.randn(2000, 8)
labels = torch.randint(0, 5, (2000,))
plot_latent_space(representations=representations, labels=labels, save_path='/tmp/plan_smoke_latent_tsne.png', show=False)
print("DONE")
PYEOF
PYTHONPATH=/home/asp/Downloads/HeaDS/ImageDGD /home/asp/.venvs/rapids_cu13/bin/python /tmp/plan_smoke_tsne_check.py > /tmp/plan_smoke_tsne_output.txt 2>&1
cat /tmp/plan_smoke_tsne_output.txt
grep -qi "nearest neighbors" /tmp/plan_smoke_tsne_output.txt && echo "FAIL: warning still present" || echo "OK: no nearest-neighbors warning"
grep -q "DONE" /tmp/plan_smoke_tsne_output.txt && echo "OK: completed without error"
```
Expected: `OK: no nearest-neighbors warning` and `OK: completed without error`, no `FAIL` line.

- [ ] **Step 5: Commit**

```bash
git add src/visualization/latent.py
git commit -m "fix: correct GMM-PCA overlay covariance shape, silence TSNE perplexity warning"
```

---

### Task 2: Timestamped experiment folders — `config/config.yaml`

**Files:**
- Modify: `config/config.yaml`

**Interfaces:**
- Produces: `paths.experiments_dir` (replacing `paths.models_dir`/`paths.figures_dir` as the static default). `training.inference.latent_noise_scale`/`.latent_noise_start`/`.latent_noise_end` (new keys, independent of the top-level `training.latent_noise_scale`/etc). Tasks 8 and 9 (notebooks) consume `experiments_dir` by exact name; Task 9 consumes the three new inference noise keys by exact name.

- [ ] **Step 1: Replace `models_dir`/`figures_dir` with `experiments_dir`**

Replace (`config/config.yaml`):
```yaml
# Filesystem roots for checkpoints and figures. Notebooks overwrite these with
# absolute paths derived from project_root at runtime (same pattern as data.root_dir).
paths:
  models_dir: "./models"
  figures_dir: "./figures"
```
with:
```yaml
# Filesystem root for timestamped experiment run folders. Notebooks overwrite
# this with an absolute path derived from project_root at runtime (same
# pattern as data.root_dir), then resolve config.paths.models_dir/figures_dir
# to point inside that run's own timestamped folder before training/inference
# starts, so no run ever overwrites another's checkpoints or figures.
paths:
  experiments_dir: "./experiments"
```

- [ ] **Step 2: Add inference-specific noise keys**

Replace:
```yaml
  # Post-hoc inference (Algorithm 2): optimizing representations for the
  # held-out test split against a frozen decoder + GMM. M0 (prior_warmup_steps)
  # must be < epochs — a fresh z far from every mode is dominated by the prior
  # gradient before reconstruction has informed it where to go; a short
  # reconstruction-only warm-up lets it find its basin first.
  inference:
    epochs: 200            # M: total optimization steps
    prior_warmup_steps: 0 # M0: steps before the GMM prior term is added
```
with:
```yaml
  # Post-hoc inference (Algorithm 2): optimizing representations for the
  # held-out test split against a frozen decoder + GMM. M0 (prior_warmup_steps)
  # must be < epochs — a fresh z far from every mode is dominated by the prior
  # gradient before reconstruction has informed it where to go; a short
  # reconstruction-only warm-up lets it find its basin first.
  inference:
    epochs: 200            # M: total optimization steps
    prior_warmup_steps: 0 # M0: steps before the GMM prior term is added
    # Independent of the top-level latent_noise_scale/_start/_end above --
    # inference noise can be tuned without touching training's config.
    latent_noise_scale: 0.1   # Enable noise injection (0.0 = disabled)
    latent_noise_start: 1.0  # Starting noise scale (step 1)
    latent_noise_end: 0.01     # Ending noise scale (final step M)
```

- [ ] **Step 3: Verify — config loads with the new keys, old keys gone**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from omegaconf import OmegaConf
config = OmegaConf.load('config/config.yaml')
assert config.paths.experiments_dir == './experiments'
assert 'models_dir' not in config.paths
assert 'figures_dir' not in config.paths
assert config.training.inference.epochs == 200
assert config.training.inference.prior_warmup_steps == 0
assert config.training.inference.latent_noise_scale == 0.1
assert config.training.inference.latent_noise_start == 1.0
assert config.training.inference.latent_noise_end == 0.01
print('OK: config.yaml has experiments_dir and inference noise keys, old path keys gone')
"
```
Expected: `OK: ...`.

- [ ] **Step 4: Commit**

```bash
git add config/config.yaml
git commit -m "feat: replace models_dir/figures_dir with experiments_dir, add inference noise config"
```

---

### Task 3: Metric swap NMI → AMI + ARI — `src/training/trainer.py`

**Files:**
- Modify: `src/training/trainer.py` (`__init__` tracking lists; both GMM-refit branches; per-epoch log line; `training_results.pth` schema)

**Interfaces:**
- Consumes: `tgmm.ClusteringMetrics.adjusted_mutual_info_score(labels_true, labels_pred) -> float` and `.adjusted_rand_score(labels_true, labels_pred) -> float` (already imported via `from tgmm import GaussianMixture, ClusteringMetrics`).
- Produces: `self.ami_scores`, `self.val_ami_scores`, `self.ari_scores`, `self.val_ari_scores: List[float]` (replacing `self.nmi_scores`/`self.val_nmi_scores`). Checkpoint metadata keys `train_ami`/`val_ami`/`train_ari`/`val_ari` (replacing `train_nmi`/`val_nmi`). `training_results.pth` keys `ami_scores`/`val_ami_scores`/`ari_scores`/`val_ari_scores` (replacing `nmi_scores`/`val_nmi_scores`) — Task 5 (`loss.py`) and Task 6 (`report.py`) consume these by exact name.

- [ ] **Step 1: Rename the tracking lists**

Replace (`trainer.py`, in `__init__`):
```python
        # Clustering metrics tracking
        self.nmi_scores = []
        self.val_nmi_scores = []
```
with:
```python
        # Clustering metrics tracking
        self.ami_scores = []
        self.val_ami_scores = []
        self.ari_scores = []
        self.val_ari_scores = []
```

- [ ] **Step 2: Replace NMI computation with AMI+ARI in both GMM-refit branches**

Replace the full block from the clustering-metrics-calculator comment through the end of the `elif epoch > first_epoch_gmm:` branch:
```python
            # Initialize clustering metrics calculator
            cluster_metrics = ClusteringMetrics()
            current_train_nmi = 0.0
            current_val_nmi = 0.0

            is_gmm_refit_epoch = epoch == first_epoch_gmm or (refit_gmm_interval and epoch % refit_gmm_interval == 0)

            if is_gmm_refit_epoch:
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=1000 if epoch == first_epoch_gmm else 100)

                    # Calculate NMI for training data
                    predicted_labels = gmm.predict(representations)
                    current_train_nmi = cluster_metrics.normalized_mutual_info_score(train_labels, predicted_labels)
                    self.nmi_scores.append(current_train_nmi)

                    # Calculate NMI for val data
                    val_representations = val_rep.z.detach()
                    val_predicted_labels = gmm.predict(val_representations)
                    current_val_nmi = cluster_metrics.normalized_mutual_info_score(val_labels, val_predicted_labels)
                    self.val_nmi_scores.append(current_val_nmi)

                # Persist a checkpoint at every GMM-refit epoch
                save_checkpoint(
                    checkpoint_root / f"epoch_{epoch:04d}",
                    model.decoder, rep, val_rep, gmm,
                    metadata={
                        'epoch': epoch,
                        'train_nmi': current_train_nmi,
                        'val_nmi': current_val_nmi,
                    }
                )

                # Activate early stopping after first GMM fit
                if epoch == first_epoch_gmm:
                    self.early_stopping_active = True
                    self.best_train_loss = float('inf')  # Reset best loss
                    self.best_val_loss = float('inf')  # Reset val loss too (GMM adds error term)
                    self.epochs_without_improvement = 0
            elif epoch > first_epoch_gmm:
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=100, warm_start=True)

                    # Calculate NMI for training data
                    predicted_labels = gmm.predict(representations)
                    current_train_nmi = cluster_metrics.normalized_mutual_info_score(train_labels, predicted_labels)
                    self.nmi_scores.append(current_train_nmi)

                    # Calculate NMI for val data
                    val_representations = val_rep.z.detach()
                    val_predicted_labels = gmm.predict(val_representations)
                    current_val_nmi = cluster_metrics.normalized_mutual_info_score(val_labels, val_predicted_labels)
                    self.val_nmi_scores.append(current_val_nmi)
```
with:
```python
            # Initialize clustering metrics calculator
            cluster_metrics = ClusteringMetrics()
            current_train_ami = 0.0
            current_val_ami = 0.0
            current_train_ari = 0.0
            current_val_ari = 0.0

            is_gmm_refit_epoch = epoch == first_epoch_gmm or (refit_gmm_interval and epoch % refit_gmm_interval == 0)

            if is_gmm_refit_epoch:
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=1000 if epoch == first_epoch_gmm else 100)

                    # Calculate AMI and ARI for training data
                    predicted_labels = gmm.predict(representations)
                    current_train_ami = cluster_metrics.adjusted_mutual_info_score(train_labels, predicted_labels)
                    self.ami_scores.append(current_train_ami)
                    current_train_ari = cluster_metrics.adjusted_rand_score(train_labels, predicted_labels)
                    self.ari_scores.append(current_train_ari)

                    # Calculate AMI and ARI for val data
                    val_representations = val_rep.z.detach()
                    val_predicted_labels = gmm.predict(val_representations)
                    current_val_ami = cluster_metrics.adjusted_mutual_info_score(val_labels, val_predicted_labels)
                    self.val_ami_scores.append(current_val_ami)
                    current_val_ari = cluster_metrics.adjusted_rand_score(val_labels, val_predicted_labels)
                    self.val_ari_scores.append(current_val_ari)

                # Persist a checkpoint at every GMM-refit epoch
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

                # Activate early stopping after first GMM fit
                if epoch == first_epoch_gmm:
                    self.early_stopping_active = True
                    self.best_train_loss = float('inf')  # Reset best loss
                    self.best_val_loss = float('inf')  # Reset val loss too (GMM adds error term)
                    self.epochs_without_improvement = 0
            elif epoch > first_epoch_gmm:
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=100, warm_start=True)

                    # Calculate AMI and ARI for training data
                    predicted_labels = gmm.predict(representations)
                    current_train_ami = cluster_metrics.adjusted_mutual_info_score(train_labels, predicted_labels)
                    self.ami_scores.append(current_train_ami)
                    current_train_ari = cluster_metrics.adjusted_rand_score(train_labels, predicted_labels)
                    self.ari_scores.append(current_train_ari)

                    # Calculate AMI and ARI for val data
                    val_representations = val_rep.z.detach()
                    val_predicted_labels = gmm.predict(val_representations)
                    current_val_ami = cluster_metrics.adjusted_mutual_info_score(val_labels, val_predicted_labels)
                    self.val_ami_scores.append(current_val_ami)
                    current_val_ari = cluster_metrics.adjusted_rand_score(val_labels, val_predicted_labels)
                    self.val_ari_scores.append(current_val_ari)
```

- [ ] **Step 3: Update the per-epoch log line**

Replace:
```python
            train_nmi_str = f", NMI={current_train_nmi:.4f}" if epoch >= first_epoch_gmm else ""
            val_nmi_str = f", NMI={current_val_nmi:.4f}" if epoch >= first_epoch_gmm else ""

            print(f"Epoch {epoch}/{self.training_config.epochs} [Time per Epoch: {epoch_time_str}, Remaining Time: {remaining_time_str}, LR: Dec={lr_decoder:.2e}, Rep={lr_rep:.2e}, Noise={noise_scale:.4f}]")
            print(f"       - Train Loss: {train_loss:.4f} (B: {self.best_train_loss:.4f}), Recon: {recon_train_loss:.4f} (B: {self.best_recon_train:.4f}), GMM: {gmm_train_str}{train_nmi_str}")
            print(f"       - Val   Loss: {val_loss:.4f} (B: {self.best_val_loss:.4f}), Recon: {recon_val_loss:.4f} (B: {self.best_recon_val:.4f}), GMM: {gmm_val_str}{val_nmi_str}")
```
with:
```python
            train_ami_ari_str = f", AMI={current_train_ami:.4f}, ARI={current_train_ari:.4f}" if epoch >= first_epoch_gmm else ""
            val_ami_ari_str = f", AMI={current_val_ami:.4f}, ARI={current_val_ari:.4f}" if epoch >= first_epoch_gmm else ""

            print(f"Epoch {epoch}/{self.training_config.epochs} [Time per Epoch: {epoch_time_str}, Remaining Time: {remaining_time_str}, LR: Dec={lr_decoder:.2e}, Rep={lr_rep:.2e}, Noise={noise_scale:.4f}]")
            print(f"       - Train Loss: {train_loss:.4f} (B: {self.best_train_loss:.4f}), Recon: {recon_train_loss:.4f} (B: {self.best_recon_train:.4f}), GMM: {gmm_train_str}{train_ami_ari_str}")
            print(f"       - Val   Loss: {val_loss:.4f} (B: {self.best_val_loss:.4f}), Recon: {recon_val_loss:.4f} (B: {self.best_recon_val:.4f}), GMM: {gmm_val_str}{val_ami_ari_str}")
```

- [ ] **Step 4: Update the `training_results.pth` schema**

Replace:
```python
        torch.save({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_train_losses': self.recon_train_losses,
            'recon_val_losses': self.recon_val_losses,
            'gmm_train_losses': self.gmm_train_losses,
            'gmm_val_losses': self.gmm_val_losses,
            'nmi_scores': self.nmi_scores,
            'val_nmi_scores': self.val_nmi_scores,
            'learning_rates': self.learning_rates,
            'momentum_betas': self.momentum_betas,
            'epoch_times': self.epoch_times,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }, best_dir / "training_results.pth")
```
with:
```python
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
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }, best_dir / "training_results.pth")
```

- [ ] **Step 5: Verify — no NMI symbols remain, AMI+ARI schema present with correct lengths**

```bash
grep -n "nmi\|normalized_mutual_info_score" src/training/trainer.py
```
Expected: no output (empty).

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from pathlib import Path
from omegaconf import OmegaConf
import torch
from src.data import create_dataloaders
from src.training import DGDTrainer

config = OmegaConf.load('config/config.yaml')
config.data.total_subset_fraction = 0.02
config.data.download = False
config.experiment_name = 'plan_smoke_ami_ari'
config.paths.models_dir = '/tmp/plan_smoke_models_ami_ari'
config.training.epochs = 3
config.training.first_epoch_gmm = 1
# Larger than epochs, so the only refit-checkpoint epoch is epoch 1 (via the
# epoch == first_epoch_gmm match). This deliberately keeps the final epoch
# (3) from ALSO being a refit-checkpoint epoch: save_checkpoint does a plain
# torch.save with no merge, so if the final epoch coincided with a refit
# checkpoint, the unconditional loss-only final-epoch save (below, after the
# training loop) would silently overwrite that epoch's AMI/ARI metadata with
# {'epoch', 'train_loss', 'val_loss'} only -- a pre-existing, out-of-scope
# bug in the final-checkpoint save that predates this task and is not this
# task's to fix (same behavior existed for ari/silhouette, and then nmi,
# before this plan). Avoiding the collision here, rather than reading around
# it, keeps this verification meaningful without tempting an implementer to
# "fix" the collision by touching the final-checkpoint save -- which is
# exactly the out-of-scope edit this task must NOT make.
config.training.refit_gmm_interval = 5
config.training.early_stopping_patience = 100
config.training.latent_noise_scale = 0.0

device = torch.device('cpu')
train_loader, val_loader, test_loader, class_names = create_dataloaders(config)

trainer = DGDTrainer(config=config, device=device, verbose=False)
trainer.train(train_loader, val_loader, sample_data=None, class_names=class_names)

assert len(trainer.ami_scores) == 3 and len(trainer.val_ami_scores) == 3
assert len(trainer.ari_scores) == 3 and len(trainer.val_ari_scores) == 3
assert not hasattr(trainer, 'nmi_scores')

# NOTE: Task 4 (not yet applied at this point in the plan) is what removes
# the "/ experiment_name" join from trainer.py's experiment_dir computation.
# Until then, trainer.py still writes under models_dir/experiment_name, so
# this verification reads from there to match trainer.py's actual current
# behavior -- not the post-Task-4 layout.
run_experiment_dir = Path(config.paths.models_dir) / config.experiment_name
best_dir = run_experiment_dir / 'best'
history = torch.load(best_dir / 'training_results.pth')
for key in ['ami_scores', 'val_ami_scores', 'ari_scores', 'val_ari_scores']:
    assert key in history and len(history[key]) == 3, f'{key} missing or wrong length'
assert 'nmi_scores' not in history

# Read the epoch-1 checkpoint specifically (the only refit-checkpoint
# epoch under this config) rather than the final checkpoint -- the final
# checkpoint (epoch_0003) is written by the loss-only final-epoch save and,
# by design of trainer.py (not this task's concern), never carries AMI/ARI
# metadata.
refit_meta = torch.load(run_experiment_dir / 'checkpoints' / 'epoch_0001' / 'metadata.pth')
for key in ['train_ami', 'val_ami', 'train_ari', 'val_ari']:
    assert key in refit_meta, f'checkpoint metadata missing {key}'
final_meta = torch.load(run_experiment_dir / 'checkpoints' / 'epoch_0003' / 'metadata.pth')
assert 'train_ami' not in final_meta and 'val_ami' not in final_meta and 'train_ari' not in final_meta and 'val_ari' not in final_meta, (
    'final-epoch checkpoint metadata must stay loss-only ({epoch, train_loss, val_loss}) -- '
    'if AMI/ARI keys appear here, the final-checkpoint save in trainer.py was touched, which is out of scope for this task'
)
print('OK: AMI+ARI schema present, NMI gone, list lengths correct, final-checkpoint save untouched')
"
```
Expected: `OK: ...`.

- [ ] **Step 6: Commit**

```bash
git add src/training/trainer.py
git commit -m "refactor: swap NMI for AMI+ARI, tracked side by side"
```

---

### Task 4: Timestamped experiment folders — `src/training/trainer.py`

**Files:**
- Modify: `src/training/trainer.py` (`train()`'s `experiment_dir` computation; remove the end-of-training `OmegaConf.save` call; drop the now-unused `OmegaConf` import)

**Interfaces:**
- Consumes: `config.paths.models_dir`, now expected to already be a unique, run-specific path (set by the calling notebook — Task 8/9 — before `train()` is invoked), not `models_dir/experiment_name`.
- Produces: `train()`'s external behavior is otherwise unchanged (same return dict, same checkpoint/best-dir layout *inside* `experiment_dir`). No more `config.yaml` written into `best_dir` — Task 8 writes the run's single config copy at the run folder root before training starts.

- [ ] **Step 1: Drop the experiment_name join**

Replace:
```python
        experiment_dir = Path(self.config.paths.models_dir) / self.config.experiment_name
        checkpoint_root = experiment_dir / "checkpoints"
```
with:
```python
        # config.paths.models_dir is already a unique, timestamped path
        # (resolved by the calling notebook, e.g.
        # experiments/<timestamp>_<experiment_name>/models) -- no
        # experiment_name join needed here.
        experiment_dir = Path(self.config.paths.models_dir)
        checkpoint_root = experiment_dir / "checkpoints"
```

- [ ] **Step 2: Remove the end-of-training config copy**

Replace:
```python
        # Persist the best model: decoder, train/val representations, GMM, config, and loss history
        best_dir = experiment_dir / "best"
        save_checkpoint(
            best_dir,
            model.decoder, rep, val_rep, gmm,
            metadata={'best_epoch': self.best_epoch, 'best_val_loss': self.best_val_loss}
        )
        OmegaConf.save(self.config, str(best_dir / "config.yaml"))
        torch.save({
```
with:
```python
        # Persist the best model: decoder, train/val representations, GMM,
        # and loss history. The config.yaml copy for this run lives at the
        # run folder's root (written once by the calling notebook when the
        # run folder is created), not duplicated here.
        best_dir = experiment_dir / "best"
        save_checkpoint(
            best_dir,
            model.decoder, rep, val_rep, gmm,
            metadata={'best_epoch': self.best_epoch, 'best_val_loss': self.best_val_loss}
        )
        torch.save({
```

- [ ] **Step 3: Drop the now-unused `OmegaConf` import**

Replace:
```python
from omegaconf import DictConfig, OmegaConf
```
with:
```python
from omegaconf import DictConfig
```

- [ ] **Step 4: Verify — no double-nesting, no config.yaml in best/, everything else intact**

```bash
grep -n "OmegaConf\|experiment_name" src/training/trainer.py
```
Expected: no output (empty) — confirms both the import and the join are fully gone.

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from pathlib import Path
from omegaconf import OmegaConf
import torch
from src.data import create_dataloaders
from src.training import DGDTrainer

config = OmegaConf.load('config/config.yaml')
config.data.total_subset_fraction = 0.02
config.data.download = False
config.experiment_name = 'plan_smoke_folders'
run_dir = Path('/tmp/plan_smoke_run_20260101_000000_plan_smoke_folders')
config.paths.models_dir = str(run_dir / 'models')
config.training.epochs = 2
config.training.first_epoch_gmm = 1
config.training.refit_gmm_interval = 1
config.training.early_stopping_patience = 100
config.training.latent_noise_scale = 0.0

device = torch.device('cpu')
train_loader, val_loader, test_loader, class_names = create_dataloaders(config)

trainer = DGDTrainer(config=config, device=device, verbose=False)
trainer.train(train_loader, val_loader, sample_data=None, class_names=class_names)

models_dir = Path(config.paths.models_dir)
assert models_dir == run_dir / 'models', 'models_dir should be exactly what the caller set, no extra join'
assert not (models_dir / config.experiment_name).exists(), 'must not create a nested experiment_name subfolder'
assert (models_dir / 'checkpoints' / 'epoch_0000').is_dir()
assert (models_dir / 'best' / 'decoder.pth').exists()
assert (models_dir / 'best' / 'training_results.pth').exists()
assert not (models_dir / 'best' / 'config.yaml').exists(), 'trainer.py must no longer write a config.yaml copy'
print('OK: no experiment_name double-join, no config.yaml written by trainer.py')
"
```
Expected: `OK: ...`.

- [ ] **Step 5: Commit**

```bash
git add src/training/trainer.py
git commit -m "refactor: trainer.py accepts a pre-resolved run-specific models_dir"
```

---

### Task 5: AMI+ARI panel + inference analysis plot — `src/visualization/loss.py`

**Files:**
- Modify: `src/visualization/loss.py` (`plot_training_analysis` panel 4; new `plot_inference_analysis` function)

**Interfaces:**
- Consumes: a duck-typed `trainer` object exposing `ami_scores`/`val_ami_scores`/`ari_scores`/`val_ari_scores` (Task 3's renamed attributes, or the `SimpleNamespace` Task 6 builds from `training_results.pth`).
- Produces: `plot_training_analysis(...)` — same signature, panel 4 now single-axis with 4 lines. New: `plot_inference_analysis(step_losses, step_recon, step_gmm, step_noise, save_path=None, show=True) -> plt.Figure`. Task 7 (`report.py`) and Task 9 (inference notebook, via Task 7) consume `plot_inference_analysis` by exact name and parameter order.

- [ ] **Step 1: Replace the NMI panel with a single-axis AMI+ARI panel**

Replace (`loss.py`, the full "Clustering Quality (NMI)" block inside `plot_training_analysis`, through its closing `else`):
```python
    # 4. Clustering Quality (NMI)
    if hasattr(trainer, 'nmi_scores') and len(trainer.nmi_scores) > 0:
        nmi_scores = trainer.nmi_scores
        val_nmi_scores = trainer.val_nmi_scores if hasattr(trainer, 'val_nmi_scores') else []

        # Find epochs where metrics were computed (non-zero GMM epochs).
        # NOT filtered by start_idx here: nmi_scores/val_nmi_scores gain one
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

        ax_nmi = axes[3]

        if len(nmi_scores) > 0:
            x, y = _align(metric_epochs, nmi_scores)
            ax_nmi.plot(x, y, 'b-', label='Train NMI', linewidth=2, marker='o')
        if len(val_nmi_scores) > 0:
            x, y = _align(metric_epochs, val_nmi_scores)
            ax_nmi.plot(x, y, 'r-', label='Val NMI', linewidth=2, marker='o')

        ax_nmi.set_xlabel('Epoch')
        ax_nmi.set_ylabel('NMI Score')
        ax_nmi.set_title('Clustering Quality (NMI)')
        ax_nmi.legend(loc='best')
        ax_nmi.grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'No clustering metrics\navailable', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Clustering Quality (NMI)')
```
with:
```python
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
```

- [ ] **Step 2: Add `plot_inference_analysis`**

Add at the end of `loss.py`, after `plot_training_dynamics`:
```python


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
```

- [ ] **Step 3: Verify — AMI+ARI panel renders single-axis with 4 lines; `plot_inference_analysis` renders standalone**

```bash
grep -n "nmi" src/visualization/loss.py
```
Expected: no output (empty).

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from types import SimpleNamespace
from omegaconf import OmegaConf
from src.visualization.loss import plot_training_analysis, plot_inference_analysis

config = OmegaConf.create({'training': {'first_epoch_gmm': 2}})
trainer_view = SimpleNamespace(
    recon_train_losses=[1.0, 0.9, 0.8], recon_val_losses=[1.1, 1.0, 0.9],
    gmm_train_losses=[0.0, 0.5, 0.4], gmm_val_losses=[0.0, 0.6, 0.5],
    ami_scores=[0.1, 0.2], val_ami_scores=[0.08, 0.18],
    ari_scores=[0.05, 0.15], val_ari_scores=[0.04, 0.14],
)
fig = plot_training_analysis(
    train_losses=[2.0, 1.4, 1.2], val_losses=[2.1, 1.5, 1.3],
    trainer=trainer_view, config=config, save_path='/tmp/plan_smoke_loss_curves_ami_ari.png', show=False,
)
assert len(fig.axes) == 4
titles = [ax.get_title() for ax in fig.axes]
assert 'Clustering Quality (AMI & ARI)' in titles, titles
ax_metrics = fig.axes[3]
assert len(ax_metrics.lines) == 4, f'expected 4 lines (train/val AMI, train/val ARI), got {len(ax_metrics.lines)}'
print('OK: single-axis AMI+ARI panel with 4 lines')

fig2 = plot_inference_analysis(
    step_losses=[10.0, 8.0, 6.0], step_recon=[7.0, 6.0, 5.0],
    step_gmm=[3.0, 2.0, 1.0], step_noise=[1.0, 0.5, 0.1],
    save_path='/tmp/plan_smoke_inference_analysis.png', show=False,
)
assert len(fig2.axes) == 4
print('OK: plot_inference_analysis renders 4 panels')
"
```
Expected: `OK: single-axis AMI+ARI panel with 4 lines` and `OK: plot_inference_analysis renders 4 panels`.

- [ ] **Step 4: Commit**

```bash
git add src/visualization/loss.py
git commit -m "feat: single-axis AMI+ARI panel, new plot_inference_analysis for Algorithm 2"
```

---

### Task 6: Timestamped experiment folders + AMI/ARI schema — `src/visualization/report.py`

**Files:**
- Modify: `src/visualization/report.py` (`generate_training_figures`'s config-load path and `SimpleNamespace` construction)

**Interfaces:**
- Consumes: `training_results.pth` keys `ami_scores`/`val_ami_scores`/`ari_scores`/`val_ari_scores` (Task 3's schema); a `config.yaml` at the run folder root, one level above `experiment_dir` (Task 8's convention).
- Produces: no signature change to `generate_training_figures(experiment_dir, figures_dir, ...)`.

- [ ] **Step 1: Load `config.yaml` from the run root, not `best/`**

Replace:
```python
    best_dir = experiment_dir / "best"
    config = OmegaConf.load(best_dir / "config.yaml")
    decoder_factory = _build_decoder_factory(config.model)
```
with:
```python
    best_dir = experiment_dir / "best"
    # The config.yaml copy for a run lives at the run folder's root (one
    # level above experiment_dir, which is that run's models/ subfolder),
    # not inside best/ -- written once by the calling notebook when the run
    # folder is created.
    run_dir = experiment_dir.parent
    config = OmegaConf.load(run_dir / "config.yaml")
    decoder_factory = _build_decoder_factory(config.model)
```

- [ ] **Step 2: Update the schema guard and `SimpleNamespace` to AMI+ARI**

Replace:
```python
    # Guard against pre-NMI-swap checkpoints (which have ari_scores/silhouette_scores)
    if 'nmi_scores' not in history:
        raise ValueError(
            f"{best_dir / 'training_results.pth'} predates the NMI metric swap "
            "(has ari_scores/silhouette_scores instead). Re-run training to regenerate it."
        )

    trainer_view = SimpleNamespace(
        recon_train_losses=history['recon_train_losses'],
        recon_val_losses=history['recon_val_losses'],
        gmm_train_losses=history['gmm_train_losses'],
        gmm_val_losses=history['gmm_val_losses'],
        nmi_scores=history['nmi_scores'],
        val_nmi_scores=history['val_nmi_scores'],
        learning_rates=history['learning_rates'],
        momentum_betas=history['momentum_betas'],
        epoch_times=history['epoch_times'],
        training_config=config.training,
    )
```
with:
```python
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
```

- [ ] **Step 3: Verify — full pipeline with a run-root config.yaml (simulating Task 8's notebook convention)**

```bash
grep -n "nmi" src/visualization/report.py
```
Expected: no output (empty).

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from pathlib import Path
from omegaconf import OmegaConf
import torch
from src.data import create_dataloaders, get_sample_batches, collect_all_labels
from src.training import DGDTrainer
from src.visualization import generate_training_figures

config = OmegaConf.load('config/config.yaml')
config.data.total_subset_fraction = 0.02
config.data.download = False
config.experiment_name = 'plan_smoke_report_ami_ari'
run_dir = Path('/tmp/plan_smoke_run_20260101_000000_report_ami_ari')
run_dir.mkdir(parents=True, exist_ok=True)
config.paths.models_dir = str(run_dir / 'models')
config.paths.figures_dir = str(run_dir / 'figures')
config.training.epochs = 3
config.training.first_epoch_gmm = 1
config.training.refit_gmm_interval = 1
config.training.early_stopping_patience = 100
config.training.latent_noise_scale = 0.1

# Simulate what Task 8's notebook cell will do: write config.yaml at the
# run root before training starts.
OmegaConf.save(config, str(run_dir / 'config.yaml'))

device = torch.device('cpu')
train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
sample_data = get_sample_batches(train_loader, val_loader, device=device, n_per_class=2, n_classes=len(class_names))
train_labels = collect_all_labels(train_loader)
val_labels = collect_all_labels(val_loader)

trainer = DGDTrainer(config=config, device=device, verbose=False)
trainer.train(train_loader, val_loader, sample_data, class_names)

generate_training_figures(
    Path(config.paths.models_dir), Path(config.paths.figures_dir),
    class_names, train_labels, val_labels, sample_data, device=device,
)

loss_curves = Path(config.paths.figures_dir) / 'loss_curves.png'
assert loss_curves.exists() and loss_curves.stat().st_size > 0
print('OK: generate_training_figures reads config.yaml from run root and AMI/ARI schema, end-to-end')
"
```
Expected: `OK: ...` with no traceback.

- [ ] **Step 4: Commit**

```bash
git add src/visualization/report.py
git commit -m "refactor: report.py reads run-root config.yaml and AMI+ARI schema"
```

---

### Task 7: Inference figure parity — `src/visualization/report.py`

**Files:**
- Modify: `src/visualization/report.py` (extract `_plot_gmm_component_samples`; add `generate_inference_figures`; add `ClusteringMetrics` import; add `plot_inference_analysis` import)
- Modify: `src/visualization/__init__.py` (export `generate_inference_figures`)

**Interfaces:**
- Consumes: `plot_inference_analysis` (Task 5), `plot_latent_space`/`plot_images_by_class` (existing), `tgmm.ClusteringMetrics.adjusted_mutual_info_score`/`.adjusted_rand_score` (Task 3's already-verified signatures).
- Produces: `generate_inference_figures(figures_dir, decoder, gmm, test_rep, test_labels, class_names, sample_data, step_history, device) -> Tuple[float, float]` (returns `(test_ami, test_ari)`). Task 9 (inference notebook) consumes this by exact name, parameter order, and return type.

- [ ] **Step 1: Add the `ClusteringMetrics` and `plot_inference_analysis` imports**

Replace (top of `report.py`):
```python
from ..models import ConvDecoder
from ..utils.checkpoint import load_checkpoint
from .latent import plot_latent_space
from .image import plot_images_by_class, plot_generated_samples
from .loss import plot_training_analysis, plot_training_dynamics
```
with:
```python
from tgmm import ClusteringMetrics

from ..models import ConvDecoder
from ..utils.checkpoint import load_checkpoint
from .latent import plot_latent_space
from .image import plot_images_by_class, plot_generated_samples
from .loss import plot_training_analysis, plot_training_dynamics, plot_inference_analysis
```

- [ ] **Step 2: Extract the GMM-component sample grid into a shared helper**

Replace (the standalone comment + block at the bottom of `generate_training_figures`):
```python
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
```
with:
```python
    # GMM-component sample grid, best model only
    _plot_gmm_component_samples(best_checkpoint['gmm'], best_checkpoint['decoder'], figures_dir, device)
```

Add the extracted helper right before `generate_training_figures`'s `def` line:
```python
def _plot_gmm_component_samples(gmm, decoder, figures_dir: Path, device: torch.device) -> None:
    """Write one GMM-component sample grid PNG per component, sorted by weight descending."""
    if gmm is None:
        return
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


```

- [ ] **Step 3: Add `generate_inference_figures`**

Add at the end of `report.py`:
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
        Directory to write PNGs into (created if missing)
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

    Returns
    -------
    Tuple[float, float]
        (test_ami, test_ari), computed from the clean test_rep.z against the
        frozen gmm's predictions -- the same values used in the latent-space
        plot title.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        predicted_labels = gmm.predict(test_rep.z.detach())

    cluster_metrics = ClusteringMetrics()
    test_ami = cluster_metrics.adjusted_mutual_info_score(test_labels, predicted_labels)
    test_ari = cluster_metrics.adjusted_rand_score(test_labels, predicted_labels)

    plot_latent_space(
        representations=test_rep.z.detach(), labels=test_labels, gmm=gmm, class_names=class_names,
        title=f"Test Latent Space (Algorithm 2 inference) - AMI: {test_ami:.4f}, ARI: {test_ari:.4f}",
        save_path=str(figures_dir / "latent_test.png"), show=False,
    )

    indices_test, images_test, labels_test = sample_data
    with torch.no_grad():
        recon_test = decoder(test_rep(indices_test.to(device)))

    plot_images_by_class(
        images=recon_test, labels=labels_test, class_names=class_names,
        title="Test: Reconstructed Images by Class (Algorithm 2 inference)", n_per_class=5, cmap='viridis',
        save_path=str(figures_dir / "recon_test.png"), show=False,
    )

    plot_inference_analysis(
        step_history['loss'], step_history['recon'], step_history['gmm'], step_history['noise'],
        save_path=str(figures_dir / "loss_curve.png"), show=False,
    )

    _plot_gmm_component_samples(gmm, decoder, figures_dir, device)

    return test_ami, test_ari
```

- [ ] **Step 4: Export `generate_inference_figures`**

Replace (`src/visualization/__init__.py`):
```python
from .report import generate_training_figures
```
with:
```python
from .report import generate_training_figures, generate_inference_figures
```
and replace:
```python
    # Post-hoc report generation
    'generate_training_figures',
]
```
with:
```python
    # Post-hoc report generation
    'generate_training_figures',
    'generate_inference_figures',
]
```

- [ ] **Step 5: Verify — `generate_inference_figures` writes every expected file and returns correct metric values**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from pathlib import Path
from omegaconf import OmegaConf
import torch
from src.data import create_dataloaders, collect_all_labels, collect_class_samples
from src.models import RepresentationLayer, ConvDecoder
from src.training import DGDTrainer
from src.utils.checkpoint import load_checkpoint
from src.visualization import generate_inference_figures

config = OmegaConf.load('config/config.yaml')
config.data.total_subset_fraction = 0.02
config.data.download = False
config.experiment_name = 'plan_smoke_inference_figs'
run_dir = Path('/tmp/plan_smoke_run_20260101_000000_inference_figs')
run_dir.mkdir(parents=True, exist_ok=True)
config.paths.models_dir = str(run_dir / 'models')
config.training.epochs = 2
config.training.first_epoch_gmm = 1
config.training.refit_gmm_interval = 1
config.training.early_stopping_patience = 100
config.training.latent_noise_scale = 0.0

device = torch.device('cpu')
train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
trainer = DGDTrainer(config=config, device=device, verbose=False)
trainer.train(train_loader, val_loader, sample_data=None, class_names=class_names)

best_dir = Path(config.paths.models_dir) / 'best'
def decoder_factory():
    return ConvDecoder(
        latent_dim=config.model.representation.n_features,
        hidden_dims=config.model.decoder.hidden_dims,
        output_channels=config.model.decoder.output_channels,
        output_size=config.model.decoder.output_size,
        activation=config.model.decoder.activation,
        final_activation=config.model.decoder.final_activation,
        dropout_rate=config.model.decoder.dropout_rate,
        init_size=config.model.decoder.init_size,
    )
checkpoint = load_checkpoint(best_dir, decoder_factory, device=device)
decoder, gmm = checkpoint['decoder'], checkpoint['gmm']
decoder.eval()

test_rep = RepresentationLayer(
    dim=config.model.representation.n_features, n_samples=len(test_loader.dataset),
    dist='normal', dist_params={}, device=device,
)
test_labels = collect_all_labels(test_loader)
sample_data = collect_class_samples(test_loader, n_per_class=2, n_classes=len(class_names))

step_history = {'loss': [10.0, 8.0], 'recon': [7.0, 6.0], 'gmm': [3.0, 2.0], 'noise': [1.0, 0.5]}
figures_dir = run_dir / 'inference' / '20260101_000001' / 'figures'

test_ami, test_ari = generate_inference_figures(
    figures_dir=figures_dir, decoder=decoder, gmm=gmm, test_rep=test_rep,
    test_labels=test_labels, class_names=class_names, sample_data=sample_data,
    step_history=step_history, device=device,
)
assert isinstance(test_ami, float) and isinstance(test_ari, float)
assert -1.0 <= test_ari <= 1.0
assert (figures_dir / 'latent_test.png').exists()
assert (figures_dir / 'recon_test.png').exists()
assert (figures_dir / 'loss_curve.png').exists()
gmm_component_pngs = list(figures_dir.glob('gmm_component*_samples.png'))
assert len(gmm_component_pngs) == gmm.n_components, f'expected {gmm.n_components} GMM component PNGs, got {len(gmm_component_pngs)}'
print(f'OK: generate_inference_figures wrote all figures, returned AMI={test_ami:.4f}, ARI={test_ari:.4f}')
"
```
Expected: `OK: generate_inference_figures wrote all figures, returned AMI=..., ARI=...`.

- [ ] **Step 6: Commit**

```bash
git add src/visualization/report.py src/visualization/__init__.py
git commit -m "feat: generate_inference_figures for Algorithm-2 figure parity with training"
```

---

### Task 8: Timestamped experiment folders — `notebooks/dgd_training_demo.ipynb`

**Files:**
- Modify: `notebooks/dgd_training_demo.ipynb` (config-setup cell: timestamped run-folder resolution; figures cell: drop the double experiment_name join) — edited via a Python script operating on the notebook's JSON, not by hand.

**Interfaces:**
- Consumes: `config.paths.experiments_dir` (Task 2), `trainer.train()`'s pre-resolved-`models_dir` contract (Task 4), `generate_training_figures`'s run-root-`config.yaml` contract (Task 6).
- Produces: no change to what the notebook returns to a human running it, beyond the run directory now being timestamped and printed.

- [ ] **Step 1: Write and run the notebook-patching script**

Create `/tmp/claude-1000/-home-asp-Downloads-HeaDS-ImageDGD/4e7da3f5-6e40-417f-bdc5-1b55f6e03f0f/scratchpad/patch_training_notebook.py`:

```python
import json
from pathlib import Path

nb_path = Path("notebooks/dgd_training_demo.ipynb")
nb = json.loads(nb_path.read_text())

# --- Cell 2: config setup -- add timestamped run_dir resolution. ---
old_config_cell = "".join(nb["cells"][2]["source"])
assert 'config.paths.models_dir = str(project_root / "models")' in old_config_cell, "cell 2 source has changed unexpectedly"
new_config_cell = '''from datetime import datetime
from omegaconf import open_dict

with initialize(version_base=None, config_path="../config"):
    config = compose(config_name="config")

# Fix relative paths to be relative to the project root, not the notebook's cwd
config.data.root_dir = str(project_root / "data")
config.paths.experiments_dir = str(project_root / "experiments")

# Every run gets its own timestamped folder under experiments_dir, so
# re-running this notebook never overwrites a previous run's checkpoints or
# figures. config.yaml is copied here once, capturing the exact settings
# used for this run.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(config.paths.experiments_dir) / f"{timestamp}_{config.experiment_name}"
run_dir.mkdir(parents=True, exist_ok=True)

# hydra's compose() returns a struct-mode config: models_dir/figures_dir
# aren't in config.yaml's paths: block (only experiments_dir is, since they
# are resolved per-run rather than being static defaults), so assigning
# them requires open_dict() to temporarily allow new keys -- a plain
# assignment would raise ConfigAttributeError: Key 'models_dir' is not in
# struct.
with open_dict(config):
    config.paths.models_dir = str(run_dir / "models")
    config.paths.figures_dir = str(run_dir / "figures")

OmegaConf.save(config, str(run_dir / "config.yaml"))
print(f"Run directory: {run_dir}")

print("CONFIGURATION")
print(f"{'='*60}")
print(OmegaConf.to_yaml(config))
print(f"{'='*60}")
'''
nb["cells"][2]["source"] = new_config_cell.splitlines(keepends=True)
nb["cells"][2]["execution_count"] = None
nb["cells"][2]["outputs"] = []

# --- Cell 5: figures cell -- drop the double experiment_name join. ---
old_figures_cell = "".join(nb["cells"][5]["source"])
assert "experiment_dir = Path(config.paths.models_dir) / config.experiment_name" in old_figures_cell, "cell 5 source has changed unexpectedly"
new_figures_cell = '''experiment_dir = Path(config.paths.models_dir)
figures_dir = Path(config.paths.figures_dir)

train_labels = collect_all_labels(train_loader)
val_labels = collect_all_labels(val_loader)

generate_training_figures(
    experiment_dir=experiment_dir,
    figures_dir=figures_dir,
    class_names=class_names,
    train_labels=train_labels,
    val_labels=val_labels,
    sample_data=sample_data,
    device=device,
)

print(f"Figures written to {figures_dir}")
'''
nb["cells"][5]["source"] = new_figures_cell.splitlines(keepends=True)
nb["cells"][5]["execution_count"] = None
nb["cells"][5]["outputs"] = []

# Reset execution_count/outputs on every code cell, and reset kernelspec/
# language_info to generic values, to strip any local-run artifacts a
# previous execution left in the working tree (this notebook has been run
# for real against the old flat folder layout and old NMI metric names --
# none of that belongs in the committed diff).
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

nb_path.write_text(json.dumps(nb, indent=1))
print("Patched cells 2, 5")
```

Run it: `/home/asp/.venvs/rapids_cu13/bin/python /tmp/claude-1000/-home-asp-Downloads-HeaDS-ImageDGD/4e7da3f5-6e40-417f-bdc5-1b55f6e03f0f/scratchpad/patch_training_notebook.py` (run from the repo root: `cd /home/asp/Downloads/HeaDS/ImageDGD` first).

- [ ] **Step 2: Verify — notebook JSON is valid, double-join and NMI symbols are gone**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
import json
from pathlib import Path
nb = json.loads(Path('notebooks/dgd_training_demo.ipynb').read_text())
assert nb['nbformat'] == 4
full_text = json.dumps(nb)
assert 'config.paths.models_dir) / config.experiment_name' not in full_text
assert 'experiments_dir' in full_text
assert 'timestamp' in full_text
assert all(c.get('execution_count') is None and c.get('outputs', []) == [] for c in nb['cells'] if c['cell_type'] == 'code')
print('cells:', [c['cell_type'] for c in nb['cells']])
print('OK: notebook JSON valid, double-join gone, run_dir logic present, no stale outputs')
"
```
Expected: `cells: ['markdown', 'code', 'code', 'code', 'code', 'code']` then `OK: ...`.

- [ ] **Step 3: Verify — the run-folder logic, exercised standalone, produces two distinct non-overwriting runs**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
import time
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from hydra import initialize, compose
import torch
from src.data import create_dataloaders, get_sample_batches, collect_all_labels
from src.training import DGDTrainer
from src.visualization import generate_training_figures

def run_once(experiments_dir):
    # Loaded via hydra.compose(), matching the real notebook exactly (not
    # OmegaConf.load(), which -- unlike compose() -- does NOT produce a
    # struct-mode config and would silently fail to catch a missing
    # open_dict() in the cell this mirrors).
    with initialize(version_base=None, config_path='config'):
        config = compose(config_name='config')
    config.data.total_subset_fraction = 0.02
    config.data.download = False
    config.experiment_name = 'plan_smoke_two_runs'
    config.data.root_dir = 'data'
    config.paths.experiments_dir = str(experiments_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(config.paths.experiments_dir) / f'{timestamp}_{config.experiment_name}'
    run_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(config):
        config.paths.models_dir = str(run_dir / 'models')
        config.paths.figures_dir = str(run_dir / 'figures')
    OmegaConf.save(config, str(run_dir / 'config.yaml'))

    config.training.epochs = 2
    config.training.first_epoch_gmm = 1
    config.training.refit_gmm_interval = 1
    config.training.early_stopping_patience = 100
    config.training.latent_noise_scale = 0.0

    device = torch.device('cpu')
    train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
    sample_data = get_sample_batches(train_loader, val_loader, device=device, n_per_class=2, n_classes=len(class_names))
    train_labels = collect_all_labels(train_loader)
    val_labels = collect_all_labels(val_loader)

    trainer = DGDTrainer(config=config, device=device, verbose=False)
    trainer.train(train_loader, val_loader, sample_data, class_names)
    generate_training_figures(
        Path(config.paths.models_dir), Path(config.paths.figures_dir),
        class_names, train_labels, val_labels, sample_data, device=device,
    )
    return run_dir

experiments_dir = Path('/tmp/plan_smoke_experiments_two_runs')
run_dir_1 = run_once(experiments_dir)
time.sleep(1.1)  # ensure the second timestamp differs at 1-second resolution
run_dir_2 = run_once(experiments_dir)

assert run_dir_1 != run_dir_2, 'two runs must land in different folders'
assert (run_dir_1 / 'config.yaml').exists() and (run_dir_2 / 'config.yaml').exists()
assert (run_dir_1 / 'models' / 'best' / 'decoder.pth').exists()
assert (run_dir_2 / 'models' / 'best' / 'decoder.pth').exists()
assert (run_dir_1 / 'figures' / 'loss_curves.png').exists()
assert (run_dir_2 / 'figures' / 'loss_curves.png').exists()
# The first run's artifacts must still be present after the second run.
assert (run_dir_1 / 'models' / 'best' / 'decoder.pth').exists(), 'second run must not have clobbered the first'
print(f'OK: two runs produced distinct folders {run_dir_1.name} and {run_dir_2.name}, neither clobbered the other')
"
```
Expected: `OK: two runs produced distinct folders ..., neither clobbered the other`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/dgd_training_demo.ipynb
git commit -m "feat: training notebook writes to a timestamped experiment folder"
```

---

### Task 9: Timestamped folders, inference-specific noise, and figure parity — `notebooks/dgd_test_inference.ipynb`

**Files:**
- Modify: `notebooks/dgd_test_inference.ipynb` (imports cell; config-setup cell: run discovery instead of fixed path; checkpoint-loading cell: drop double join; M-step loop cell: inference-scoped noise keys + step-history collection; save-representations cell: becomes inference-run-folder creation; final figures cell: AMI+ARI + `generate_inference_figures`) — edited via a Python script operating on the notebook's JSON.

**Interfaces:**
- Consumes: `config.paths.experiments_dir` (Task 2), `config.training.inference.latent_noise_scale`/`.latent_noise_start`/`.latent_noise_end` (Task 2), `generate_inference_figures(figures_dir, decoder, gmm, test_rep, test_labels, class_names, sample_data, step_history, device) -> (test_ami, test_ari)` (Task 7).
- Produces: `experiments/<training-run>/inference/<timestamp>/{config.yaml, test_representation.pt, figures/}`.

- [ ] **Step 1: Write and run the notebook-patching script**

Create `/tmp/claude-1000/-home-asp-Downloads-HeaDS-ImageDGD/4e7da3f5-6e40-417f-bdc5-1b55f6e03f0f/scratchpad/patch_inference_notebook.py`:

```python
import json
from pathlib import Path

nb_path = Path("notebooks/dgd_test_inference.ipynb")
nb = json.loads(nb_path.read_text())

# --- Cell 1: imports -- drop the now-unused tgmm/plot_latent_space/
# plot_images_by_class imports, add generate_inference_figures. ---
old_imports_cell = "".join(nb["cells"][1]["source"])
assert "from tgmm import ClusteringMetrics" in old_imports_cell, "cell 1 source has changed unexpectedly"
new_imports_cell = '''import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf, open_dict
from hydra import initialize, compose

current_dir = Path.cwd()
if 'notebooks' in current_dir.parts:
    project_root = current_dir.parent
else:
    project_root = current_dir

sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from src.data import create_dataloaders, collect_all_labels, collect_class_samples
from src.models import RepresentationLayer, ConvDecoder
from src.utils import setup_device, set_random_seed, setup_cuml_acceleration
from src.utils.checkpoint import load_checkpoint
from src.visualization import generate_inference_figures

device = setup_device(verbose=True)
set_random_seed(seed=42, device=device)
setup_cuml_acceleration(verbose=True)
'''
nb["cells"][1]["source"] = new_imports_cell.splitlines(keepends=True)
nb["cells"][1]["execution_count"] = None
nb["cells"][1]["outputs"] = []

# --- Cell 2: config setup -- discover the most recent training run instead
# of a fixed path. ---
old_config_cell = "".join(nb["cells"][2]["source"])
assert "trained_cfg = OmegaConf.load(Path(config.paths.models_dir) / config.experiment_name" in old_config_cell, "cell 2 source has changed unexpectedly"
new_config_cell = '''config.data.root_dir = str(project_root / "data")
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

# hydra's compose() returns a struct-mode config: models_dir isn't in
# config.yaml's paths: block (only experiments_dir is, since it's resolved
# per-run rather than being a static default), so assigning it requires
# open_dict() to temporarily allow a new key -- a plain assignment would
# raise ConfigAttributeError: Key 'models_dir' is not in struct.
with open_dict(config):
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

# Re-derive the same 3-way split independently (same random_seed as training),
# so `test_loader` here is exactly the held-out split carved out during training
# and never optimized against.
train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
print(f"Test loader: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
'''
nb["cells"][2]["source"] = new_config_cell.splitlines(keepends=True)
nb["cells"][2]["execution_count"] = None
nb["cells"][2]["outputs"] = []

# --- Cell 3: checkpoint loading -- drop the double experiment_name join. ---
old_checkpoint_cell = "".join(nb["cells"][3]["source"])
assert 'best_dir = Path(config.paths.models_dir) / config.experiment_name / "best"' in old_checkpoint_cell, "cell 3 source has changed unexpectedly"
new_checkpoint_cell = old_checkpoint_cell.replace(
    'best_dir = Path(config.paths.models_dir) / config.experiment_name / "best"',
    'best_dir = Path(config.paths.models_dir) / "best"',
)
nb["cells"][3]["source"] = new_checkpoint_cell.splitlines(keepends=True)
nb["cells"][3]["execution_count"] = None
nb["cells"][3]["outputs"] = []

# --- Cell 5: M-step loop -- inference-scoped noise keys, collect step history. ---
old_loop_cell = "".join(nb["cells"][5]["source"])
assert "latent_noise_scale = config.training.get('latent_noise_scale', 0.0)" in old_loop_cell, "cell 5 source has changed unexpectedly"
new_loop_cell = '''# Algorithm 2, lines 2-5: optimize test_rep alone against the frozen decoder+GMM.
# M0 (prior_warmup_steps) < M (epochs): reconstruction-only warm-up before the
# GMM prior term is added, so a fresh z isn't dominated by the prior gradient
# before it has any reconstruction signal to work with.
import math

lr_config = config.training.lr_scheduler
rep_config = config.training.optimizer.representation

test_optimizer = torch.optim.AdamW(
    test_rep.parameters(),
    lr=rep_config.lr,
    betas=tuple(rep_config.betas),
    eps=rep_config.eps,
    weight_decay=rep_config.weight_decay,
    amsgrad=rep_config.get('amsgrad', False),
)

M = config.training.inference.epochs
M0 = config.training.inference.prior_warmup_steps
assert M0 < M, "training.inference.prior_warmup_steps must be < training.inference.epochs"

if lr_config.get('enabled', False):
    max_lr = lr_config.get('max_lr_representation', None) or rep_config.lr
    test_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        test_optimizer,
        max_lr=max_lr,
        total_steps=M,
        pct_start=lr_config.get('pct_start', 0.3),
        anneal_strategy=lr_config.get('anneal_strategy', 'cos'),
        div_factor=lr_config.get('div_factor', 25.0),
        final_div_factor=lr_config.get('final_div_factor', 10000.0),
        cycle_momentum=lr_config.get('cycle_momentum', True),
        base_momentum=lr_config.get('base_momentum', 0.85),
        max_momentum=lr_config.get('max_momentum', 0.95),
        three_phase=lr_config.get('three_phase', False),
    )
else:
    test_scheduler = None

lambda_gmm = config.training.lambda_gmm
n_test = len(test_loader.dataset)

# Inference has its own noise schedule, independent of training's
# latent_noise_scale/_start/_end -- same cosine-anneal formula, mapped onto
# step index m instead of epoch.
latent_noise_scale = config.training.inference.get('latent_noise_scale', 0.0)
noise_start = config.training.inference.get('latent_noise_start', 1.0)
noise_end = config.training.inference.get('latent_noise_end', 0.01)

print(f"Optimizing {n_test} test representations for {M} steps (prior warm-up: {M0} steps)...")

step_history = {'loss': [], 'recon': [], 'gmm': [], 'noise': []}

for m in range(1, M + 1):
    test_optimizer.zero_grad()

    if latent_noise_scale > 0:
        progress = (m - 1) / max(M - 1, 1)
        noise_scale_m = noise_end + (noise_start - noise_end) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        noise_scale_m = 0.0

    total_loss = 0.0
    total_recon = 0.0
    total_gmm = 0.0

    for index, x, _ in test_loader:
        x, index = x.to(device), index.to(device)

        z = test_rep(index)

        if noise_scale_m > 0:
            z = z + torch.randn_like(z) * noise_scale_m

        y = decoder(z)
        recon_loss = F.mse_loss(y, x, reduction='sum')

        if m >= M0:
            gmm_error = -lambda_gmm * torch.sum(gmm.score_samples(z))
            loss = recon_loss + gmm_error
        else:
            gmm_error = torch.tensor(0.0, device=device)
            loss = recon_loss

        loss.backward()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_gmm += gmm_error.item()

    test_optimizer.step()
    if test_scheduler is not None:
        test_scheduler.step()

    step_history['loss'].append(total_loss / n_test)
    step_history['recon'].append(total_recon / n_test)
    step_history['gmm'].append(total_gmm / n_test)
    step_history['noise'].append(noise_scale_m)

    if m % max(1, M // 10) == 0 or m == M:
        print(f"Step {m}/{M}: loss={total_loss/n_test:.4f}, recon={total_recon/n_test:.4f}, gmm={total_gmm/n_test:.4f}, noise={noise_scale_m:.4f}")

print("Test representation optimization complete.")
'''
nb["cells"][5]["source"] = new_loop_cell.splitlines(keepends=True)
nb["cells"][5]["execution_count"] = None
nb["cells"][5]["outputs"] = []

# --- Cell 6: save representations -- becomes inference-run-folder creation. ---
old_save_cell = "".join(nb["cells"][6]["source"])
assert 'test_inference_dir = Path(config.paths.models_dir) / config.experiment_name / "test_inference"' in old_save_cell, "cell 6 source has changed unexpectedly"
new_save_cell = '''from datetime import datetime

inference_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
inference_dir = run_dir / "inference" / inference_timestamp
inference_dir.mkdir(parents=True, exist_ok=True)

OmegaConf.save(config, str(inference_dir / "config.yaml"))

test_rep.save(str(inference_dir / "test_representation.pt"))
print(f"Saved optimized test representations to {inference_dir / 'test_representation.pt'}")
'''
nb["cells"][6]["source"] = new_save_cell.splitlines(keepends=True)
nb["cells"][6]["execution_count"] = None
nb["cells"][6]["outputs"] = []

# --- Cell 7: final figures -- AMI+ARI, generate_inference_figures. ---
old_figures_cell = "".join(nb["cells"][7]["source"])
assert "test_nmi = cluster_metrics.normalized_mutual_info_score(test_labels, predicted_labels)" in old_figures_cell, "cell 7 source has changed unexpectedly"
new_figures_cell = '''figures_dir = inference_dir / "figures"

test_labels = collect_all_labels(test_loader)

sample_data = collect_class_samples(test_loader, n_per_class=5, n_classes=len(class_names))

step_history_totals = step_history

test_ami, test_ari = generate_inference_figures(
    figures_dir=figures_dir,
    decoder=decoder,
    gmm=gmm,
    test_rep=test_rep,
    test_labels=test_labels,
    class_names=class_names,
    sample_data=sample_data,
    step_history=step_history_totals,
    device=device,
)

print(f"Test AMI: {test_ami:.4f}, Test ARI: {test_ari:.4f}")
print(f"Figures written to {figures_dir}")
'''
nb["cells"][7]["source"] = new_figures_cell.splitlines(keepends=True)
nb["cells"][7]["execution_count"] = None
nb["cells"][7]["outputs"] = []

# Reset execution_count/outputs on every code cell, and reset kernelspec/
# language_info to generic values, to strip any local-run artifacts a
# previous execution left in the working tree.
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

nb_path.write_text(json.dumps(nb, indent=1))
print("Patched cells 1, 2, 3, 5, 6, 7")
```

Run it: `/home/asp/.venvs/rapids_cu13/bin/python /tmp/claude-1000/-home-asp-Downloads-HeaDS-ImageDGD/4e7da3f5-6e40-417f-bdc5-1b55f6e03f0f/scratchpad/patch_inference_notebook.py` (run from the repo root).

Note: the generated `step_history_totals = step_history` line in cell 7 is a harmless alias kept for readability; it is not required — an implementer who prefers may simplify it to passing `step_history` directly. Either is acceptable; the verification step below does not distinguish between them.

- [ ] **Step 2: Verify — notebook JSON is valid, old symbols gone, new ones present**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
import json
from pathlib import Path
nb = json.loads(Path('notebooks/dgd_test_inference.ipynb').read_text())
assert nb['nbformat'] == 4
full_text = json.dumps(nb)
assert 'normalized_mutual_info_score' not in full_text
assert 'config.paths.models_dir) / config.experiment_name' not in full_text
assert 'ClusteringMetrics' not in full_text
assert 'adjusted_mutual_info_score' not in full_text  # lives in report.py now, not the notebook
assert 'generate_inference_figures' in full_text
assert 'experiments_dir' in full_text
assert 'config.training.inference.get' in full_text
assert 'step_history' in full_text
assert all(c.get('execution_count') is None and c.get('outputs', []) == [] for c in nb['cells'] if c['cell_type'] == 'code')
print('cells:', [c['cell_type'] for c in nb['cells']])
print('OK: notebook JSON valid, old symbols gone, run-discovery + inference-figures present')
"
```
Expected: `cells: ['markdown', 'code', 'code', 'code', 'code', 'code', 'code', 'code']` then `OK: ...`.

- [ ] **Step 3: Verify — the full inference pipeline, exercised standalone against a real trained run**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from hydra import initialize, compose
import torch
import torch.nn.functional as F
import math
from src.data import create_dataloaders, collect_all_labels, collect_class_samples
from src.training import DGDTrainer
from src.models import RepresentationLayer, ConvDecoder
from src.utils.checkpoint import load_checkpoint
from src.visualization import generate_inference_figures

# 1. Train a tiny model into a timestamped run folder (mirrors Task 8's
# notebook). OmegaConf.load() here is fine -- Task 8's own struct-mode
# handling is validated separately, in Task 8's own verification; this
# phase just needs a real trained run for phase 2 to discover.
train_config = OmegaConf.load('config/config.yaml')
train_config.data.total_subset_fraction = 0.02
train_config.data.download = False
train_config.experiment_name = 'plan_smoke_full_inference'
experiments_dir = Path('/tmp/plan_smoke_experiments_full_inference')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = experiments_dir / f'{timestamp}_{train_config.experiment_name}'
run_dir.mkdir(parents=True, exist_ok=True)
train_config.paths.models_dir = str(run_dir / 'models')
train_config.paths.figures_dir = str(run_dir / 'figures')
OmegaConf.save(train_config, str(run_dir / 'config.yaml'))

train_config.training.epochs = 2
train_config.training.first_epoch_gmm = 1
train_config.training.refit_gmm_interval = 1
train_config.training.early_stopping_patience = 100
train_config.training.latent_noise_scale = 0.1

device = torch.device('cpu')
train_loader, val_loader, test_loader, class_names = create_dataloaders(train_config)
trainer = DGDTrainer(config=train_config, device=device, verbose=False)
trainer.train(train_loader, val_loader, sample_data=None, class_names=class_names)

# 2. Mirror the inference notebook's OWN cell 2 exactly: a fresh
# hydra.compose() (a separate process/notebook from training, so a fresh
# config object -- not train_config above), run-discovery, and the
# open_dict()-guarded models_dir assignment. hydra.compose() -- unlike
# OmegaConf.load() -- produces a struct-mode config, so this is the only
# way to actually exercise the open_dict() requirement the real notebook
# cell depends on.
with initialize(version_base=None, config_path='config'):
    config = compose(config_name='config')
config.data.total_subset_fraction = 0.02
config.data.download = False
config.experiment_name = 'plan_smoke_full_inference'
config.paths.experiments_dir = str(experiments_dir)
config.training.inference.epochs = 5
config.training.inference.prior_warmup_steps = 0
config.training.inference.latent_noise_scale = 0.2
config.training.inference.latent_noise_start = 1.0
config.training.inference.latent_noise_end = 0.01

candidates = sorted(
    p for p in experiments_dir.glob(f'*_{config.experiment_name}')
    if (p / 'models' / 'best').is_dir()
)
assert candidates
discovered_run_dir = candidates[-1]
assert discovered_run_dir == run_dir

with open_dict(config):
    config.paths.models_dir = str(discovered_run_dir / 'models')

trained_cfg = OmegaConf.load(discovered_run_dir / 'config.yaml')
assert config.random_seed == trained_cfg.random_seed
for key in ['total_subset_fraction', 'val_split', 'test_split']:
    assert config.data[key] == trained_cfg.data[key]

# 3. Mirror cell 3's checkpoint loading (single join, no experiment_name).
best_dir = Path(config.paths.models_dir) / 'best'
def decoder_factory():
    return ConvDecoder(
        latent_dim=config.model.representation.n_features,
        hidden_dims=config.model.decoder.hidden_dims,
        output_channels=config.model.decoder.output_channels,
        output_size=config.model.decoder.output_size,
        activation=config.model.decoder.activation,
        final_activation=config.model.decoder.final_activation,
        dropout_rate=config.model.decoder.dropout_rate,
        init_size=config.model.decoder.init_size,
    )
checkpoint = load_checkpoint(best_dir, decoder_factory, device=device)
decoder, gmm = checkpoint['decoder'], checkpoint['gmm']
decoder.eval()
for p in decoder.parameters():
    p.requires_grad_(False)

# 4. Mirror cell 5's M-step loop with inference-scoped noise config.
test_rep = RepresentationLayer(
    dim=config.model.representation.n_features, n_samples=len(test_loader.dataset),
    dist='normal', dist_params={}, device=device,
)
test_optimizer = torch.optim.AdamW(test_rep.parameters(), lr=config.training.optimizer.representation.lr)
M = config.training.inference.epochs
M0 = config.training.inference.prior_warmup_steps
lambda_gmm = config.training.lambda_gmm
n_test = len(test_loader.dataset)
latent_noise_scale = config.training.inference.get('latent_noise_scale', 0.0)
noise_start = config.training.inference.get('latent_noise_start', 1.0)
noise_end = config.training.inference.get('latent_noise_end', 0.01)

step_history = {'loss': [], 'recon': [], 'gmm': [], 'noise': []}
for m in range(1, M + 1):
    test_optimizer.zero_grad()
    if latent_noise_scale > 0:
        progress = (m - 1) / max(M - 1, 1)
        noise_scale_m = noise_end + (noise_start - noise_end) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        noise_scale_m = 0.0
    total_loss = total_recon = total_gmm = 0.0
    for index, x, _ in test_loader:
        x, index = x.to(device), index.to(device)
        z = test_rep(index)
        if noise_scale_m > 0:
            z = z + torch.randn_like(z) * noise_scale_m
        y = decoder(z)
        recon_loss = F.mse_loss(y, x, reduction='sum')
        if m >= M0:
            gmm_error = -lambda_gmm * torch.sum(gmm.score_samples(z))
            loss = recon_loss + gmm_error
        else:
            gmm_error = torch.tensor(0.0, device=device)
            loss = recon_loss
        loss.backward()
        total_loss += loss.item(); total_recon += recon_loss.item(); total_gmm += gmm_error.item()
    test_optimizer.step()
    step_history['loss'].append(total_loss / n_test)
    step_history['recon'].append(total_recon / n_test)
    step_history['gmm'].append(total_gmm / n_test)
    step_history['noise'].append(noise_scale_m)

assert abs(step_history['noise'][0] - 1.0) < 1e-9
assert abs(step_history['noise'][-1] - 0.01) < 1e-9

# 5. Mirror cell 6's inference-run-folder creation.
inference_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
inference_dir = discovered_run_dir / 'inference' / inference_timestamp
inference_dir.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config, str(inference_dir / 'config.yaml'))
test_rep.save(str(inference_dir / 'test_representation.pt'))
assert (inference_dir / 'config.yaml').exists()
assert (inference_dir / 'test_representation.pt').exists()

# 6. Mirror cell 7's figure generation.
test_labels = collect_all_labels(test_loader)
sample_data = collect_class_samples(test_loader, n_per_class=2, n_classes=len(class_names))
figures_dir = inference_dir / 'figures'
test_ami, test_ari = generate_inference_figures(
    figures_dir=figures_dir, decoder=decoder, gmm=gmm, test_rep=test_rep,
    test_labels=test_labels, class_names=class_names, sample_data=sample_data,
    step_history=step_history, device=device,
)
assert -1.0 <= test_ari <= 1.0
assert (figures_dir / 'latent_test.png').exists()
assert (figures_dir / 'recon_test.png').exists()
assert (figures_dir / 'loss_curve.png').exists()

print(f'noise schedule: first={step_history[\"noise\"][0]:.4f}, last={step_history[\"noise\"][-1]:.4f}')
print(f'test AMI: {test_ami:.4f}, test ARI: {test_ari:.4f}')
print('OK: full inference pipeline works end-to-end with inference-scoped noise, nested run folder, figure parity')
"
```
Expected: `noise schedule: first=1.0000, last=0.0100`, a `test AMI: ..., test ARI: ...` line, then `OK: ...`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/dgd_test_inference.ipynb
git commit -m "feat: inference notebook gets nested timestamped run folders, own noise config, figure parity"
```

---

## Final Integration Check

After all nine tasks are committed, confirm the whole feature holds together as one working pipeline:

```bash
grep -rln 'nmi\|normalized_mutual_info_score\|models_dir) / config.experiment_name\|models_dir) / self.config.experiment_name' src/ notebooks/dgd_training_demo.ipynb notebooks/dgd_test_inference.ipynb config/config.yaml
```
Note: scoped to the two notebooks this plan touches, not `notebooks/*.ipynb` — the other notebooks in this repo carry embedded base64 image data that randomly contains the substring "nmi" and would otherwise false-positive. Also note the parens are intentionally unescaped: this repo's `grep` is `ugrep`, which treats bare `(`/`)` as literal but `\(`/`\)` as regex grouping (the opposite of what an escaped paren usually means) — an escaped `\)` with no matching `\(` errors out rather than matching literally.
Expected: no output (empty) — confirms no leftover NMI references or double-experiment-name joins anywhere this plan touched.

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from src.visualization import generate_training_figures, generate_inference_figures
from src.visualization.loss import plot_training_analysis, plot_inference_analysis
from src.training import DGDTrainer
print('OK: all new/renamed symbols import cleanly')
"
```
Expected: `OK: all new/renamed symbols import cleanly`.

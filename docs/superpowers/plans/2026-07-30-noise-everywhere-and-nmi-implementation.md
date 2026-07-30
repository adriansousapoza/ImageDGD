# Noise Injection Everywhere + NMI Metric Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend latent-space noise injection from training-only to validation (in `DGDTrainer.train`) and to the held-out-inference notebook's M-step optimization loop, and replace ARI + silhouette everywhere with a single Normalized Mutual Information (NMI) metric per split.

**Architecture:** No new files, no new config keys, no new abstractions. Five small, sequential edits to existing files: the noise gate and validation batch loop in `trainer.py`, the ARI/silhouette tracking-and-computation blocks in the same file, the clustering-metrics panel in `loss.py`, the schema-reading `SimpleNamespace` in `report.py`, and the M-step loop + final metrics cell in `dgd_test_inference.ipynb`.

**Tech Stack:** PyTorch, `tgmm.ClusteringMetrics.normalized_mutual_info_score`, OmegaConf, Jupyter notebook JSON (edited via a small Python script, not by hand).

**Spec:** `docs/superpowers/specs/2026-07-28-noise-everywhere-and-nmi-design.md` — read it if any task step below seems to conflict with this plan; this plan is a direct implementation of it.

## Global Constraints

- Use `/home/asp/.venvs/rapids_cu13/bin/python` for every verification command in this plan. The system `python3` has no torch/tgmm installed. A second venv, `torch_cu13`, has torch+tgmm+CUDA but lacks `cudf`/`cuml` — `src/utils/device.py` does `import cudf` at module level, so anything importing `src.utils` (nearly everything) fails under `torch_cu13`. `rapids_cu13` has the full stack (torch, tgmm, cudf, cuml, CUDA) and is the only one that imports the `src` package cleanly.
- No pytest suite in this project. Verification is standalone runnable Python scripts against a tiny real-data config, matching the pattern used throughout the train/val/test split refactor (`config.data.total_subset_fraction` set small, a scratch `experiment_name`, `paths.models_dir` pointed at `/tmp`).
- `notebooks/dgd_generative_model.ipynb` (the untracked math/design notebook the spec's Goal 5 says to delete) is **already absent from the working tree** — confirmed via `ls notebooks/dgd_generative_model.ipynb` (no such file) and `git status` (no entry, tracked or untracked). No task in this plan deletes it; there is nothing left to delete.
- Validation's `best_epoch`/`best_val_loss` selection logic in `trainer.py` is **not modified** by this plan — it already just reads whatever `val_loss` the val phase produces; that loss becomes noise-influenced as a side effect of Task 1, with no code change to the selection logic itself.
- NMI, like the ARI it replaces, is computed from the clean, stored `rep.z`/`val_rep.z`/`test_rep.z` — never the transient noised copy used only inside the loss computation. No task changes what representations a metric is computed against.
- GMM refit cadence, checkpoint cadence, and every other mechanism carried over from the train/val/test split refactor are unchanged.
- Every file this plan touches was last modified by the train/val/test split refactor (commits `24301be`..`ce11e41`) plus one uncommitted direct fix to `src/visualization/loss.py` (the `_align` helper, applied directly in response to a user-reported crash, never committed). This plan's edits are written against the **current working-tree content** of each file (including that uncommitted fix), not the last commit — read the file before editing if picking this plan up cold.

---

### Task 1: Extend noise injection to the validation phase — `src/training/trainer.py`

**Files:**
- Modify: `src/training/trainer.py:509-512` (training batch loop noise gate), `src/training/trainer.py:553-559` (validation batch loop)

**Interfaces:**
- Consumes: nothing new — `noise_scale` is already computed once per epoch at `trainer.py:492-499`, unchanged by this task.
- Produces: no new interface. Validation's `z` now receives the same noise the training batch loop does, using the same per-epoch `noise_scale` value. `best_val_loss`/`best_epoch` selection (`trainer.py:610-623`) is untouched but now reads a noise-influenced value.

- [ ] **Step 1: Remove the `model.decoder.training` proxy from the noise gate**

`model.decoder.training` was only ever a stand-in for "is this the train phase, not val" — the decoder is switched to `.eval()` before the val phase runs, so this condition happened to be `False` during validation. Since validation should now receive noise too, drop it.

Replace (`trainer.py:509-512`):
```python
                # Latent Space Noise Injection (regularization during training)
                if model.decoder.training and noise_scale > 0:
                    noise = torch.randn_like(z) * noise_scale
                    z = z + noise
```
with:
```python
                # Latent Space Noise Injection (regularization, applied in
                # both train and val phases so noise robustness generalizes)
                if noise_scale > 0:
                    noise = torch.randn_like(z) * noise_scale
                    z = z + noise
```

- [ ] **Step 2: Inject the same noise in the validation batch loop**

Replace (`trainer.py:553-559`):
```python
            for i, (index, x, _) in enumerate(val_loader):
                x, index = x.to(self.device), index.to(self.device)

                # Forward pass
                z = val_rep(index)
                y = model.decoder(z)
                recon_loss = F.mse_loss(y, x, reduction='sum')
```
with:
```python
            for i, (index, x, _) in enumerate(val_loader):
                x, index = x.to(self.device), index.to(self.device)

                # Forward pass
                z = val_rep(index)

                # Latent Space Noise Injection (same per-epoch schedule as
                # the train phase above — no separate/independent schedule)
                if noise_scale > 0:
                    noise = torch.randn_like(z) * noise_scale
                    z = z + noise

                y = model.decoder(z)
                recon_loss = F.mse_loss(y, x, reduction='sum')
```

- [ ] **Step 3: Verify — noise fires in both train and val batches, and stays off when disabled**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from unittest.mock import patch
from omegaconf import OmegaConf
import torch
from src.data import create_dataloaders
from src.training import DGDTrainer

def run(noise_scale, tag):
    config = OmegaConf.load('config/config.yaml')
    config.data.total_subset_fraction = 0.02
    config.data.download = False
    config.experiment_name = f'plan_smoke_noise_{tag}'
    config.paths.models_dir = f'/tmp/plan_smoke_models_noise_{tag}'
    config.training.epochs = 2
    config.training.first_epoch_gmm = 1
    config.training.refit_gmm_interval = 1
    config.training.early_stopping_patience = 100
    config.training.latent_noise_scale = noise_scale

    device = torch.device('cpu')
    train_loader, val_loader, test_loader, class_names = create_dataloaders(config)

    call_count = {'n': 0}
    real_randn_like = torch.randn_like
    def spy(*args, **kwargs):
        call_count['n'] += 1
        return real_randn_like(*args, **kwargs)

    trainer = DGDTrainer(config=config, device=device, verbose=False)
    with patch('src.training.trainer.torch.randn_like', side_effect=spy):
        trainer.train(train_loader, val_loader, sample_data=None, class_names=class_names)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    expected = (n_train_batches + n_val_batches) * config.training.epochs if noise_scale > 0 else 0
    print(f'{tag}: randn_like calls={call_count[\"n\"]}, expected={expected} (train_batches={n_train_batches}, val_batches={n_val_batches})')
    assert call_count['n'] == expected, f'{tag}: noise call count mismatch'

run(0.5, 'enabled')
run(0.0, 'disabled')
print('OK: noise fires in every train+val batch when enabled, and not at all when disabled')
"
```
Expected: two `randn_like calls=...` lines where `enabled` shows a nonzero count equal to `(train_batches + val_batches) * 2 epochs`, `disabled` shows `0`, then `OK: ...`.

- [ ] **Step 4: Commit**

```bash
git add src/training/trainer.py
git commit -m "feat: extend latent-space noise injection to the validation phase"
```

---

### Task 2: Replace ARI + silhouette with NMI — `src/training/trainer.py`

**Files:**
- Modify: `src/training/trainer.py:59-63` (tracking lists in `__init__`), `src/training/trainer.py:307-327` (`_safe_silhouette_score` — delete), `src/training/trainer.py:411-485` (GMM-refit computation, both branches), `src/training/trainer.py:655-660` (per-epoch log line), `src/training/trainer.py:713-729` (`training_results.pth` schema)

**Interfaces:**
- Consumes: `tgmm.ClusteringMetrics.normalized_mutual_info_score(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float` (already imported via `from tgmm import GaussianMixture, ClusteringMetrics` at the top of the file — no import change needed).
- Produces: `self.nmi_scores: List[float]`, `self.val_nmi_scores: List[float]` (replacing the four old lists). Checkpoint metadata at GMM-refit epochs now has keys `train_nmi`/`val_nmi` (replacing `train_ari`/`val_ari`/`train_silhouette`/`val_silhouette`). `best/training_results.pth` now has keys `nmi_scores`/`val_nmi_scores` (replacing `ari_scores`/`val_ari_scores`/`silhouette_scores`/`val_silhouette_scores`) — Task 3 (`loss.py`) and Task 4 (`report.py`) consume these by exact name.

- [ ] **Step 1: Rename the tracking lists**

Replace (`trainer.py:59-63`):
```python
        # Clustering metrics tracking
        self.ari_scores = []
        self.val_ari_scores = []
        self.silhouette_scores = []
        self.val_silhouette_scores = []
```
with:
```python
        # Clustering metrics tracking
        self.nmi_scores = []
        self.val_nmi_scores = []
```

- [ ] **Step 2: Delete `_safe_silhouette_score`**

Delete the entire method at `trainer.py:307-327`:
```python
    def _safe_silhouette_score(
        self,
        cluster_metrics: ClusteringMetrics,
        representations: torch.Tensor,
        labels: torch.Tensor,
        n_components: int
    ) -> float:
        """Compute silhouette score, subsampling large sets to bound GPU memory use.

        ClusteringMetrics.silhouette_score materializes a full (N, N) pairwise
        distance matrix, which OOMs once N reaches the tens of thousands. Subsampling
        keeps memory bounded while still giving a representative score, mirroring
        sklearn's silhouette_score `sample_size` parameter.
        """
        max_samples = getattr(self.training_config, 'silhouette_max_samples', 5000)
        n_samples = representations.size(0)
        if n_samples > max_samples:
            idx = torch.randperm(n_samples, device=representations.device)[:max_samples]
            representations = representations[idx]
            labels = labels[idx]
        return cluster_metrics.silhouette_score(representations, labels, n_components)

```
NMI does not need subsampling — it has no O(N²) pairwise-distance step, unlike silhouette — so there is no replacement helper.

- [ ] **Step 3: Replace ARI + silhouette computation with NMI in both GMM-refit branches**

Replace (`trainer.py:411-485`, the full block from the clustering-metrics-calculator comment through the end of the `elif epoch > first_epoch_gmm:` branch):
```python
            # Initialize clustering metrics calculator
            cluster_metrics = ClusteringMetrics()
            current_train_ari = 0.0
            current_val_ari = 0.0
            current_train_silhouette = 0.0
            current_val_silhouette = 0.0

            is_gmm_refit_epoch = epoch == first_epoch_gmm or (refit_gmm_interval and epoch % refit_gmm_interval == 0)

            if is_gmm_refit_epoch:
                with torch.no_grad():
                    representations = rep.z.detach()
                    gmm.fit(representations, max_iter=1000 if epoch == first_epoch_gmm else 100)

                    # Calculate ARI for training data
                    predicted_labels = gmm.predict(representations)
                    current_train_ari = cluster_metrics.adjusted_rand_score(train_labels, predicted_labels)
                    self.ari_scores.append(current_train_ari)

                    # Calculate Silhouette Score for training data
                    current_train_silhouette = self._safe_silhouette_score(cluster_metrics, representations, predicted_labels, gmm.n_components)
                    self.silhouette_scores.append(current_train_silhouette)

                    # Calculate ARI for val data
                    val_representations = val_rep.z.detach()
                    val_predicted_labels = gmm.predict(val_representations)
                    current_val_ari = cluster_metrics.adjusted_rand_score(val_labels, val_predicted_labels)
                    self.val_ari_scores.append(current_val_ari)

                    # Calculate Silhouette Score for val data
                    current_val_silhouette = self._safe_silhouette_score(cluster_metrics, val_representations, val_predicted_labels, gmm.n_components)
                    self.val_silhouette_scores.append(current_val_silhouette)

                # Persist a checkpoint at every GMM-refit epoch
                save_checkpoint(
                    checkpoint_root / f"epoch_{epoch:04d}",
                    model.decoder, rep, val_rep, gmm,
                    metadata={
                        'epoch': epoch,
                        'train_ari': current_train_ari,
                        'val_ari': current_val_ari,
                        'train_silhouette': current_train_silhouette,
                        'val_silhouette': current_val_silhouette,
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

                    # Calculate ARI for training data
                    predicted_labels = gmm.predict(representations)
                    current_train_ari = cluster_metrics.adjusted_rand_score(train_labels, predicted_labels)
                    self.ari_scores.append(current_train_ari)

                    # Calculate Silhouette Score for training data
                    current_train_silhouette = self._safe_silhouette_score(cluster_metrics, representations, predicted_labels, gmm.n_components)
                    self.silhouette_scores.append(current_train_silhouette)

                    # Calculate ARI for val data
                    val_representations = val_rep.z.detach()
                    val_predicted_labels = gmm.predict(val_representations)
                    current_val_ari = cluster_metrics.adjusted_rand_score(val_labels, val_predicted_labels)
                    self.val_ari_scores.append(current_val_ari)

                    # Calculate Silhouette Score for val data
                    current_val_silhouette = self._safe_silhouette_score(cluster_metrics, val_representations, val_predicted_labels, gmm.n_components)
                    self.val_silhouette_scores.append(current_val_silhouette)
```
with:
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

- [ ] **Step 4: Update the per-epoch log line**

Replace (`trainer.py:655-660`):
```python
            train_ari_str = f", ARI={current_train_ari:.4f}, Sil={current_train_silhouette:.4f}" if epoch >= first_epoch_gmm else ""
            val_ari_str = f", ARI={current_val_ari:.4f}, Sil={current_val_silhouette:.4f}" if epoch >= first_epoch_gmm else ""

            print(f"Epoch {epoch}/{self.training_config.epochs} [Time per Epoch: {epoch_time_str}, Remaining Time: {remaining_time_str}, LR: Dec={lr_decoder:.2e}, Rep={lr_rep:.2e}, Noise={noise_scale:.4f}]")
            print(f"       - Train Loss: {train_loss:.4f} (B: {self.best_train_loss:.4f}), Recon: {recon_train_loss:.4f} (B: {self.best_recon_train:.4f}), GMM: {gmm_train_str}{train_ari_str}")
            print(f"       - Val   Loss: {val_loss:.4f} (B: {self.best_val_loss:.4f}), Recon: {recon_val_loss:.4f} (B: {self.best_recon_val:.4f}), GMM: {gmm_val_str}{val_ari_str}")
```
with:
```python
            train_nmi_str = f", NMI={current_train_nmi:.4f}" if epoch >= first_epoch_gmm else ""
            val_nmi_str = f", NMI={current_val_nmi:.4f}" if epoch >= first_epoch_gmm else ""

            print(f"Epoch {epoch}/{self.training_config.epochs} [Time per Epoch: {epoch_time_str}, Remaining Time: {remaining_time_str}, LR: Dec={lr_decoder:.2e}, Rep={lr_rep:.2e}, Noise={noise_scale:.4f}]")
            print(f"       - Train Loss: {train_loss:.4f} (B: {self.best_train_loss:.4f}), Recon: {recon_train_loss:.4f} (B: {self.best_recon_train:.4f}), GMM: {gmm_train_str}{train_nmi_str}")
            print(f"       - Val   Loss: {val_loss:.4f} (B: {self.best_val_loss:.4f}), Recon: {recon_val_loss:.4f} (B: {self.best_recon_val:.4f}), GMM: {gmm_val_str}{val_nmi_str}")
```

- [ ] **Step 5: Update the `training_results.pth` schema**

Replace (`trainer.py:713-729`):
```python
        torch.save({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_train_losses': self.recon_train_losses,
            'recon_val_losses': self.recon_val_losses,
            'gmm_train_losses': self.gmm_train_losses,
            'gmm_val_losses': self.gmm_val_losses,
            'ari_scores': self.ari_scores,
            'val_ari_scores': self.val_ari_scores,
            'silhouette_scores': self.silhouette_scores,
            'val_silhouette_scores': self.val_silhouette_scores,
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
            'nmi_scores': self.nmi_scores,
            'val_nmi_scores': self.val_nmi_scores,
            'learning_rates': self.learning_rates,
            'momentum_betas': self.momentum_betas,
            'epoch_times': self.epoch_times,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }, best_dir / "training_results.pth")
```

- [ ] **Step 6: Verify — no ARI/silhouette symbols remain, NMI schema present, correct list lengths**

```bash
grep -n "adjusted_rand_score\|silhouette_score\|_safe_silhouette_score\|ari_scores\|silhouette_scores" src/training/trainer.py
```
Expected: no output (empty). Then:

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
config.experiment_name = 'plan_smoke_nmi'
config.paths.models_dir = '/tmp/plan_smoke_models_nmi'
config.training.epochs = 3
config.training.first_epoch_gmm = 1
config.training.refit_gmm_interval = 1
config.training.early_stopping_patience = 100
config.training.latent_noise_scale = 0.0

device = torch.device('cpu')
train_loader, val_loader, test_loader, class_names = create_dataloaders(config)

trainer = DGDTrainer(config=config, device=device, verbose=False)
trainer.train(train_loader, val_loader, sample_data=None, class_names=class_names)

assert len(trainer.nmi_scores) == 3, f'expected 3 NMI entries (one per GMM-active epoch), got {len(trainer.nmi_scores)}'
assert len(trainer.val_nmi_scores) == 3
assert not hasattr(trainer, 'ari_scores')
assert not hasattr(trainer, 'silhouette_scores')

best_dir = Path(config.paths.models_dir) / config.experiment_name / 'best'
history = torch.load(best_dir / 'training_results.pth')
assert 'nmi_scores' in history and 'val_nmi_scores' in history
assert 'ari_scores' not in history and 'silhouette_scores' not in history and 'val_ari_scores' not in history and 'val_silhouette_scores' not in history
assert len(history['nmi_scores']) == 3

checkpoint_dirs = sorted((Path(config.paths.models_dir) / config.experiment_name / 'checkpoints').glob('epoch_*'))
last_meta = torch.load(checkpoint_dirs[-1] / 'metadata.pth')
assert 'train_nmi' in last_meta and 'val_nmi' in last_meta, f'checkpoint metadata missing NMI keys: {last_meta.keys()}'
assert 'train_ari' not in last_meta and 'val_ari' not in last_meta and 'train_silhouette' not in last_meta and 'val_silhouette' not in last_meta
print('checkpoint dirs:', [c.name for c in checkpoint_dirs])
print('OK: NMI schema present, ARI/silhouette gone, list lengths correct')
"
```
Expected: `checkpoint dirs: [...]` then `OK: ...`.

- [ ] **Step 7: Commit**

```bash
git add src/training/trainer.py
git commit -m "refactor: replace ARI+silhouette with NMI in trainer.py"
```

---

### Task 3: Rewrite the clustering-metrics panel to single-axis NMI — `src/visualization/loss.py`

**Files:**
- Modify: `src/visualization/loss.py:116-191` (panel 4 of `plot_training_analysis`)

**Interfaces:**
- Consumes: a duck-typed `trainer` object exposing `nmi_scores`, `val_nmi_scores` (Task 2's renamed attributes, or the `SimpleNamespace` Task 4 builds from `training_results.pth`).
- Produces: no signature change to `plot_training_analysis` — same `(train_losses, val_losses, trainer, config, skip_first_epoch=True, save_path=None, show=True) -> plt.Figure`. Panel 4 (`axes[3]`) becomes a single-axis, 2-line plot (no `twinx()`).

- [ ] **Step 1: Replace the ARI+silhouette dual-axis block with a single-axis NMI plot**

Replace (`loss.py:116-191`, the full "Clustering Metrics" block through the closing `else` of that `if`):
```python
    # 4. Clustering Metrics (ARI and Silhouette Score)
    if hasattr(trainer, 'ari_scores') and len(trainer.ari_scores) > 0:
        ari_scores = trainer.ari_scores
        val_ari_scores = trainer.val_ari_scores if hasattr(trainer, 'val_ari_scores') else []
        silhouette_scores = trainer.silhouette_scores if hasattr(trainer, 'silhouette_scores') else []
        val_silhouette_scores = trainer.val_silhouette_scores if hasattr(trainer, 'val_silhouette_scores') else []

        # Find epochs where metrics were computed (non-zero GMM epochs).
        # NOT filtered by start_idx here: ari_scores/silhouette_scores gain one
        # entry per GMM-active epoch unconditionally (trainer.py appends them
        # every time, regardless of skip_first_epoch), so their natural x-axis
        # is unfiltered too. Filtering here would desync the two whenever the
        # GMM is active starting at epoch 1 itself (metric_epochs would drop
        # epoch 1 while the score lists still include its entry).
        metric_epochs = [i+1 for i, x in enumerate(gmm_train_losses) if x != 0]

        # Create twin axes for different y-scales
        ax_ari = axes[3]
        ax_sil = ax_ari.twinx()

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

        # Plot ARI scores
        if len(ari_scores) > 0:
            x, y = _align(metric_epochs, ari_scores)
            line1 = ax_ari.plot(x, y, 'b-', label='Train ARI', linewidth=2, marker='o')
        if len(val_ari_scores) > 0:
            x, y = _align(metric_epochs, val_ari_scores)
            line2 = ax_ari.plot(x, y, 'r-', label='Val ARI', linewidth=2, marker='o')

        # Plot Silhouette scores
        if len(silhouette_scores) > 0:
            x, y = _align(metric_epochs, silhouette_scores)
            line3 = ax_sil.plot(x, y, 'g--', label='Train Silhouette', linewidth=2, marker='s')
        if len(val_silhouette_scores) > 0:
            x, y = _align(metric_epochs, val_silhouette_scores)
            line4 = ax_sil.plot(x, y, 'orange', linestyle='--', label='Val Silhouette', linewidth=2, marker='s')

        ax_ari.set_xlabel('Epoch')
        ax_ari.set_ylabel('ARI Score', color='b')
        ax_sil.set_ylabel('Silhouette Score', color='g')
        ax_ari.set_title('Clustering Metrics (ARI & Silhouette)')
        ax_ari.tick_params(axis='y', labelcolor='b')
        ax_sil.tick_params(axis='y', labelcolor='g')
        ax_ari.grid(True, alpha=0.3)

        # Combine legends
        lines = []
        labels = []
        if len(ari_scores) > 0:
            lines.extend(line1)
            labels.append('Train ARI')
        if len(val_ari_scores) > 0:
            lines.extend(line2)
            labels.append('Val ARI')
        if len(silhouette_scores) > 0:
            lines.extend(line3)
            labels.append('Train Silhouette')
        if len(val_silhouette_scores) > 0:
            lines.extend(line4)
            labels.append('Val Silhouette')
        ax_ari.legend(lines, labels, loc='best')
    else:
        axes[3].text(0.5, 0.5, 'No clustering metrics\navailable', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Clustering Metrics')
```
with:
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

- [ ] **Step 2: Verify — renders without error, single axis (no `twinx`), no ARI/silhouette symbols remain**

```bash
grep -n "ari\|silhouette\|twinx" src/visualization/loss.py
```
Expected: no output (empty).

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
from types import SimpleNamespace
from omegaconf import OmegaConf
from src.visualization.loss import plot_training_analysis

config = OmegaConf.create({'training': {'first_epoch_gmm': 2}})
trainer_view = SimpleNamespace(
    recon_train_losses=[1.0, 0.9, 0.8],
    recon_val_losses=[1.1, 1.0, 0.9],
    gmm_train_losses=[0.0, 0.5, 0.4],
    gmm_val_losses=[0.0, 0.6, 0.5],
    nmi_scores=[0.1, 0.2],
    val_nmi_scores=[0.08, 0.18],
)
fig = plot_training_analysis(
    train_losses=[2.0, 1.4, 1.2], val_losses=[2.1, 1.5, 1.3],
    trainer=trainer_view, config=config, save_path='/tmp/plan_smoke_loss_curves.png', show=False,
)
assert len(fig.axes) == 4, f'expected 4 axes (no twinx), got {len(fig.axes)}'
print('titles:', [ax.get_title() for ax in fig.axes])
print('OK: single-axis NMI panel renders correctly')
"
```
Expected: `titles: [..., 'Clustering Quality (NMI)']` then `OK: ...`.

- [ ] **Step 3: Commit**

```bash
git add src/visualization/loss.py
git commit -m "refactor: single-axis NMI plot replaces dual-axis ARI+silhouette panel"
```

---

### Task 4: Read the NMI schema in the post-hoc figure generator — `src/visualization/report.py`

**Files:**
- Modify: `src/visualization/report.py:113-116`

**Interfaces:**
- Consumes: `training_results.pth` keys `nmi_scores`/`val_nmi_scores` (Task 2's schema).
- Produces: no signature change to `generate_training_figures`.

- [ ] **Step 1: Update the `SimpleNamespace` construction**

Replace (`report.py:113-116`):
```python
        ari_scores=history['ari_scores'],
        val_ari_scores=history['val_ari_scores'],
        silhouette_scores=history['silhouette_scores'],
        val_silhouette_scores=history['val_silhouette_scores'],
```
with:
```python
        nmi_scores=history['nmi_scores'],
        val_nmi_scores=history['val_nmi_scores'],
```

- [ ] **Step 2: Verify — full pipeline (tiny train → generate_training_figures) runs end to end**

```bash
grep -n "ari\|silhouette" src/visualization/report.py
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
config.experiment_name = 'plan_smoke_report'
config.paths.models_dir = '/tmp/plan_smoke_models_report'
config.paths.figures_dir = '/tmp/plan_smoke_figures_report'
config.training.epochs = 3
config.training.first_epoch_gmm = 1
config.training.refit_gmm_interval = 1
config.training.early_stopping_patience = 100
config.training.latent_noise_scale = 0.1

device = torch.device('cpu')
train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
sample_data = get_sample_batches(train_loader, val_loader, device=device, n_per_class=2, n_classes=len(class_names))
train_labels = collect_all_labels(train_loader)
val_labels = collect_all_labels(val_loader)

trainer = DGDTrainer(config=config, device=device, verbose=False)
trainer.train(train_loader, val_loader, sample_data, class_names)

exp_dir = Path(config.paths.models_dir) / config.experiment_name
figures_dir = Path(config.paths.figures_dir) / config.experiment_name
generate_training_figures(exp_dir, figures_dir, class_names, train_labels, val_labels, sample_data, device=device)

loss_curves = figures_dir / 'loss_curves.png'
assert loss_curves.exists() and loss_curves.stat().st_size > 0
print('OK: generate_training_figures ran end-to-end with the NMI schema, loss_curves.png written')
"
```
Expected: `OK: ...` with no traceback.

- [ ] **Step 3: Commit**

```bash
git add src/visualization/report.py
git commit -m "refactor: report.py reads NMI schema from training_results.pth"
```

---

### Task 5: Noise in the M-step loop + NMI swap — `notebooks/dgd_test_inference.ipynb`

**Files:**
- Modify: `notebooks/dgd_test_inference.ipynb` (cell 0 markdown header, cell 5 M-step optimization loop, cell 7 final metrics/figures cell) — edited via a Python script operating on the notebook's JSON, not by hand.

**Interfaces:**
- Consumes: `config.training.latent_noise_scale`/`latent_noise_start`/`latent_noise_end` (existing config keys, no new surface), `tgmm.ClusteringMetrics.normalized_mutual_info_score` (already imported in cell 1 via `from tgmm import ClusteringMetrics`).
- Produces: no change to the notebook's external outputs (`test_representation.pt`, figures under `figures_dir`) beyond the plot title and printed metric switching from `ARI: .../Sil: ...` to `NMI: ...`.

- [ ] **Step 1: Write and run the notebook-patching script**

Create `/tmp/claude-1000/-home-asp-Downloads-HeaDS-ImageDGD/4e7da3f5-6e40-417f-bdc5-1b55f6e03f0f/scratchpad/patch_test_inference_notebook.py`:

```python
import json
from pathlib import Path

nb_path = Path("notebooks/dgd_test_inference.ipynb")
nb = json.loads(nb_path.read_text())

# --- Cell 0: markdown header — drop the dangling reference to the deleted
# dgd_generative_model.ipynb, add a note about the noise deviation from a
# textbook-clean MAP estimate. ---
old_header = "".join(nb["cells"][0]["source"])
assert "dgd_generative_model.ipynb" in old_header, "cell 0 source has changed unexpectedly"
new_header = """# DGD Test Inference (Algorithm 2)

Loads the frozen `best/` decoder + GMM and optimizes a fresh representation layer for the genuinely held-out test split -- the first real use of `test_loader` anywhere in this codebase.

**Note:** the optimization below applies the same latent-space noise regularization (`training.latent_noise_scale`/`training.latent_noise_start`/`training.latent_noise_end`) that training uses, annealed over the `M` optimization steps instead of epochs. This is a deliberate deviation from a textbook-clean MAP estimate of `z`: the frozen decoder and GMM were themselves trained under noise, so evaluating against a clean (noise-free) `z` would silently shift the operating point relative to what the model was optimized for."""
nb["cells"][0]["source"] = new_header.splitlines(keepends=True)

# --- Cell 5: M-step optimization loop — add a per-step noise scale, cosine
# annealed over m=1..M (mirrors trainer.py's per-epoch schedule). ---
old_loop = "".join(nb["cells"][5]["source"])
assert "for m in range(1, M + 1):" in old_loop, "cell 5 source has changed unexpectedly"
new_loop = '''# Algorithm 2, lines 2-5: optimize test_rep alone against the frozen decoder+GMM.
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

# Same cosine-anneal noise schedule training uses, mapped onto step index m
# instead of epoch. Reuses training's config keys verbatim -- no separate
# inference-noise config surface.
latent_noise_scale = config.training.get('latent_noise_scale', 0.0)
noise_start = config.training.get('latent_noise_start', 1.0)
noise_end = config.training.get('latent_noise_end', 0.01)

print(f"Optimizing {n_test} test representations for {M} steps (prior warm-up: {M0} steps)...")

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

    if m % max(1, M // 10) == 0 or m == M:
        print(f"Step {m}/{M}: loss={total_loss/n_test:.4f}, recon={total_recon/n_test:.4f}, gmm={total_gmm/n_test:.4f}, noise={noise_scale_m:.4f}")

print("Test representation optimization complete.")
'''
nb["cells"][5]["source"] = new_loop.splitlines(keepends=True)
nb["cells"][5]["execution_count"] = None
nb["cells"][5]["outputs"] = []

# --- Cell 7: final metrics/figures cell — ARI+silhouette -> NMI. ---
old_figs = "".join(nb["cells"][7]["source"])
assert "adjusted_rand_score" in old_figs, "cell 7 source has changed unexpectedly"
new_figs = '''figures_dir = Path(config.paths.figures_dir) / f"{config.experiment_name}_test_inference"
figures_dir.mkdir(parents=True, exist_ok=True)

test_labels = collect_all_labels(test_loader)

with torch.no_grad():
    predicted_labels = gmm.predict(test_rep.z.detach())

cluster_metrics = ClusteringMetrics()
test_nmi = cluster_metrics.normalized_mutual_info_score(test_labels, predicted_labels)

print(f"Test NMI: {test_nmi:.4f}")

plot_latent_space(
    representations=test_rep.z.detach(), labels=test_labels, gmm=gmm, class_names=class_names,
    title=f"Test Latent Space (Algorithm 2 inference) - NMI: {test_nmi:.4f}",
    save_path=str(figures_dir / "latent_test.png"), show=False,
)

indices_test_sample, images_test_sample, labels_test_sample = collect_class_samples(
    test_loader, n_per_class=5, n_classes=len(class_names)
)
indices_test_sample = indices_test_sample.to(device)

with torch.no_grad():
    recon_test_sample = decoder(test_rep(indices_test_sample))

plot_images_by_class(
    images=recon_test_sample, labels=labels_test_sample, class_names=class_names,
    title="Test: Reconstructed Images by Class (Algorithm 2 inference)", n_per_class=5, cmap='viridis',
    save_path=str(figures_dir / "recon_test.png"), show=False,
)

print(f"Figures written to {figures_dir}")
'''
nb["cells"][7]["source"] = new_figs.splitlines(keepends=True)
nb["cells"][7]["execution_count"] = None
nb["cells"][7]["outputs"] = []

nb_path.write_text(json.dumps(nb, indent=1))
print("Patched cells 0, 5, 7")
```

Run it: `/home/asp/.venvs/rapids_cu13/bin/python /tmp/claude-1000/-home-asp-Downloads-HeaDS-ImageDGD/4e7da3f5-6e40-417f-bdc5-1b55f6e03f0f/scratchpad/patch_test_inference_notebook.py` (run from the repo root: `cd /home/asp/Downloads/HeaDS/ImageDGD` first, since the script reads/writes `notebooks/...` relative to cwd).

- [ ] **Step 2: Verify — notebook JSON is valid, dangling reference and ARI/silhouette symbols are gone**

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
import json
from pathlib import Path
nb = json.loads(Path('notebooks/dgd_test_inference.ipynb').read_text())
assert nb['nbformat'] == 4
full_text = json.dumps(nb)
assert 'dgd_generative_model.ipynb' not in full_text
assert 'adjusted_rand_score' not in full_text
assert 'silhouette_score' not in full_text
assert 'normalized_mutual_info_score' in full_text
assert 'noise_scale_m' in full_text
print('cells:', [c['cell_type'] for c in nb['cells']])
print('OK: notebook JSON valid, dangling reference and ARI/silhouette gone, NMI + noise present')
"
```
Expected: `cells: ['markdown', 'code', 'code', 'code', 'code', 'code', 'code', 'code']` then `OK: ...`.

- [ ] **Step 3: Verify — the M-step noise schedule and frozen-decoder/NMI logic, exercised standalone against a real tiny checkpoint**

This mirrors the cell 5 and cell 7 logic exactly (same formulas, same call signatures) but runs as a plain script against a freshly trained tiny checkpoint, so it can assert on intermediate behavior a notebook can't easily assert on itself: the noise schedule's endpoints, that noise is actually applied when enabled and not when disabled, and that the frozen decoder is untouched by the optimization.

```bash
/home/asp/.venvs/rapids_cu13/bin/python -c "
import math
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from tgmm import ClusteringMetrics
from src.data import create_dataloaders, collect_all_labels
from src.training import DGDTrainer
from src.models import RepresentationLayer
from src.utils.checkpoint import load_checkpoint
from src.models import ConvDecoder

# 1. Train a tiny model to get a real frozen decoder+GMM to run inference against.
config = OmegaConf.load('config/config.yaml')
config.data.total_subset_fraction = 0.02
config.data.download = False
config.experiment_name = 'plan_smoke_inference'
config.paths.models_dir = '/tmp/plan_smoke_models_inference'
config.training.epochs = 2
config.training.first_epoch_gmm = 1
config.training.refit_gmm_interval = 1
config.training.early_stopping_patience = 100
config.training.latent_noise_scale = 0.1
config.training.inference.epochs = 5
config.training.inference.prior_warmup_steps = 0

device = torch.device('cpu')
train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
trainer = DGDTrainer(config=config, device=device, verbose=False)
trainer.train(train_loader, val_loader, sample_data=None, class_names=class_names)

best_dir = Path(config.paths.models_dir) / config.experiment_name / 'best'
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
decoder = checkpoint['decoder']
decoder.eval()
for p in decoder.parameters():
    p.requires_grad_(False)
gmm = checkpoint['gmm']
decoder_state_before = {k: v.clone() for k, v in decoder.state_dict().items()}

# 2. Mirror cell 5's M-step loop against test_loader (standing in for the
# notebook's held-out test split), with noise enabled.
test_rep = RepresentationLayer(
    dim=config.model.representation.n_features, n_samples=len(test_loader.dataset),
    dist='normal', dist_params={}, device=device,
)
test_optimizer = torch.optim.AdamW(test_rep.parameters(), lr=config.training.optimizer.representation.lr)
M = config.training.inference.epochs
M0 = config.training.inference.prior_warmup_steps
lambda_gmm = config.training.lambda_gmm
latent_noise_scale = config.training.get('latent_noise_scale', 0.0)
noise_start = config.training.get('latent_noise_start', 1.0)
noise_end = config.training.get('latent_noise_end', 0.01)

schedule = []
call_count = {'n': 0}
real_randn_like = torch.randn_like
def spy(*args, **kwargs):
    call_count['n'] += 1
    return real_randn_like(*args, **kwargs)

with patch('torch.randn_like', side_effect=spy):
    for m in range(1, M + 1):
        test_optimizer.zero_grad()
        if latent_noise_scale > 0:
            progress = (m - 1) / max(M - 1, 1)
            noise_scale_m = noise_end + (noise_start - noise_end) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            noise_scale_m = 0.0
        schedule.append(noise_scale_m)
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
                loss = recon_loss
            loss.backward()
        test_optimizer.step()

assert abs(schedule[0] - noise_start) < 1e-9, f'first-step noise should equal noise_start, got {schedule[0]}'
assert abs(schedule[-1] - noise_end) < 1e-9, f'last-step noise should equal noise_end, got {schedule[-1]}'
assert call_count['n'] == M * len(test_loader), f'expected {M * len(test_loader)} randn_like calls, got {call_count[\"n\"]}'

decoder_state_after = decoder.state_dict()
for k in decoder_state_before:
    assert torch.equal(decoder_state_before[k], decoder_state_after[k]), f'decoder param {k} changed -- decoder must stay frozen'

# 3. Mirror cell 7's NMI computation.
test_labels = collect_all_labels(test_loader)
with torch.no_grad():
    predicted_labels = gmm.predict(test_rep.z.detach())
cluster_metrics = ClusteringMetrics()
test_nmi = cluster_metrics.normalized_mutual_info_score(test_labels, predicted_labels)
assert 0.0 <= test_nmi <= 1.0, f'NMI out of range: {test_nmi}'

print(f'noise schedule: first={schedule[0]:.4f}, last={schedule[-1]:.4f}')
print(f'test NMI: {test_nmi:.4f}')
print('OK: noise schedule correct, decoder stayed frozen, NMI computed')
"
```
Expected: `noise schedule: first=1.0000, last=0.0100`, a `test NMI: ...` line, then `OK: ...`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/dgd_test_inference.ipynb
git commit -m "feat: add noise regularization to Algorithm 2 M-step loop, swap ARI+silhouette for NMI"
```

---

## Final Integration Check

After all five tasks are committed, confirm the whole feature holds together as one working pipeline (not just per-task in isolation):

```bash
grep -rn "adjusted_rand_score\|silhouette_score\|_safe_silhouette_score\|ari_scores\|silhouette_scores\|dgd_generative_model" src/ notebooks/dgd_test_inference.ipynb
```
Expected: no output (empty) — confirms Task 2/3/4/5's removals didn't leave a stray reference anywhere outside the files each task explicitly targeted (e.g. no leftover mention in a docstring or comment this plan didn't quote verbatim).

```bash
grep -n "model.decoder.training and noise_scale" src/training/trainer.py
```
Expected: no output (empty) — confirms the old train-only gate is gone from both loop sites, not just one.

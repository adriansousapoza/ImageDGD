# Noise Injection Everywhere + NMI Clustering Metric — Design

## Problem

Two changes to the training/inference pipeline built in the train/val/test split refactor:

1. **Latent-space noise injection is train-only.** `DGDTrainer.train()` adds Gaussian noise to
   `z` before decoding and before scoring against the GMM, but only inside the training batch
   loop (gated on `model.decoder.training`). Validation, and the separate held-out-inference
   notebook (`dgd_test_inference.ipynb`, Algorithm 2), never see this noise. For the noise to
   act as a real robustness regularizer rather than a training-time-only artifact, it needs to
   apply everywhere a representation is optimized against the frozen or live decoder.
2. **ARI + silhouette are the wrong clustering-quality metrics for this project going forward.**
   Replace both with Normalized Mutual Information (NMI), computed once per split instead of
   two metrics on two different scales.

## Goals

1. Validation, during training, gets the same per-epoch noise injection as the training batch
   loop — same schedule, same effect on both the reconstruction and GMM loss terms.
2. The held-out-inference notebook's M-step optimization gets an equivalent noise schedule,
   annealed over `m = 1..M` instead of epoch, reusing the existing
   `training.latent_noise_scale`/`latent_noise_start`/`latent_noise_end` config keys (no new
   config surface).
3. `best_epoch`/`best_val_loss` selection continues to use a single (now noised) validation
   pass — no second, clean validation pass is added. This is an accepted tradeoff: model
   selection becomes somewhat noisier, mitigated by the schedule annealing to
   `latent_noise_end` (small) by the end of training.
4. ARI and silhouette are removed everywhere (not aliased, not kept alongside NMI) and replaced
   with `ClusteringMetrics.normalized_mutual_info_score(labels_true, labels_pred) -> float`
   (from `tgmm`), computed once per split (train, val, and — in the inference notebook — test).
5. `notebooks/dgd_generative_model.ipynb` (the untracked math/design notebook whose Algorithm 1
   prose describes recording "ARI, silhouette", and whose Algorithm 2 the inference notebook
   still names in its header) is deleted outright rather than kept in sync — it was never
   committed to git, so this is a plain file removal, no history to consider.

## Non-goals

- No second "clean" validation pass for model selection — noisy `val_loss` is the sole
  selection signal, unchanged in mechanism from today, just now noise-influenced.
- No change to how clustering-quality metrics are computed *from* — NMI, like the ARI it
  replaces, is computed from the clean, stored `rep.z`/`val_rep.z`/`test_rep.z` (never the
  transient noised copy used only inside the loss computation). Swapping the metric does not
  change what representations it's computed against.
- No change to the GMM refit cadence, checkpoint cadence, or any other mechanism carried over
  from the train/val/test split refactor.

## 1. Noise injection — `src/training/trainer.py`

The per-epoch noise-scale calculation (cosine anneal, `latent_noise_start` → `latent_noise_end`
over `training.epochs`) is unchanged and computed once per epoch, before the training batch
loop, exactly as today. What changes is where it's applied and how it's gated.

Today, the training batch loop gates noise injection on `model.decoder.training`:

```python
if model.decoder.training and noise_scale > 0:
    noise = torch.randn_like(z) * noise_scale
    z = z + noise
```

`model.decoder.training` was only ever a proxy for "is this the train phase, not val" — the
decoder is switched to `.eval()` before the validation phase runs. Since validation should now
receive noise too, the gate becomes simply:

```python
if noise_scale > 0:
    noise = torch.randn_like(z) * noise_scale
    z = z + noise
```

applied identically in both the training batch loop (before `y = model.decoder(z)` and the GMM
score) and the validation batch loop (same placement, before `y = model.decoder(z)` and the GMM
score). The same `noise_scale` value (computed once per epoch) is reused in both loops — there
is no separate/independent schedule for validation.

Nothing else about the validation phase changes: the decoder is still frozen
(`requires_grad = False` on its parameters) during the val phase, only `val_rep` receives
gradients, and `best_model_state`/`best_val_loss`/`best_epoch` selection logic is untouched
(it already just reads whatever `val_loss` comes out of the val phase — that loss is now
noise-influenced, per Goal 3).

## 2. Noise injection — `notebooks/dgd_test_inference.ipynb`

The M-step optimization loop (cell containing `for m in range(1, M + 1):`) gains a per-step
noise scale, computed with the same cosine-anneal formula as training but mapped onto step
index instead of epoch:

```python
noise_start = config.training.get('latent_noise_start', 1.0)
noise_end = config.training.get('latent_noise_end', 0.01)
latent_noise_scale = config.training.get('latent_noise_scale', 0.0)

...
for m in range(1, M + 1):
    if latent_noise_scale > 0:
        progress = (m - 1) / max(M - 1, 1)
        noise_scale_m = noise_end + (noise_start - noise_end) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        noise_scale_m = 0.0
    ...
    for index, x, _ in test_loader:
        ...
        z = test_rep(index)
        if noise_scale_m > 0:
            z = z + torch.randn_like(z) * noise_scale_m
        y = decoder(z)
        recon_loss = F.mse_loss(y, x, reduction='sum')
        if m >= M0:
            gmm_error = -lambda_gmm * torch.sum(gmm.score_samples(z))
            ...
```

No new config keys — this reuses `training.latent_noise_scale`/`latent_noise_start`/
`latent_noise_end` verbatim, the same values training uses. The notebook's markdown header
gains a short note that this optimization includes the same noise regularization training uses,
so a reader comparing it against a clean-`argmin` description of Algorithm 2 isn't confused by
the discrepancy — the notebook this used to point to for that formal description is being
deleted (see Goal 5), so there's no longer a "conflicting" external doc to reconcile against,
but the note documents the deliberate deviation from a textbook-clean MAP estimate for anyone
reading the notebook in isolation.

## 3. `notebooks/dgd_generative_model.ipynb` — delete

Untracked file (never committed). Deleted outright, not edited. No further action needed beyond
`rm`.

## 4. ARI + silhouette → NMI

`tgmm.ClusteringMetrics.normalized_mutual_info_score(labels_true, labels_pred) -> float` is the
replacement for `adjusted_rand_score`. Unlike silhouette, it does not need subsampling (no O(N²)
pairwise distance matrix), so `DGDTrainer._safe_silhouette_score` is deleted outright, not just
its call sites.

### `src/training/trainer.py`

- Tracking lists: `self.ari_scores`, `self.val_ari_scores`, `self.silhouette_scores`,
  `self.val_silhouette_scores` → `self.nmi_scores`, `self.val_nmi_scores` (two lists, not four).
- Every `cluster_metrics.adjusted_rand_score(...)` / `self._safe_silhouette_score(...)` pair of
  calls (there are two call sites — the `is_gmm_refit_epoch` branch and the `elif epoch >
  first_epoch_gmm` branch, both currently computing train+val ARI and train+val silhouette)
  becomes one `cluster_metrics.normalized_mutual_info_score(...)` call per split (train, val).
- `_safe_silhouette_score` method: deleted.
- Per-epoch log line: `ARI=0.0003, Sil=0.0800` → `NMI=0.0800` (train and val each get one
  number instead of two).
- Checkpoint metadata dicts written at GMM-refit epochs: `train_ari`/`val_ari`/
  `train_silhouette`/`val_silhouette` → `train_nmi`/`val_nmi`.
- `training_results.pth` schema: `ari_scores`/`val_ari_scores`/`silhouette_scores`/
  `val_silhouette_scores` → `nmi_scores`/`val_nmi_scores`. This is a load-bearing schema change
  — `report.py`'s reader must match exactly (see below).

### `src/visualization/loss.py`

`plot_training_analysis`'s panel 4 currently plots ARI on a primary y-axis and silhouette on a
`twinx()` secondary axis (4 lines total: train/val ARI, train/val silhouette). With only NMI
left, train and val NMI share one axis, one scale — the `twinx()` call and the dual-axis
legend-combining logic are removed; panel 4 becomes a single-axis, 2-line plot titled
"Clustering Quality (NMI)", axis label "NMI Score", lines labeled "Train NMI"/"Val NMI". The
`metric_epochs` x-axis alignment logic (fixed
yesterday for the ARI/silhouette length-mismatch bug) carries over unchanged in spirit — it
still aligns `metric_epochs` against whichever score list it's paired with — but now only needs
to serve two lists instead of four.

### `src/visualization/report.py`

The `SimpleNamespace` built from `training_results.pth` for `plot_training_analysis`/
`plot_training_dynamics` reads `nmi_scores`/`val_nmi_scores` instead of the four old keys,
matching trainer.py's new schema exactly.

### `notebooks/dgd_test_inference.ipynb`

The final figures cell's `test_ari`/`test_silhouette` computation (including the silhouette
subsampling block) collapses into:

```python
test_nmi = cluster_metrics.normalized_mutual_info_score(test_labels, predicted_labels)
```

used in the `print(...)` and the latent-space plot's title (replacing `ARI: {test_ari:.4f},
Sil: {test_silhouette:.4f}` with `NMI: {test_nmi:.4f}`).

## Testing / verification approach

Consistent with the rest of this codebase: no pytest suite, standalone runnable Python scripts
against a tiny real-data config (matching the pattern used throughout the train/val/test split
refactor), run via `/home/asp/.venvs/rapids_cu13/bin/python`. Specifically:

- A tiny end-to-end training run (small `total_subset_fraction`, few epochs, `latent_noise_scale
  > 0`) confirming: (a) validation's reported loss differs run-to-run in a way consistent with
  noise injection (i.e., isn't identical to a noise-free baseline), (b) `nmi_scores`/
  `val_nmi_scores` are present in `training_results.pth` with the expected length, (c) no
  `ari_scores`/`silhouette_scores` keys remain anywhere.
- `generate_training_figures` run against that tiny checkpoint, confirming `loss_curves.png`
  renders without error (panel 4 now single-axis) and no `AttributeError`/`KeyError` from the
  schema change.
- The inference notebook's core M-step loop logic exercised standalone (mirroring how the
  original Algorithm-2 loop was verified): confirm noise actually perturbs `z` during the loop
  (non-zero effect when `latent_noise_scale > 0`) and that the frozen decoder's parameters are
  still untouched after optimization (regression check against the property already verified
  when that notebook was first built).

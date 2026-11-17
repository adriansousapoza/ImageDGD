"""
Latent space visualization utilities.
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Import tgmm plotting helpers
try:
    from tgmm.plotting import plot_gmm as _tgmm_plot_gmm
    TGMM_PLOTTING_AVAILABLE = True
except ImportError:
    TGMM_PLOTTING_AVAILABLE = False
    warnings.warn("tgmm.plotting not available. GMM cluster figures will be skipped.")


def _try_cuml_import():
    """Try to import cuML components, fall back to sklearn if unavailable."""
    try:
        from cuml import PCA as cuPCA
        from cuml import UMAP as cuUMAP
        from cuml import TSNE as cuTSNE
        return cuPCA, cuUMAP, cuTSNE, True
    except ImportError:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        try:
            from umap import UMAP
        except ImportError:
            UMAP = None
        return PCA, UMAP, TSNE, False


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@contextlib.contextmanager
def _suppress_stdout_lines(*substrings: str):
    """
    Suppress stdout lines containing any of `substrings` for the duration of
    the block, passing everything else through unchanged.

    cuML's TSNE/UMAP occasionally print benign auto-tuning/fallback notices
    (e.g. "# of Nearest Neighbors should be at least 3 * perplexity",
    "Spectral initialization failed, using random initialization instead")
    as raw prints from their C++/CUDA implementation rather than through
    cuml's own logger -- logger.set_level has no effect on them (confirmed
    empirically) -- so filtering the printed text is the only way to hide
    them without changing the underlying behavior they're reporting on.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    for line in buf.getvalue().splitlines(keepends=True):
        if not any(sub in line for sub in substrings):
            sys.stdout.write(line)


def _cuml_to_numpy(x):
    """Convert a cuDF/cuPy result to numpy; pass through if already numpy."""
    if hasattr(x, 'to_numpy'):
        return x.to_numpy()
    if hasattr(x, 'values'):
        return x.values
    return x


def _compute_embeddings(
    z: np.ndarray,
    extra_points: Optional[np.ndarray] = None,
    random_state: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Compute PCA, UMAP, and t-SNE 2D embeddings of z, once.

    If extra_points is given (e.g. GMM component means), it is placed into
    the same embedding space as z: exactly, via `.transform()`, for PCA (a
    linear map, so this is lossless); by fitting UMAP/t-SNE jointly on
    concat([z, extra_points]) and splitting the result for UMAP/t-SNE, since
    neither has an exact transform for held-out points -- folding the extra
    points into the same fit is the standard way to place them consistently
    in the embedding used for the data, instead of running a second,
    unrelated fit that would land them anywhere.

    Returns a dict with key 'using_cuml' and one entry per method ('pca',
    'umap', 'tsne'). Each is either None (method unavailable or failed) or a
    dict with 'z' (N x 2 ndarray) and 'extra' (K x 2 ndarray, or None if
    extra_points was None). 'pca' additionally carries 'model' (the fitted
    PCA, for further exact transforms) and 'var_text' (explained variance).
    """
    PCA, UMAP, TSNE, using_cuml = _try_cuml_import()
    n_extra = 0 if extra_points is None else len(extra_points)

    if using_cuml:
        import cudf
        z_gpu = cudf.DataFrame(z)
    else:
        z_gpu = z

    result = {'using_cuml': using_cuml}

    # --- PCA: exact transform for extra_points, no concatenation needed ---
    if verbose:
        print("Computing PCA...")
    pca = PCA(n_components=2) if using_cuml else PCA(n_components=2, random_state=random_state)
    z_pca = pca.fit_transform(z_gpu)
    z_pca = _cuml_to_numpy(z_pca) if using_cuml else z_pca

    var_ratio = getattr(pca, 'explained_variance_ratio_', None)
    if var_ratio is not None:
        var_ratio = _cuml_to_numpy(var_ratio) if using_cuml else var_ratio
        var_text = f"({var_ratio[0]*100:.1f}%, {var_ratio[1]*100:.1f}%)"
    else:
        var_text = ""

    extra_pca = None
    if extra_points is not None:
        if using_cuml:
            extra_pca = _cuml_to_numpy(pca.transform(cudf.DataFrame(extra_points)))
        else:
            extra_pca = pca.transform(extra_points)

    result['pca'] = {
        'z': np.asarray(z_pca),
        'extra': None if extra_pca is None else np.asarray(extra_pca),
        'model': pca,
        'var_text': var_text,
    }

    # --- UMAP: fit once on concat([z, extra_points]), split the result ---
    if UMAP is not None:
        if verbose:
            print("Computing UMAP...")
        try:
            if n_extra:
                joint = np.vstack([z, extra_points])
                umap_input = cudf.DataFrame(joint) if using_cuml else joint
            else:
                umap_input = z_gpu
            umap = UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
            # Spectral initialization can fail on a disconnected/degenerate
            # neighbor graph (e.g. near-duplicate points from a checkpoint
            # early in training, or the epoch-0 "zeros" init where every
            # representation starts at the same point) -- cuML already
            # handles this gracefully by falling back to random init on its
            # own, it just prints a benign notice when it does. Same
            # raw-C++-print situation as TSNE above, so same fix.
            with _suppress_stdout_lines("Spectral initialization failed"):
                z_umap = umap.fit_transform(umap_input)
            z_umap = np.asarray(_cuml_to_numpy(z_umap) if using_cuml else z_umap)
            if n_extra:
                result['umap'] = {'z': z_umap[:-n_extra], 'extra': z_umap[-n_extra:]}
            else:
                result['umap'] = {'z': z_umap, 'extra': None}
        except Exception as e:
            if verbose:
                print(f"UMAP failed: {e}")
            result['umap'] = None
    else:
        result['umap'] = None

    # --- t-SNE: same joint-fit approach as UMAP ---
    if verbose:
        print("Computing t-SNE...")
    try:
        if n_extra:
            joint = np.vstack([z, extra_points])
            tsne_input = cudf.DataFrame(joint) if using_cuml else joint
        else:
            tsne_input = z_gpu
        if using_cuml:
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, n_neighbors=3 * 30 + 1)
            # cuML's adaptive_learning (on by default) recomputes n_neighbors
            # from n_samples for any real dataset (n_samples > 2000),
            # overriding the n_neighbors passed above -- e.g. ~34 at
            # n_samples=56000 -- which falls below 3*perplexity and prints a
            # benign "# of Nearest Neighbors should be at least 3 *
            # perplexity" notice. This is expected behavior of cuML's own
            # auto-tuning (not a sign perplexity/n_neighbors are misconfigured
            # here) -- see _suppress_stdout_lines for why filtering is the
            # only way to hide it.
            with _suppress_stdout_lines("Nearest Neighbors should be at least"):
                z_tsne = tsne.fit_transform(tsne_input)
        else:
            tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, max_iter=1000)
            z_tsne = tsne.fit_transform(tsne_input)
        z_tsne = np.asarray(_cuml_to_numpy(z_tsne) if using_cuml else z_tsne)
        if n_extra:
            result['tsne'] = {'z': z_tsne[:-n_extra], 'extra': z_tsne[-n_extra:]}
        else:
            result['tsne'] = {'z': z_tsne, 'extra': None}
    except Exception as e:
        if verbose:
            print(f"t-SNE failed: {e}")
        result['tsne'] = None

    return result


def _plot_2d_projection(
    ax: plt.Axes,
    z_2d: np.ndarray,
    labels: Optional[np.ndarray],
    class_names: Optional[list],
    title: str,
    alpha: float,
    s: int,
    cmap: str,
    xlabel: str = 'Component 1',
    ylabel: str = 'Component 2',
):
    """Scatter a 2D embedding, colored by class label if given."""
    if labels is not None:
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, alpha=alpha, s=s, cmap=cmap)

        if class_names is not None:
            unique_labels = np.unique(labels)
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=scatter.cmap(scatter.norm(label)),
                                 markersize=8, label=class_names[int(label)])
                      for label in unique_labels if int(label) < len(class_names)]
            ax.legend(handles=handles, loc='upper right', fontsize=8)
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=alpha, s=s, c='blue')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _overlay_gmm_means(ax, means_2d: np.ndarray, marker: str = 'X', size: int = 90) -> None:
    """Scatter GMM component means on top of an already-drawn 2D projection."""
    ax.scatter(means_2d[:, 0], means_2d[:, 1], c='black', marker=marker, s=size,
               edgecolor='white', linewidths=1.2, zorder=6)


def _overlay_confidence_ellipses(ax, means_2d: np.ndarray, covs_2d: np.ndarray, confidence: float = 0.95) -> None:
    """Draw one Mahalanobis-distance confidence ellipse per GMM component, at
    the given confidence level (default 95%), on an already-fit 2D
    (necessarily linear, i.e. PCA) projection. The ellipse radius is
    sqrt(chi2.ppf(confidence, df=2)) standard deviations, not `confidence`
    standard deviations directly -- e.g. ~2.4477, not 2, for 95% (the two
    only coincide in 1D). Same relationship tgmm.plotting.plot_gmm's own
    ellipse_std_devs uses (see confidence_ellipse_std_devs there) --
    recomputed directly via scipy here so this overlay doesn't depend on
    tgmm.plotting being importable (only tgmm.GaussianMixture, always needed
    just to have a `gmm` to plot)."""
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2

    std_dev = float(np.sqrt(chi2.ppf(confidence, df=2)))
    for mean, cov in zip(means_2d, covs_2d):
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2.0 * std_dev * np.sqrt(np.clip(vals, 0, None))
        ax.add_patch(Ellipse(
            (mean[0], mean[1]), width, height, angle=angle,
            facecolor='none', edgecolor='black', linewidth=1.2,
            linestyle='--', alpha=0.8, zorder=5,
        ))


def plot_latent_space(
    representations: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    class_names: Optional[list] = None,
    title: str = "Latent Space Visualization",
    figsize: Tuple[int, int] = (20, 6),
    alpha: float = 0.6,
    s: int = 20,
    cmap: str = 'tab10',
    save_path: Optional[str] = None,
    show: bool = True,
    random_state: int = 42,
    verbose: bool = False,
    gmm=None,
    confidence_level: float = 0.95,
) -> plt.Figure:
    """
    Visualize latent space using PCA, UMAP, and t-SNE, colored by class
    label. Uses cuML for GPU acceleration when available.

    For a GMM-cluster-colored view with per-component ellipses/means as the
    primary subject (rather than an overlay) and one file per cluster, see
    generate_latent_space_figures.

    Parameters:
    ----------
    representations: Latent representations (N x D tensor)
    labels: Optional class labels (N tensor)
    class_names: Optional list of class names
    title: Main title for the figure
    figsize: Figure size
    alpha: Point transparency
    s: Point size
    cmap: Colormap for points
    save_path: Optional path to save the figure
    show: Whether to display the figure
    random_state: Random seed for reproducibility
    verbose: Whether to print diagnostic information
    gmm : fitted GaussianMixture, optional
        If given, overlay component means (black X) on all three panels, and
        a `confidence_level` confidence ellipse per component (PCA panel
        only -- UMAP/t-SNE are nonlinear, so a covariance ellipse computed in
        the original space has no faithful equivalent there). None (default)
        draws the plain label-colored view with no GMM overlay -- e.g. for a
        checkpoint saved before the GMM's first fit.
    confidence_level : float, default 0.95
        Confidence level for the PCA-panel ellipses (ignored if gmm is None).

    Returns:
    -------
    matplotlib.figure.Figure: The created figure
    """
    z = _to_numpy(representations)
    labels_np = _to_numpy(labels) if labels is not None else None
    means = None if gmm is None else _to_numpy(gmm.means_)

    embeddings = _compute_embeddings(z, extra_points=means, random_state=random_state, verbose=verbose)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    _plot_2d_projection(axes[0], embeddings['pca']['z'], labels_np, class_names,
                        f"PCA {embeddings['pca']['var_text']}", alpha, s, cmap)
    if gmm is not None:
        means_2d_pca, covs_2d_pca = _project_gmm_to_pca(gmm, embeddings['pca']['model'], embeddings['using_cuml'])
        _overlay_confidence_ellipses(axes[0], means_2d_pca, covs_2d_pca, confidence=confidence_level)
        _overlay_gmm_means(axes[0], means_2d_pca)
        axes[0].text(0.02, 0.98, f"✕ = GMM mean (k={gmm.n_components})\n"
                     f"dashed = {confidence_level*100:.0f}% CI",
                     transform=axes[0].transAxes, fontsize=8, va='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if embeddings['umap'] is not None:
        _plot_2d_projection(axes[1], embeddings['umap']['z'], labels_np, class_names, "UMAP", alpha, s, cmap)
        if gmm is not None and embeddings['umap']['extra'] is not None:
            _overlay_gmm_means(axes[1], embeddings['umap']['extra'])
    else:
        axes[1].text(0.5, 0.5, 'UMAP not available\nInstall: pip install umap-learn',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("UMAP")

    if embeddings['tsne'] is not None:
        _plot_2d_projection(axes[2], embeddings['tsne']['z'], labels_np, class_names, "t-SNE", alpha, s, cmap)
        if gmm is not None and embeddings['tsne']['extra'] is not None:
            _overlay_gmm_means(axes[2], embeddings['tsne']['extra'])
    else:
        axes[2].text(0.5, 0.5, 't-SNE failed', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("t-SNE")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


##############################################################################
# Noise-injection diagnostics: clean-vs-noised population + single-point
# noise ball, both in PCA space only (see plot_noise_comparison docstring
# for why PCA-only, not UMAP/t-SNE).
##############################################################################

def _fit_pca_only(z: np.ndarray, random_state: int = 42):
    """Fit a fresh 2-component PCA on z. Returns (pca_model, z_2d, using_cuml, var_text)."""
    PCA, _, _, using_cuml = _try_cuml_import()
    if using_cuml:
        import cudf
        pca = PCA(n_components=2)
        z_2d = _cuml_to_numpy(pca.fit_transform(cudf.DataFrame(z)))
        var_ratio = _cuml_to_numpy(pca.explained_variance_ratio_)
    else:
        pca = PCA(n_components=2, random_state=random_state)
        z_2d = pca.fit_transform(z)
        var_ratio = pca.explained_variance_ratio_
    var_text = f"({var_ratio[0]*100:.1f}%, {var_ratio[1]*100:.1f}%)"
    return pca, np.asarray(z_2d), using_cuml, var_text


def _pca_transform(pca_model, points: np.ndarray, using_cuml: bool) -> np.ndarray:
    """Transform points through an already-fit PCA (see _project_gmm_to_pca for the same pattern)."""
    if using_cuml:
        import cudf
        return np.asarray(_cuml_to_numpy(pca_model.transform(cudf.DataFrame(points))))
    return np.asarray(pca_model.transform(points))


def plot_noise_comparison(
    representations: torch.Tensor,
    noise_scale: float,
    point_indices: List[int],
    point_labels: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
    n_noise_draws: int = 100,
    title: str = "Noise Injection",
    random_state: int = 42,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    labels: Optional[torch.Tensor] = None,
    track_labels: Optional[torch.Tensor] = None,
    z_clean: Optional[torch.Tensor] = None,
    z_noised_real: Optional[torch.Tensor] = None,
    z_after: Optional[torch.Tensor] = None,
) -> plt.Figure:
    """
    Two-panel diagnostic of latent-space noise injection at a given
    noise_scale, both panels in PCA space only: a linear projection is the
    only one where displacement in the picture is a faithful (uniformly
    scaled) picture of displacement in real z-space -- UMAP/t-SNE would
    distort the apparent size of the noise ball in ways that don't reflect
    what the model actually sees.

    noise_scale is treated the same way training treats it (see
    noise_std_for_dim in src/utils/schedules.py): a dimension-independent
    *target expected displacement*, not the raw per-coordinate std. The
    actual noise drawn below uses noise_std_for_dim(noise_scale, d) as its
    per-coordinate std, so E[||eps||] ~ noise_scale regardless of d.

    Left panel: every point's clean z (blue) vs. the same points with noise
    added (red), both projected through one PCA fit on the clean points.

    Right panel: for each of point_indices (typically one representative
    point per class), n_noise_draws sampled noise draws scattered around it,
    color-coded to match point_labels/class_names if given. Illustrative
    only -- these are freshly sampled at plot time, not the actual noise
    draws a real training step used (for that, see
    plot_noise_optimization_shift, which plots the real captured values).
    Annotated with:
      - noise_scale: the target expected displacement (dimension-independent)
      - per-coordinate std: noise_scale/sqrt(d) (d = representation
        dimensionality) -- the actual std each coordinate is drawn from. A 2D
        PCA projection of isotropic noise in D dimensions only shows a
        fraction of the true spread, so the picture alone understates the
        true displacement -- this and the next number correct for that.
      - median NN distance: median nearest-neighbor distance among this
        split's own points, computed directly in raw z-space (no GMM
        dependency, so this works even for a pre-GMM-fit checkpoint).
      - ratio: noise_scale / median NN distance. This is the number worth
        tracking across checkpoints -- it's dimensionless, so it stays
        comparable even though PCA is refit independently per checkpoint
        (axes can rotate/flip between epochs, so the *picture* alone is not
        a valid before/after comparison).

    Optional third panel: when `z_clean`/`z_noised_real`/`z_after`/
    `track_labels`/`labels` are all given (the real per-epoch trajectory
    captured in trainer.py's z_trajectory.pth -- see
    plot_noise_optimization_shift, whose single-panel logic this reuses),
    draws the real clean->noised->after-optimization-step shift for the same
    tracked points, in the *same* PCA fit as the left/right panels above
    (rather than a separately-fit one), so all three panels share one
    projection. Omit them (the default) for the original two-panel figure.

    Parameters
    ----------
    representations : torch.Tensor
        This split's full set of representations (N x D)
    noise_scale : float
        The scheduled target displacement at this checkpoint (0 disables
        noise -- the right panel then just shows the points with no draws)
    point_indices : list of int
        Rows of `representations` to scatter noise draws around. Pass the
        same indices across every checkpoint for a given split (e.g. via
        pick_class_representative_indices) so the picture is comparable over
        the course of training.
    point_labels : list of int, optional
        Class label for each of point_indices, for coloring + legend. If
        omitted, points are colored individually rather than by class.
    class_names : list of str, optional
        Display names for point_labels. Required for named legend entries;
        ignored if point_labels is None.
    n_noise_draws : int
        Number of noise draws to scatter around each point.
    labels, track_labels, z_clean, z_noised_real, z_after : optional
        Enable the third (zshift) panel -- see above. `labels` is this
        split's full per-row class labels (for the panel's background
        scatter); the rest mirror plot_noise_optimization_shift's own
        same-named parameters exactly.
    """
    from sklearn.neighbors import NearestNeighbors
    from ..utils.schedules import noise_std_for_dim

    has_zshift = z_clean is not None and z_noised_real is not None and z_after is not None and track_labels is not None
    n_panels = 3 if has_zshift else 2
    if figsize is None:
        figsize = (20, 6) if has_zshift else (14, 6)

    z = _to_numpy(representations)
    n, d = z.shape
    rng = np.random.RandomState(random_state)
    noise_std = noise_std_for_dim(noise_scale, d)

    pca, z_2d, using_cuml, var_text = _fit_pca_only(z, random_state=random_state)

    if noise_std > 0:
        z_noised = z + rng.randn(n, d) * noise_std
        z_noised_2d = _pca_transform(pca, z_noised, using_cuml)
    else:
        z_noised_2d = z_2d.copy()

    # Median nearest-neighbor distance in raw z-space -- no GMM dependency,
    # so this is safe even for the pre-training (epoch 0) checkpoint.
    nn = NearestNeighbors(n_neighbors=2).fit(z)
    nn_dist, _ = nn.kneighbors(z)
    median_nn_dist = float(np.median(nn_dist[:, 1]))

    ratio = noise_scale / median_nn_dist if median_nn_dist > 0 else float('inf')

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # --- Left: population clean vs. noised (unchanged) ---
    axes[0].scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.4, s=12, c='tab:blue', label='clean z')
    axes[0].scatter(z_noised_2d[:, 0], z_noised_2d[:, 1], alpha=0.4, s=12, c='tab:red',
                     label=f'z + noise (target={noise_scale:.4f})')
    axes[0].set_title(f"PCA {var_text} -- clean vs. noised")
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Right: one noise ball per tracked point, color-coded by class ---
    point_indices = [int(i) % n for i in point_indices]
    points = z[point_indices]
    points_2d = _pca_transform(pca, points, using_cuml)
    if point_labels is not None and class_names is not None:
        palette = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        point_colors = [palette[label] for label in point_labels]
        point_names = [class_names[label] for label in point_labels]
    else:
        palette = plt.cm.tab10(np.linspace(0, 1, len(point_indices)))
        point_colors = list(palette)
        point_names = [f'point #{idx}' for idx in point_indices]

    axes[1].scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.15, s=8, c='lightgray', label='all points (clean)')
    for i in range(len(point_indices)):
        if noise_std > 0:
            draws = points[i][None, :] + rng.randn(n_noise_draws, d) * noise_std
            draws_2d = _pca_transform(pca, draws, using_cuml)
            axes[1].scatter(draws_2d[:, 0], draws_2d[:, 1], alpha=0.35, s=10, color=point_colors[i])
        axes[1].scatter([points_2d[i, 0]], [points_2d[i, 1]], s=110, color=point_colors[i],
                         edgecolor='black', marker='*', linewidths=1, zorder=5, label=point_names[i])
    axes[1].set_title(f"{len(point_indices)}-point noise draws ({n_noise_draws} each)")
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    axes[1].grid(True, alpha=0.3)

    annotation = (
        f"noise_scale (target) = {noise_scale:.4f}\n"
        f"per-coordinate std = {noise_std:.4f}  (= target/√d, d={d})\n"
        f"median NN distance = {median_nn_dist:.4f}\n"
        f"ratio (target / NN dist) = {ratio:.3f}"
    )
    axes[1].text(0.02, 0.02, annotation, transform=axes[1].transAxes, fontsize=9,
                 va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # --- Optional third panel: real clean -> noised -> after-optimizer-step shift ---
    if has_zshift:
        _draw_zshift_panel(
            axes[2], pca, using_cuml, var_text, z_2d, _to_numpy(labels), class_names,
            track_labels, z_clean, z_noised_real, z_after,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


##############################################################################
# GMM cluster figures: gmm_overview.png + one cluster{rank}.png per component
##############################################################################

def _project_gmm_to_pca(gmm, pca_model, using_cuml: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a fitted n-dimensional GaussianMixture's means and covariances
    into the 2D space of an already-fit 2-component PCA. Valid because PCA
    is linear: the mean projects via pca.transform, the covariance projects
    via the rotation matrix (components @ cov @ components.T).

    Returns (means_2d, covariances_2d), shapes (n_components, 2) and
    (n_components, 2, 2).
    """
    means = _to_numpy(gmm.means_)
    covariances = _to_numpy(gmm.covariances_)
    n_components = gmm.n_components
    n_features = gmm.n_features

    if using_cuml:
        import cudf
        means_2d = _cuml_to_numpy(pca_model.transform(cudf.DataFrame(means)))
        components = _cuml_to_numpy(pca_model.components_)
    else:
        means_2d = pca_model.transform(means)
        components = pca_model.components_
    means_2d = np.asarray(means_2d)[:, :2]
    components = np.asarray(components)

    covs_2d = np.zeros((n_components, 2, 2))
    cov_type = gmm.covariance_type
    for i in range(n_components):
        if cov_type == 'full':
            cov = covariances[i]
        elif cov_type == 'diag':
            cov = np.diag(covariances[i])
        elif cov_type == 'spherical':
            cov = covariances[i] * np.eye(n_features)
        elif cov_type == 'tied_full':
            cov = covariances
        elif cov_type == 'tied_diag':
            cov = np.diag(covariances)
        elif cov_type == 'tied_spherical':
            var = covariances.item() if hasattr(covariances, 'item') else covariances
            cov = var * np.eye(n_features)
        else:
            raise ValueError(f"Unsupported covariance_type: {cov_type}")
        covs_2d[i] = components @ cov @ components.T

    return means_2d, covs_2d


class _Projected2DGMM:
    """
    Duck-typed 2D view of a fitted n-dimensional GaussianMixture, so it can
    be handed to tgmm.plotting.plot_gmm -- which only supports GMMs that
    live natively in 2D (it reads .means_/.covariances_ as already-2D, and
    unconditionally calls .predict(X) on whatever 2D X it's given).

    If `real_labels` is given, predict() returns it regardless of the X
    argument: the correct cluster assignment for each point was already
    computed once, in the real n-dimensional latent space -- predicting in
    the lossy 2D projection would disagree with it, and with how every other
    panel in these figures colors the same points. This is necessary, not a
    bug: plot_gmm has no hook for "use these precomputed labels".

    If `real_labels` is None (single-cluster highlight views, where X only
    ever contains that one cluster's own members), every passed-in point
    belongs to the one component being shown, so predict() returns
    all-zeros sized to match X.
    """
    covariance_type = 'full'

    def __init__(self, means_2d: np.ndarray, covariances_2d: Optional[np.ndarray],
                 weights: np.ndarray, real_labels: Optional[torch.Tensor] = None):
        self.means_ = torch.as_tensor(means_2d, dtype=torch.float32)
        self.covariances_ = None if covariances_2d is None else torch.as_tensor(covariances_2d, dtype=torch.float32)
        self.weights_ = torch.as_tensor(weights, dtype=torch.float32)
        self.n_components = len(weights)
        self._real_labels = real_labels

    def predict(self, X):
        if self._real_labels is not None:
            return self._real_labels
        return torch.zeros(X.shape[0], dtype=torch.int64)


def _gmm_panel(ax, z_2d, gmm_view, title, xlabel, ylabel, show_ellipses,
                legend_labels=None, ellipse_std_devs=(1, 2, 3), point_size=8, legend=True,
                cluster_colors='turbo'):
    # plot_gmm labels points "Cluster N" and ellipses "Component N" as separate
    # legend entries -- with many components that's a 2N-entry legend that
    # dwarfs the plot (see gmm_overview's all-cluster panels). Off there;
    # left on for the single-cluster highlight views, where 1-2 entries and
    # the raw-index/weight text in legend_labels are actually useful.
    X = torch.as_tensor(z_2d, dtype=torch.float32)
    _tgmm_plot_gmm(
        X, gmm=gmm_view,
        color_by_cluster=True,
        show_ellipses=show_ellipses,
        ellipse_std_devs=list(ellipse_std_devs),
        point_size=point_size,
        title=title,
        xlabel=xlabel, ylabel=ylabel,
        legend_labels=legend_labels,
        legend=legend,
        cluster_colors=cluster_colors,
        ax=ax,
    )


def generate_latent_space_figures(
    representations: torch.Tensor,
    labels: torch.Tensor,
    gmm,
    class_names: List[str],
    save_dir,
    title_prefix: str = "",
    ellipse_std_devs: Tuple[int, ...] = (1, 2, 3),
    random_state: int = 42,
    verbose: bool = False,
) -> None:
    """
    Write one split's full latent-space diagnostic figure set to save_dir:

      overview.png           -- PCA/UMAP/t-SNE, colored by ground-truth
                                 label, no GMM (see plot_latent_space)
      gmm_overview.png       -- same 3 embeddings, colored by predicted GMM
                                 cluster; the PCA panel also draws the
                                 ellipse_std_devs confidence ellipses for
                                 every component
      cluster{rank:02d}.png  -- one file per GMM component, ranked by
                                 descending weight (rank 1 = highest weight):
                                 that component's members are colored and
                                 its ellipse/mean shown, every other point is
                                 drawn as a small black dot

    PCA/UMAP/t-SNE are each computed once -- with the GMM means folded into
    the same UMAP/t-SNE fit so their positions land consistently in the same
    embedding as the data -- and reused for every file above, so every plot
    in save_dir shows the same point layout and dimensionality reduction
    never reruns per file.
    """
    if not TGMM_PLOTTING_AVAILABLE:
        warnings.warn("tgmm.plotting not available -- skipping GMM cluster figures.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    z = _to_numpy(representations)
    labels_np = _to_numpy(labels)
    means = _to_numpy(gmm.means_)
    weights = _to_numpy(gmm.weights_)
    n_components = gmm.n_components

    # Rank components by descending weight: rank 0 (displayed as "1") = highest weight.
    rank_order = np.argsort(weights)[::-1]
    weights_ranked = weights[rank_order]

    z_tensor = representations if isinstance(representations, torch.Tensor) else torch.as_tensor(z, dtype=torch.float32)
    real_pred = _to_numpy(gmm.predict(z_tensor)).astype(np.int64)
    rank_of_original = np.empty(n_components, dtype=np.int64)
    rank_of_original[rank_order] = np.arange(n_components)
    real_pred_rank = torch.as_tensor(rank_of_original[real_pred], dtype=torch.int64)

    embeddings = _compute_embeddings(z, extra_points=means, random_state=random_state, verbose=verbose)

    # --- overview.png: plain label-colored, no GMM ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{title_prefix}Latent Space", fontsize=16, fontweight='bold')
    _plot_2d_projection(axes[0], embeddings['pca']['z'], labels_np, class_names,
                        f"PCA {embeddings['pca']['var_text']}", 0.6, 20, 'tab10')
    if embeddings['umap'] is not None:
        _plot_2d_projection(axes[1], embeddings['umap']['z'], labels_np, class_names, "UMAP", 0.6, 20, 'tab10')
    else:
        axes[1].text(0.5, 0.5, 'UMAP not available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("UMAP")
    if embeddings['tsne'] is not None:
        _plot_2d_projection(axes[2], embeddings['tsne']['z'], labels_np, class_names, "t-SNE", 0.6, 20, 'tab10')
    else:
        axes[2].text(0.5, 0.5, 't-SNE failed', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("t-SNE")
    plt.tight_layout()
    fig.savefig(save_dir / "overview.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Project GMM means/covariances into PCA space (exact, linear) ---
    means_2d_pca, covs_2d_pca = _project_gmm_to_pca(gmm, embeddings['pca']['model'], embeddings['using_cuml'])
    means_2d_pca_ranked = means_2d_pca[rank_order]
    covs_2d_pca_ranked = covs_2d_pca[rank_order]

    umap_means_ranked = None if embeddings['umap'] is None else embeddings['umap']['extra'][rank_order]
    tsne_means_ranked = None if embeddings['tsne'] is None else embeddings['tsne']['extra'][rank_order]

    sigma_text = ", ".join(f"{sd}σ" for sd in ellipse_std_devs)

    # One pastel color per component (rank order), reused for both
    # gmm_overview.png (all clusters at once) and each cluster{rank}.png (that
    # component's own highlight color) -- so a given cluster is the same color
    # everywhere it appears, not just within a single figure.
    pastel_colors = sns.color_palette('pastel', n_components).as_hex()

    # --- gmm_overview.png: all clusters ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{title_prefix}GMM Clusters", fontsize=16, fontweight='bold')

    # legend=False here: plot_gmm labels points "Cluster N" and ellipses
    # "Component N" as separate entries, so with N clusters a legend would
    # show 2N lines -- unreadable, and it dwarfed the plot in testing (see
    # _gmm_panel). Per-cluster identity is what the individual cluster{rank}
    # files are for; this overview is for the overall shape/separation.
    all_view_pca = _Projected2DGMM(means_2d_pca_ranked, covs_2d_pca_ranked, weights_ranked, real_pred_rank)
    _gmm_panel(axes[0], embeddings['pca']['z'], all_view_pca, f"PCA -- {sigma_text} Ellipses",
               'Component 1', 'Component 2', show_ellipses=True, ellipse_std_devs=ellipse_std_devs,
               legend=False, cluster_colors=pastel_colors)

    if umap_means_ranked is not None:
        all_view_umap = _Projected2DGMM(umap_means_ranked, None, weights_ranked, real_pred_rank)
        _gmm_panel(axes[1], embeddings['umap']['z'], all_view_umap, "UMAP -- Cluster Means",
                   'UMAP 1', 'UMAP 2', show_ellipses=False, legend=False, cluster_colors=pastel_colors)
    else:
        axes[1].text(0.5, 0.5, 'UMAP not available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("UMAP")

    if tsne_means_ranked is not None:
        all_view_tsne = _Projected2DGMM(tsne_means_ranked, None, weights_ranked, real_pred_rank)
        _gmm_panel(axes[2], embeddings['tsne']['z'], all_view_tsne, "t-SNE -- Cluster Means",
                   't-SNE 1', 't-SNE 2', show_ellipses=False, legend=False, cluster_colors=pastel_colors)
    else:
        axes[2].text(0.5, 0.5, 't-SNE failed', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("t-SNE")

    plt.tight_layout()
    fig.savefig(save_dir / "gmm_overview.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- cluster{rank:02d}.png: one per component, highlighted ---
    for rank in range(n_components):
        weight = float(weights_ranked[rank])
        member_mask = (real_pred_rank == rank).numpy()
        label_text = f"Cluster {rank + 1} (weight={weight:.4f})"

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"{title_prefix}{label_text}", fontsize=16, fontweight='bold')

        panel_specs = [(axes[0], embeddings['pca']['z'], means_2d_pca_ranked[rank:rank + 1],
                        covs_2d_pca_ranked[rank:rank + 1], True, f"PCA -- {sigma_text} Ellipse",
                        'Component 1', 'Component 2')]
        if umap_means_ranked is not None:
            panel_specs.append((axes[1], embeddings['umap']['z'], umap_means_ranked[rank:rank + 1],
                                None, False, "UMAP", 'UMAP 1', 'UMAP 2'))
        else:
            axes[1].text(0.5, 0.5, 'UMAP not available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title("UMAP")
        if tsne_means_ranked is not None:
            panel_specs.append((axes[2], embeddings['tsne']['z'], tsne_means_ranked[rank:rank + 1],
                                None, False, "t-SNE", 't-SNE 1', 't-SNE 2'))
        else:
            axes[2].text(0.5, 0.5, 't-SNE failed', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title("t-SNE")

        for ax, panel_z, mean_1, cov_1, show_ell, panel_title, xlabel, ylabel in panel_specs:
            ax.scatter(panel_z[~member_mask, 0], panel_z[~member_mask, 1],
                      c='black', s=3, alpha=0.3, zorder=1)
            single_view = _Projected2DGMM(mean_1, cov_1, weights_ranked[rank:rank + 1], real_labels=None)
            _gmm_panel(ax, panel_z[member_mask], single_view, panel_title, xlabel, ylabel,
                       show_ellipses=show_ell, legend_labels=[label_text], ellipse_std_devs=ellipse_std_devs,
                       cluster_colors=[pastel_colors[rank]])

        plt.tight_layout()
        fig.savefig(save_dir / f"cluster{rank + 1:02d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)


##############################################################################
# Representation trajectory diagnostic: does noise durably move a point, or
# does re-optimizing it just undo the perturbation?
##############################################################################

def plot_representation_trajectory(
    decoder: torch.nn.Module,
    gmm,
    representations: torch.Tensor,
    labels: torch.Tensor,
    images: torch.Tensor,
    class_names: List[str],
    noise_std: float,
    lambda_gmm: float = 1.0,
    gmm_active: bool = True,
    rep_lr: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    n_steps: int = 15,
    point_indices: Optional[torch.Tensor] = None,
    random_state: int = 42,
    title: str = "Representation Trajectory Under Noise + Optimization",
    figsize: Tuple[int, int] = (9, 9),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    For one point per class: record its clean position, then run n_steps of
    the same mechanics a real representation goes through each epoch/step
    (fresh noise draw -> frozen decoder(+GMM) forward -> backward -> AdamW
    step on z alone), tracking the full path. Answers "does noise durably
    move a representation, or does optimizing it just undo the
    perturbation?" empirically, on a real checkpoint, instead of by theory.

    decoder and gmm are used frozen (eval mode, no_grad on their own
    parameters) -- only the marked points' z is optimized, isolating
    representation-level dynamics from any simultaneous decoder/GMM change,
    the same setup dgd_test_inference.ipynb's Algorithm 2 uses.

    A useful before/after signal beyond the picture: compare
    ||final - clean|| to the first step's ||noised - clean||. If the ratio is
    << 1, optimization is undoing the noise as you'd naively expect. If it's
    >= 1 -- especially once noise_std is small -- something other than noise
    (typically the GMM's pull) is doing more of the moving than noise itself.

    Parameters
    ----------
    decoder, gmm : the frozen model this checkpoint's representations were
        trained against. gmm may be None (equivalent to gmm_active=False).
    representations : this split's full set of representations (N x D) --
        also used to fit the PCA projection and draw the background scatter,
        so trajectories are shown in a stable, meaningful frame.
    labels : this split's true class labels (N,), used for coloring and for
        picking one representative point per class when point_indices=None.
    images : this split's true images (N x C x H x W), used to compute
        recon_loss for the marked points during optimization.
    noise_std : per-coordinate std to draw fresh noise from at every step
        (already dimension-corrected -- see noise_std_for_dim in
        src/utils/schedules.py -- not the raw target displacement).
    gmm_active : whether the GMM term is included in the loss (matches
        `epoch >= first_epoch_gmm` in trainer.py for the checkpoint you're
        probing). Ignored if gmm is None.
    rep_lr, betas, eps, weight_decay : representation optimizer
        hyperparameters -- pass the run's own config.training.optimizer.representation
        values (or lr_scheduler.base_lr_representation if the LR schedule is
        enabled) so the diagnostic matches what actually happened at this
        point in training.
    n_steps : number of optimization steps to run, each with an
        independently-drawn noise sample (matching how training redraws
        noise every step, never reusing a draw).
    point_indices : which row of `representations` to track per class. If
        None, picks the point closest to each class's own centroid (a stable,
        reproducible choice rather than a random one).
    """
    device = representations.device
    n_total, n_features = representations.shape
    n_classes = len(class_names)

    if point_indices is None:
        point_indices = []
        for c in range(n_classes):
            class_idx = torch.where(labels == c)[0]
            class_z = representations[class_idx]
            centroid = class_z.mean(dim=0)
            closest = class_idx[torch.argmin(((class_z - centroid) ** 2).sum(dim=1))]
            point_indices.append(closest.item())
        point_indices = torch.tensor(point_indices)

    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    z_clean = representations[point_indices].detach().clone()
    x_targets = images[point_indices].to(device)
    n_points = len(point_indices)

    z_param = z_clean.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [z_param], lr=rep_lr, betas=betas, eps=eps, weight_decay=weight_decay,
    )

    trajectory = [z_clean.clone()]
    first_noised = None
    rng = torch.Generator(device=device).manual_seed(random_state)
    use_gmm = gmm_active and (gmm is not None)
    for step in range(n_steps):
        optimizer.zero_grad()
        noise = torch.randn(z_param.shape, generator=rng, device=device) * noise_std
        z_noised = z_param + noise
        if step == 0:
            first_noised = z_noised.detach().clone()

        y = decoder(z_noised)
        recon_loss = F.mse_loss(y, x_targets, reduction='sum') / n_points
        if use_gmm:
            gmm_error = -lambda_gmm * torch.sum(gmm.score_samples(z_noised)) / n_points
            loss = recon_loss + gmm_error
        else:
            loss = recon_loss
        loss.backward()
        optimizer.step()
        trajectory.append(z_param.detach().clone())

    trajectory = torch.stack(trajectory)  # (n_steps+1, n_points, n_features)
    final_z = trajectory[-1]
    dist_final_to_clean = (final_z - z_clean).norm(dim=1)
    dist_noised1_to_clean = (first_noised - z_clean).norm(dim=1)

    pca, z2d_full, using_cuml, var_text = _fit_pca_only(_to_numpy(representations), random_state=random_state)
    traj2d = np.stack([
        _pca_transform(pca, _to_numpy(trajectory[t]), using_cuml) for t in range(trajectory.shape[0])
    ])

    fig, ax = plt.subplots(figsize=figsize)
    labels_np = _to_numpy(labels)
    ax.scatter(z2d_full[:, 0], z2d_full[:, 1], c=labels_np, cmap='tab10', alpha=0.15, s=6)
    colors = plt.cm.tab10(np.linspace(0, 1, n_points))
    for i, c in enumerate(_to_numpy(labels[point_indices]).tolist()):
        path = traj2d[:, i, :]
        ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], alpha=0.6, linewidth=1)
        ax.scatter(path[0, 0], path[0, 1], marker='*', s=200, color=colors[i], edgecolor='black',
                   zorder=5, label=f"{class_names[c]} (clean start)")
        ax.scatter(path[-1, 0], path[-1, 1], marker='X', s=120, color=colors[i], edgecolor='black', zorder=5)
    ax.set_title(f"{title}\nPCA {var_text} -- star=clean start, X=after {n_steps} steps, "
                 f"background=all points by class")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))

    ratio_text = "\n".join(
        f"{class_names[c]:15s} final/first-noise = {(dist_final_to_clean[i] / dist_noised1_to_clean[i]).item():.2f}"
        for i, c in enumerate(_to_numpy(labels[point_indices]).tolist())
    )
    print(f"||final - clean|| / ||first noise draw - clean|| per point "
          f"(<<1 = optimization undoes noise, >=1 = something else is moving the point more than noise does):")
    print(ratio_text)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


##############################################################################
# Real (not simulated) per-epoch clean -> noised -> post-optimization shift,
# from trainer.py's z_trajectory.pth.
##############################################################################

def _draw_zshift_panel(
    ax, pca, using_cuml: bool, var_text: str,
    z2d_full: np.ndarray, labels_np: np.ndarray, class_names: List[str],
    track_labels: torch.Tensor, z_clean: torch.Tensor, z_noised: torch.Tensor, z_after: torch.Tensor,
) -> None:
    """Draw the clean->noised->after trajectory (star->circle->X per tracked
    point) onto an already-created axes, using an already-fit PCA. Shared by
    plot_noise_optimization_shift (its own standalone figure) and
    plot_noise_comparison (as an optional third panel, reusing that
    function's own PCA fit so all panels share one projection)."""
    def _proj(z):
        return _pca_transform(pca, _to_numpy(z), using_cuml)

    clean_2d = _proj(z_clean)
    noised_2d = _proj(z_noised)
    after_2d = _proj(z_after)

    dist_noised_to_clean = (z_noised - z_clean).norm(dim=1)
    dist_after_to_clean = (z_after - z_clean).norm(dim=1)

    ax.scatter(z2d_full[:, 0], z2d_full[:, 1], c=labels_np, cmap='tab10', alpha=0.15, s=6, zorder=1)

    palette = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    track_labels_list = _to_numpy(track_labels).tolist()
    for i, c in enumerate(track_labels_list):
        color = palette[c]
        ax.plot([clean_2d[i, 0], noised_2d[i, 0], after_2d[i, 0]],
                 [clean_2d[i, 1], noised_2d[i, 1], after_2d[i, 1]],
                 '-', color=color, alpha=0.7, linewidth=1.2, zorder=4)
        ax.scatter(clean_2d[i, 0], clean_2d[i, 1], marker='*', s=190, color=color,
                    edgecolor='black', linewidths=1, zorder=6, label=f"{class_names[c]}")
        ax.scatter(noised_2d[i, 0], noised_2d[i, 1], marker='o', s=90, color=color,
                    edgecolor='black', linewidths=1, zorder=5)
        ax.scatter(after_2d[i, 0], after_2d[i, 1], marker='X', s=120, color=color,
                    edgecolor='black', linewidths=1, zorder=6)

    ax.set_title(f"PCA {var_text} -- star=clean, circle=noised, X=after optimizer step")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0), title='class (star=clean)')

    annotation = "‖after−clean‖ / ‖noised−clean‖ per point:\n" + "\n".join(
        f"{class_names[c]:14s} {(dist_after_to_clean[i] / dist_noised_to_clean[i]).item():.2f}"
        if dist_noised_to_clean[i] > 0 else f"{class_names[c]:14s}  n/a (no noise)"
        for i, c in enumerate(track_labels_list)
    )
    ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=7,
             va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), zorder=10)


def plot_noise_optimization_shift(
    representations: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    track_labels: torch.Tensor,
    z_clean: torch.Tensor,
    z_noised: torch.Tensor,
    z_after: torch.Tensor,
    title: str = "Noise + Optimization Shift",
    random_state: int = 42,
    figsize: Tuple[int, int] = (9, 9),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    For a fixed set of tracked points (one per class), plots three *real*
    positions captured from an actual training epoch: the clean z before
    that epoch's noise, z immediately after the real noise draw that epoch's
    forward pass actually used, and z after that epoch's representation-
    optimizer step. Every position here is the literal value trainer.py's
    real training loop produced (see z_trajectory.pth, written alongside
    every plot_interval checkpoint) -- unlike plot_representation_trajectory,
    which simulates its own steps against a frozen decoder+GMM to probe the
    dynamics in isolation, this is what actually happened.

    Answers, on real data: does the noised position get walked back toward
    clean (the update mostly undoes the noise), does "after" end up somewhere
    else entirely (something other than noise dominates the update), or does
    it stay near the noised position (the perturbation persists)?

    Parameters
    ----------
    representations, labels : this split's full representation set + labels,
        for the background scatter and the PCA fit (stable, meaningful frame).
    track_labels : class label per tracked point, shape (n_tracked,).
    z_clean, z_noised, z_after : shape (n_tracked, D) each -- exactly the
        dict trainer.py writes into z_trajectory.pth's 'train'/'val' entries
        (as z_clean/z_noised/z_after).
    """
    pca, z2d_full, using_cuml, var_text = _fit_pca_only(_to_numpy(representations), random_state=random_state)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_zshift_panel(
        ax, pca, using_cuml, var_text, z2d_full, _to_numpy(labels), class_names,
        track_labels, z_clean, z_noised, z_after,
    )
    ax.set_title(f"{title}\n{ax.get_title()}")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

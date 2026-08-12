"""
Latent space visualization utilities.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

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
            # cuML's default n_neighbors (90) sits exactly on 3*perplexity
            # (30), and cuML's internal check requires strictly more than
            # that boundary -- the default always triggers a spurious "# of
            # Nearest Neighbors should be at least 3 * perplexity" warning.
            # One above the boundary silences it with no effect on results.
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, n_neighbors=3 * 30 + 1)
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
) -> plt.Figure:
    """
    Visualize latent space using PCA, UMAP, and t-SNE, colored by class
    label. Uses cuML for GPU acceleration when available.

    For a GMM-cluster-colored view with component ellipses/means instead of
    (or alongside) this label-colored view, see generate_latent_space_figures.

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

    Returns:
    -------
    matplotlib.figure.Figure: The created figure
    """
    z = _to_numpy(representations)
    labels_np = _to_numpy(labels) if labels is not None else None

    embeddings = _compute_embeddings(z, random_state=random_state, verbose=verbose)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    _plot_2d_projection(axes[0], embeddings['pca']['z'], labels_np, class_names,
                        f"PCA {embeddings['pca']['var_text']}", alpha, s, cmap)

    if embeddings['umap'] is not None:
        _plot_2d_projection(axes[1], embeddings['umap']['z'], labels_np, class_names, "UMAP", alpha, s, cmap)
    else:
        axes[1].text(0.5, 0.5, 'UMAP not available\nInstall: pip install umap-learn',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("UMAP")

    if embeddings['tsne'] is not None:
        _plot_2d_projection(axes[2], embeddings['tsne']['z'], labels_np, class_names, "t-SNE", alpha, s, cmap)
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
    point_index: int,
    title: str = "Noise Injection",
    n_noise_draws: int = 200,
    random_state: int = 42,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Two-panel diagnostic of latent-space noise injection at a given
    noise_scale, both panels in PCA space only: a linear projection is the
    only one where displacement in the picture is a faithful (uniformly
    scaled) picture of displacement in real z-space -- UMAP/t-SNE would
    distort the apparent size of the noise ball in ways that don't reflect
    what the model actually sees.

    Left panel: every point's clean z (blue) vs. the same points with
    noise_scale-sized isotropic Gaussian noise added (red), both projected
    through one PCA fit on the clean points.

    Right panel: one fixed point (point_index) with n_noise_draws
    independent noise draws scattered around it, annotated with:
      - sigma: the noise_scale itself
      - sigma*sqrt(d): the actual expected per-draw displacement in raw
        z-space (d = representation dimensionality). A 2D PCA projection of
        isotropic noise in D dimensions only shows ~sigma*sqrt(2) of
        spread, so the picture alone understates the true displacement --
        this number corrects for that.
      - median NN distance: median nearest-neighbor distance among this
        split's own points, computed directly in raw z-space (no GMM
        dependency, so this works even for a pre-GMM-fit checkpoint).
      - ratio: sigma*sqrt(d) / median NN distance. This is the number worth
        tracking across checkpoints -- it's dimensionless, so it stays
        comparable even though PCA is refit independently per checkpoint
        (axes can rotate/flip between epochs, so the *picture* alone is not
        a valid before/after comparison).

    Parameters
    ----------
    representations : torch.Tensor
        This split's full set of representations (N x D)
    noise_scale : float
        The scheduled noise scale at this checkpoint (0 disables noise --
        the right panel then just shows n_noise_draws copies of the same point)
    point_index : int
        Row of `representations` to use for the single-point panel. Pass the
        same index across every checkpoint for a given split so the noise
        ball is comparable over the course of training.
    n_noise_draws : int
        Number of noise draws to scatter around the single point
    """
    from sklearn.neighbors import NearestNeighbors

    z = _to_numpy(representations)
    n, d = z.shape
    rng = np.random.RandomState(random_state)

    pca, z_2d, using_cuml, var_text = _fit_pca_only(z, random_state=random_state)

    if noise_scale > 0:
        z_noised = z + rng.randn(n, d) * noise_scale
        z_noised_2d = _pca_transform(pca, z_noised, using_cuml)
    else:
        z_noised_2d = z_2d.copy()

    # Median nearest-neighbor distance in raw z-space -- no GMM dependency,
    # so this is safe even for the pre-training (epoch 0) checkpoint.
    nn = NearestNeighbors(n_neighbors=2).fit(z)
    nn_dist, _ = nn.kneighbors(z)
    median_nn_dist = float(np.median(nn_dist[:, 1]))

    sigma_raw = noise_scale * np.sqrt(d)
    ratio = sigma_raw / median_nn_dist if median_nn_dist > 0 else float('inf')

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # --- Left: population clean vs. noised ---
    axes[0].scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.4, s=12, c='tab:blue', label='clean z')
    axes[0].scatter(z_noised_2d[:, 0], z_noised_2d[:, 1], alpha=0.4, s=12, c='tab:red',
                     label=f'z + noise (σ={noise_scale:.4f})')
    axes[0].set_title(f"PCA {var_text} -- clean vs. noised")
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Right: single fixed point's noise ball ---
    point_index = int(point_index) % n
    point = z[point_index]
    if noise_scale > 0:
        draws = point[None, :] + rng.randn(n_noise_draws, d) * noise_scale
    else:
        draws = np.tile(point, (n_noise_draws, 1))
    draws_2d = _pca_transform(pca, draws, using_cuml)
    point_2d = _pca_transform(pca, point[None, :], using_cuml)[0]

    axes[1].scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.15, s=8, c='lightgray', label='all points (clean)')
    axes[1].scatter(draws_2d[:, 0], draws_2d[:, 1], alpha=0.5, s=12, c='tab:red',
                     label=f'{n_noise_draws} noise draws')
    axes[1].scatter([point_2d[0]], [point_2d[1]], s=120, c='black', marker='x', linewidths=2,
                     label=f'point #{point_index}')
    axes[1].set_title("Single-point noise ball")
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    annotation = (
        f"σ = {noise_scale:.4f}\n"
        f"σ·√d = {sigma_raw:.4f}  (true displacement scale, d={d})\n"
        f"median NN distance = {median_nn_dist:.4f}\n"
        f"ratio (σ·√d / NN dist) = {ratio:.3f}"
    )
    axes[1].text(0.02, 0.02, annotation, transform=axes[1].transAxes, fontsize=9,
                 va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

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
                legend_labels=None, ellipse_std_devs=(1, 2, 3), point_size=8, legend=True):
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
               legend=False)

    if umap_means_ranked is not None:
        all_view_umap = _Projected2DGMM(umap_means_ranked, None, weights_ranked, real_pred_rank)
        _gmm_panel(axes[1], embeddings['umap']['z'], all_view_umap, "UMAP -- Cluster Means",
                   'UMAP 1', 'UMAP 2', show_ellipses=False, legend=False)
    else:
        axes[1].text(0.5, 0.5, 'UMAP not available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("UMAP")

    if tsne_means_ranked is not None:
        all_view_tsne = _Projected2DGMM(tsne_means_ranked, None, weights_ranked, real_pred_rank)
        _gmm_panel(axes[2], embeddings['tsne']['z'], all_view_tsne, "t-SNE -- Cluster Means",
                   't-SNE 1', 't-SNE 2', show_ellipses=False, legend=False)
    else:
        axes[2].text(0.5, 0.5, 't-SNE failed', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("t-SNE")

    plt.tight_layout()
    fig.savefig(save_dir / "gmm_overview.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- cluster{rank:02d}.png: one per component, highlighted ---
    for rank in range(n_components):
        orig_idx = int(rank_order[rank])
        weight = float(weights_ranked[rank])
        member_mask = (real_pred_rank == rank).numpy()
        label_text = f"Cluster {rank + 1} (raw idx {orig_idx}, weight={weight:.4f})"

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
                       show_ellipses=show_ell, legend_labels=[label_text], ellipse_std_devs=ellipse_std_devs)

        plt.tight_layout()
        fig.savefig(save_dir / f"cluster{rank + 1:02d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

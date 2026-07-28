"""
Checkpoint persistence for DGD training runs.

Bundles the decoder state dict with the existing RepresentationLayer.save/load
and GaussianMixture.save/load (from `tgmm`) rather than inventing a new format.
"""

from pathlib import Path
from typing import Callable, Dict, Any, Optional

import torch
from tgmm import GaussianMixture

from ..models import RepresentationLayer


def save_checkpoint(
    checkpoint_dir: Path,
    decoder: torch.nn.Module,
    rep: RepresentationLayer,
    val_rep: RepresentationLayer,
    gmm: GaussianMixture,
    metadata: Dict[str, Any],
) -> None:
    """Save a full training checkpoint: decoder, train/val representations, GMM, metadata.

    The GMM is skipped (no gmm.pkl written) when `gmm.fitted_` is False —
    `GaussianMixture.load()` crashes on an unfitted model (means_ is None),
    and an unfitted GMM carries no information worth persisting anyway.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(decoder.state_dict(), checkpoint_dir / "decoder.pth")
    rep.save(str(checkpoint_dir / "train_representation.pt"))
    val_rep.save(str(checkpoint_dir / "val_representation.pt"))
    if gmm is not None and gmm.fitted_:
        gmm.save(str(checkpoint_dir / "gmm.pkl"))
    torch.save(metadata, checkpoint_dir / "metadata.pth")


def load_checkpoint(
    checkpoint_dir: Path,
    decoder_factory: Callable[[], torch.nn.Module],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint written by `save_checkpoint`.

    Parameters
    ----------
    checkpoint_dir: Directory previously written by `save_checkpoint`.
    decoder_factory: Zero-arg callable that builds a fresh, untrained decoder
        matching the architecture used at save time. `load_checkpoint` fills
        it with the saved state dict and moves it to `device`.
    device: Device to load tensors onto. Defaults to CPU.

    Returns
    -------
    dict with keys: 'decoder', 'rep', 'val_rep', 'gmm' (None if unfitted at
    save time), 'metadata'.
    """
    checkpoint_dir = Path(checkpoint_dir)
    device = device or torch.device('cpu')

    decoder = decoder_factory()
    decoder.load_state_dict(torch.load(checkpoint_dir / "decoder.pth", map_location=device))
    decoder = decoder.to(device)

    rep = RepresentationLayer.load(str(checkpoint_dir / "train_representation.pt"), device=device)
    val_rep = RepresentationLayer.load(str(checkpoint_dir / "val_representation.pt"), device=device)

    gmm_path = checkpoint_dir / "gmm.pkl"
    gmm = GaussianMixture.load(str(gmm_path), device=str(device)) if gmm_path.exists() else None

    metadata = torch.load(checkpoint_dir / "metadata.pth", map_location=device)

    return {
        'decoder': decoder,
        'rep': rep,
        'val_rep': val_rep,
        'gmm': gmm,
        'metadata': metadata,
    }

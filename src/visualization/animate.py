"""
Stitch a sequence of already-written per-epoch/per-step PNGs (from
generate_training_figures / generate_inference_figures) into a GIF and/or
MP4, in sorted order. Does not create any new figures itself -- purely a
frame-collector for whatever plot_interval-gated PNGs report.py already
wrote to disk.
"""

import re
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import imageio.v2 as imageio
from PIL import Image

PathLike = Union[str, Path]


def _sort_key(path: Path) -> int:
    """Sort by the last integer found in the filename (epoch/step number)."""
    matches = re.findall(r'\d+', path.stem)
    return int(matches[-1]) if matches else 0


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _read_and_pad(image_paths: Sequence[PathLike], macro_block_size: int = 16) -> List[np.ndarray]:
    """Load each PNG and pad (white, top-left aligned) to the max width/height
    across all of them. Consecutive matplotlib savefig(bbox_inches='tight')
    calls don't guarantee identical pixel dimensions -- legend/tick-label
    content varies slightly frame to frame -- but every writer below needs a
    uniform frame size.

    The padded size is also rounded up to the nearest multiple of
    macro_block_size, matching imageio-ffmpeg's own default -- libx264
    requires each dimension divisible by it, and without this the mp4 writer
    pads (correctly) but prints a warning about doing so every time. Padding
    to the right size ourselves means it never has to. GIF frames get the
    same few extra white pixels; not visible."""
    images = [Image.open(str(p)).convert('RGB') for p in image_paths]
    max_w = _round_up(max(im.width for im in images), macro_block_size)
    max_h = _round_up(max(im.height for im in images), macro_block_size)
    frames = []
    for im in images:
        if im.size != (max_w, max_h):
            canvas = Image.new('RGB', (max_w, max_h), color='white')
            canvas.paste(im, (0, 0))
            im = canvas
        frames.append(np.asarray(im))
    return frames


def make_animation(
    image_paths: Sequence[PathLike],
    out_path: PathLike,
    fps: int = 2,
    formats: Sequence[str] = ('gif', 'mp4'),
) -> List[Path]:
    """
    Stitch image_paths (already in the order frames should play) into an
    animation. Written to out_path with its suffix replaced per format, e.g.
    out_path='foo.png' with formats=('gif','mp4') writes foo.gif and foo.mp4.

    Parameters
    ----------
    image_paths : PNG paths, in playback order
    out_path : base output path (suffix is replaced per format)
    fps : frames per second
    formats : 'gif', 'mp4', or both

    Returns
    -------
    List[Path] of files actually written, in `formats` order.
    """
    out_path = Path(out_path)
    frames = _read_and_pad(image_paths)

    written = []
    for fmt in formats:
        dest = out_path.with_suffix(f'.{fmt}')
        if fmt == 'gif':
            imageio.mimsave(str(dest), frames, fps=fps)
        elif fmt == 'mp4':
            # Frames are already macro_block_size-aligned by _read_and_pad,
            # so libx264 (which requires even/16-aligned dimensions) never
            # needs to pad further. ffmpeg_log_level='error' additionally
            # drops a handful of harmless informational notices ffmpeg's own
            # rawvideo demuxer prints for short clips (e.g. "not enough
            # frames to estimate rate") -- these are about ffmpeg's internal
            # stream probing, not about anything wrong with the output, and
            # aren't controllable from our end other than by log level.
            imageio.mimsave(str(dest), frames, fps=fps, ffmpeg_log_level='error')
        else:
            raise ValueError(f"Unsupported format: {fmt!r} (expected 'gif' or 'mp4')")
        written.append(dest)
    return written


def animate_from_pattern(
    directory: PathLike,
    pattern: str,
    out_path: PathLike,
    fps: int = 2,
    formats: Sequence[str] = ('gif', 'mp4'),
) -> List[Path]:
    """
    Glob `pattern` under `directory`, sort by the trailing number in each
    filename (epoch/step), and stitch into an animation via make_animation.
    Returns [] without writing anything if fewer than 2 files match --
    nothing to animate (e.g. plot_interval was never set for this run).
    """
    directory = Path(directory)
    paths = sorted(directory.glob(pattern), key=_sort_key)
    if len(paths) < 2:
        return []
    return make_animation(paths, out_path, fps=fps, formats=formats)

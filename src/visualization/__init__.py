import seaborn as sns

# Applied once, at import time, so every figure produced through this package
# -- whichever submodule builds it -- picks up the same look without each
# plotting function setting its own style. Image grids are unaffected: every
# imshow axes in image.py calls ax.axis('off'), which strips grid/spines/ticks
# regardless of the active theme. Explicit colors/cmaps passed by individual
# plot calls (tab10, viridis, the pastel GMM palette, ...) aren't touched by
# this -- set_theme only changes background/grid/font defaults, not color choices.
sns.set_theme(context='notebook', style='whitegrid', font_scale=1.0)

from .image import (
    organize_by_class,
    plot_image_grid,
    plot_images_by_class,
    plot_generated_samples,
    plot_ground_truth_and_reconstructions_by_class,
)

from .latent import (
    plot_latent_space,
    generate_latent_space_figures,
    plot_noise_comparison,
    plot_representation_trajectory,
    plot_noise_optimization_shift,
)

from .loss import (
    plot_training_analysis,
    plot_training_dynamics
)

from .report import generate_training_figures, generate_inference_figures

from .animate import make_animation, animate_from_pattern


__all__ = [
    # Image visualization functions
    'organize_by_class',
    'plot_image_grid',
    'plot_images_by_class',
    'plot_generated_samples',
    'plot_ground_truth_and_reconstructions_by_class',
    # Latent space visualization functions
    'plot_latent_space',
    'generate_latent_space_figures',
    'plot_noise_comparison',
    'plot_representation_trajectory',
    'plot_noise_optimization_shift',
    # Loss and dynamics visualization functions
    'plot_training_analysis',
    'plot_training_dynamics',
    # Post-hoc report generation
    'generate_training_figures',
    'generate_inference_figures',
    # GIF/MP4 animation from already-written per-epoch/per-step PNGs
    'make_animation',
    'animate_from_pattern',
]

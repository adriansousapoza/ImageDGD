"""
Shared schedule math used by both live training/inference loops and
post-hoc figure regeneration, so the two never drift apart.
"""

import math


def cosine_noise_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """
    Latent-space noise scale at `step` (1-indexed), cosine-annealed from
    `start` (step 1) to `end` (step `total_steps`).

    Progress is clamped to [0, 1] so step 0 (e.g. the pre-training
    checkpoint) returns exactly `start` instead of a value past it from a
    negative progress fraction.
    """
    progress = (step - 1) / max(total_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * progress))


def constant_noise_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """
    Flat noise target for the whole run, at `start`. `end` and `step` are
    accepted (unused) so this has the same signature as the other
    schedules and can be swapped in via noise_schedule() without special-casing.
    """
    return start


def linear_noise_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """
    Latent-space noise scale at `step` (1-indexed), linearly interpolated
    from `start` (step 1) to `end` (step `total_steps`). Same clamping
    convention as cosine_noise_schedule.
    """
    progress = (step - 1) / max(total_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def warmup_cosine_noise_schedule(
    step: int, total_steps: int, start: float, end: float, warmup_steps: int = 0
) -> float:
    """
    Zero noise for the first `warmup_steps` steps (representations get a
    noise-free head start, mirroring how first_epoch_gmm delays the GMM
    prior), then a cosine anneal from `start` to `end` over the remaining
    `total_steps - warmup_steps` steps.
    """
    if warmup_steps <= 0:
        return cosine_noise_schedule(step, total_steps, start, end)
    if step <= warmup_steps:
        return 0.0
    return cosine_noise_schedule(step - warmup_steps, total_steps - warmup_steps, start, end)


_NOISE_SCHEDULES = {
    'constant': constant_noise_schedule,
    'linear': linear_noise_schedule,
    'cosine': cosine_noise_schedule,
    'warmup_cosine': warmup_cosine_noise_schedule,
}


def noise_schedule(
    step: int, total_steps: int, start: float, end: float,
    schedule_type: str = 'cosine', warmup_steps: int = 0,
) -> float:
    """
    Dispatch to the named noise schedule shape (see latent_noise_schedule in
    config/config.yaml for the options and what each looks like). This is
    the single entry point trainer.py and every post-hoc figure/notebook
    that needs to reconstruct a checkpoint's noise level should call,
    instead of calling a specific *_noise_schedule function directly, so
    they can never drift onto different shapes for the same run.

    warmup_steps is only meaningful for schedule_type='warmup_cosine' -- it's
    accepted (and ignored) for the others so every call site can pass it
    unconditionally.
    """
    try:
        fn = _NOISE_SCHEDULES[schedule_type]
    except KeyError:
        raise ValueError(
            f"Unknown latent_noise_schedule {schedule_type!r}; choose one of {sorted(_NOISE_SCHEDULES)}"
        )
    if fn is warmup_cosine_noise_schedule:
        return fn(step, total_steps, start, end, warmup_steps=warmup_steps)
    return fn(step, total_steps, start, end)


def noise_std_for_dim(noise_scale: float, dim: int) -> float:
    """
    Convert a dimension-independent *target displacement* (`noise_scale`,
    the noise_schedule output) into the actual per-coordinate std to draw
    isotropic Gaussian noise from, so the realized displacement stays the
    same magnitude regardless of latent dimension `dim`.

    For eps ~ N(0, sigma^2 I_d), each coordinate independently contributes
    sigma^2 to E[||eps||^2] (variances add), so E[||eps||^2] = d * sigma^2
    exactly -- this is just ||eps||^2/sigma^2 ~ chi-squared(d), whose mean is
    exactly d for every d, with no Gamma-function approximation needed (unlike
    E[||eps||] itself, the chi-distribution's mean, which only has a closed
    form via Gamma functions and needs a large-d approximation, sqrt(d), to
    be usable). So sigma = noise_scale / sqrt(d) makes RMS(||eps||) =
    noise_scale exactly, for every d -- including small d like 1 or 2, not
    just "large enough" d.

    Without this correction, drawing each coordinate as N(0, noise_scale^2)
    directly makes the realized displacement grow as noise_scale*sqrt(d) --
    e.g. sqrt(2)x bigger at dim=16 than at dim=8 for the identical config
    value -- so the *same* noise_scale has a dimension-dependent effect:
    changing representation.n_features silently changes what
    latent_noise_start/end mean, and unrelated latent-dimension experiments
    stop being comparable.

    dim=0 (no representation, degenerate) returns noise_scale unchanged --
    there's nothing to correct for.
    """
    if dim <= 0:
        return noise_scale
    return noise_scale / math.sqrt(dim)

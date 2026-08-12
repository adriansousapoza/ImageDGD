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

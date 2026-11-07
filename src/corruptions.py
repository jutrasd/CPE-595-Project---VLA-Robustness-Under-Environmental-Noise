"""Camera image corruptions (M1, owner: William). See the contract in CLAUDE.md."""

import numpy as np


def corrupt(frame: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Return a degraded copy of an HxWx3 uint8 RGB frame.

    kind: "none" | "motion_blur" | "defocus_blur" | "low_light" | "fog"
    severity: 0..5. Severity 0 (or kind "none") MUST return the input unchanged.
    Output has the same shape and dtype as the input.
    """
    raise NotImplementedError("M1")

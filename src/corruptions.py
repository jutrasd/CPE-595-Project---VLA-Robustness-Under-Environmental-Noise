"""Camera image corruptions (M1). See the contract in CLAUDE.md.

Parameters are in pixels for the 256x256 frames both SmolVLA checkpoints use.
"""

import copy

import cv2
import numpy as np

# Index is severity. Values follow docs/MVP_PLAN.md except low_light, which was tuned because the
# plan's values were clearly visible at severity 1 and nearly blank at 5 (see docs/SETUP_NOTES.md).
# Check any change with scripts/frame_grid.py.
MOTION_BLUR_LENGTH = (0, 3, 5, 9, 13, 19)
DEFOCUS_BLUR_RADIUS = (0, 1, 2, 3, 5, 7)
LOW_LIGHT_BRIGHTNESS = (1.0, 0.85, 0.65, 0.45, 0.30, 0.20)
LOW_LIGHT_GAMMA = (1.0, 1.04, 1.08, 1.12, 1.16, 1.20)
LOW_LIGHT_NOISE_SIGMA = (0, 1, 2, 3, 4, 6)
FOG_TRANSMISSION = (1.0, 0.85, 0.70, 0.55, 0.40, 0.25)
FOG_HAZE = 200

KINDS = ("none", "motion_blur", "defocus_blur", "low_light", "fog")
MAX_SEVERITY = 5


def motion_blur(frame: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Blur along a straight line whose angle comes from rng.

    The angle is drawn from a copy of rng, leaving rng untouched. Reusing one rng for a whole
    episode therefore gives every frame the same angle, which the contract's signature could
    not otherwise express. Use a new rng per episode to get a new angle.
    """
    angle = np.deg2rad(copy.deepcopy(rng).uniform(0.0, 180.0))
    length = MOTION_BLUR_LENGTH[severity]
    center = (length - 1) / 2
    dx, dy = center * np.cos(angle), center * np.sin(angle)
    kernel = np.zeros((length, length), dtype=np.uint8)
    start = (round(center - dx), round(center - dy))
    end = (round(center + dx), round(center + dy))
    cv2.line(kernel, start, end, 1)
    return _convolve(frame, kernel)


def defocus_blur(frame: np.ndarray, severity: int) -> np.ndarray:
    """Blur with a disk kernel, the shape an out-of-focus lens produces."""
    radius = DEFOCUS_BLUR_RADIUS[severity]
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return _convolve(frame, x * x + y * y <= radius * radius)


def low_light(frame: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Darken toward black, then add noise, because sensor noise shows up after the light loss."""
    dark = (frame / 255.0 * LOW_LIGHT_BRIGHTNESS[severity]) ** LOW_LIGHT_GAMMA[severity] * 255.0
    noise = rng.normal(0.0, LOW_LIGHT_NOISE_SIGMA[severity], frame.shape)
    return _to_uint8(dark + noise)


def fog(frame: np.ndarray, severity: int) -> np.ndarray:
    """Blend toward uniform haze. This is a simplification: real fog thickens with depth."""
    transmission = FOG_TRANSMISSION[severity]
    return _to_uint8(frame * transmission + FOG_HAZE * (1.0 - transmission))


def corrupt(frame: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Return a degraded copy of an HxWx3 uint8 RGB frame.

    kind: "none" | "motion_blur" | "defocus_blur" | "low_light" | "fog"
    severity: 0..5. Severity 0 (or kind "none") MUST return the input unchanged.
    Output has the same shape and dtype as the input.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown corruption {kind!r}. Expected one of {KINDS}.")
    if not 0 <= severity <= MAX_SEVERITY:
        raise ValueError(f"Severity must be 0..{MAX_SEVERITY}, got {severity}.")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 uint8 frame, got {frame.shape} {frame.dtype}.")

    if kind == "none" or severity == 0:
        return frame.copy()
    if kind == "motion_blur":
        return motion_blur(frame, severity, rng)
    if kind == "defocus_blur":
        return defocus_blur(frame, severity)
    if kind == "low_light":
        return low_light(frame, severity, rng)
    return fog(frame, severity)


def _convolve(frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve with a kernel normalized to sum to 1, so overall brightness is preserved."""
    kernel = kernel.astype(np.float32)
    return cv2.filter2D(frame, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REPLICATE)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Round and clip a float image to uint8."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)

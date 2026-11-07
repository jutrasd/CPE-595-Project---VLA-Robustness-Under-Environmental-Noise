"""Tests for src/corruptions.py (M1)."""

import numpy as np
import pytest

from src.corruptions import KINDS, MAX_SEVERITY, corrupt

CORRUPTIONS = [kind for kind in KINDS if kind != "none"]
SEVERITIES = range(1, MAX_SEVERITY + 1)


@pytest.fixture
def frame() -> np.ndarray:
    """Return a textured frame, so blurs have edges to change."""
    return np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)


def rng(seed: int = 1) -> np.random.Generator:
    """Return a seeded generator."""
    return np.random.default_rng(seed)


@pytest.mark.parametrize("kind", KINDS)
def test_severity_zero_returns_identical_array(frame, kind):
    assert np.array_equal(corrupt(frame, kind, 0, rng()), frame)


@pytest.mark.parametrize("severity", range(MAX_SEVERITY + 1))
def test_kind_none_returns_identical_array(frame, severity):
    assert np.array_equal(corrupt(frame, "none", severity, rng()), frame)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("severity", range(MAX_SEVERITY + 1))
def test_shape_and_dtype_preserved(frame, kind, severity):
    out = corrupt(frame, kind, severity, rng())
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("severity", range(MAX_SEVERITY + 1))
def test_same_seed_gives_same_output(frame, kind, severity):
    assert np.array_equal(corrupt(frame, kind, severity, rng()), corrupt(frame, kind, severity, rng()))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("severity", range(MAX_SEVERITY + 1))
def test_input_is_not_modified_or_aliased(frame, kind, severity):
    original = frame.copy()
    out = corrupt(frame, kind, severity, rng())
    assert np.array_equal(frame, original)
    assert not np.shares_memory(out, frame)


@pytest.mark.parametrize("kind", CORRUPTIONS)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_positive_severity_changes_the_frame(frame, kind, severity):
    assert not np.array_equal(corrupt(frame, kind, severity, rng()), frame)


def test_low_light_brightness_decreases_with_severity(frame):
    means = [corrupt(frame, "low_light", s, rng()).mean() for s in range(MAX_SEVERITY + 1)]
    assert all(darker < brighter for brighter, darker in zip(means, means[1:]))


@pytest.mark.parametrize("kind", ["motion_blur", "defocus_blur"])
def test_blur_reduces_detail_with_severity_and_keeps_brightness(frame, kind):
    outs = [corrupt(frame, kind, s, rng()) for s in range(MAX_SEVERITY + 1)]
    detail = [out.std() for out in outs]
    assert all(less < more for more, less in zip(detail, detail[1:]))
    assert all(abs(out.mean() - frame.mean()) < 1.0 for out in outs)


def test_fog_moves_toward_haze_with_severity(frame):
    distances = [
        np.abs(corrupt(frame, "fog", s, rng()).astype(float) - 200).mean()
        for s in range(MAX_SEVERITY + 1)
    ]
    assert all(closer < farther for farther, closer in zip(distances, distances[1:]))


def test_motion_blur_angle_is_constant_within_an_episode(frame):
    episode_rng = rng()
    first = corrupt(frame, "motion_blur", 5, episode_rng)
    second = corrupt(frame, "motion_blur", 5, episode_rng)
    assert np.array_equal(first, second)


def test_motion_blur_angle_varies_between_episodes(frame):
    outs = [corrupt(frame, "motion_blur", 5, rng(seed)) for seed in range(5)]
    assert any(not np.array_equal(outs[0], out) for out in outs[1:])


def test_low_light_noise_varies_between_frames(frame):
    episode_rng = rng()
    first = corrupt(frame, "low_light", 3, episode_rng)
    second = corrupt(frame, "low_light", 3, episode_rng)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("kind", "severity"), [("snow", 1), ("fog", -1), ("fog", MAX_SEVERITY + 1)]
)
def test_invalid_arguments_raise(frame, kind, severity):
    with pytest.raises(ValueError):
        corrupt(frame, kind, severity, rng())


def test_non_uint8_frame_raises(frame):
    with pytest.raises(ValueError):
        corrupt(frame.astype(np.float32), "fog", 1, rng())

"""Tests for src/stats.py (M3)."""

import pandas as pd
import pytest

from src.stats import summarize, wilson_interval


def test_wilson_matches_a_known_value():
    # 9/10, the example in LeRobot's docs: roughly 60% to 98%.
    low, high = wilson_interval(9, 10)
    assert low == pytest.approx(0.596, abs=0.001)
    assert high == pytest.approx(0.982, abs=0.001)


def test_wilson_at_zero_successes_is_not_zero_width():
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert high == pytest.approx(0.161, abs=0.001)


def test_wilson_at_all_successes_is_not_zero_width():
    low, high = wilson_interval(20, 20)
    assert low == pytest.approx(0.839, abs=0.001)
    assert high == 1.0


def test_wilson_narrows_with_more_episodes():
    low_small, high_small = wilson_interval(10, 20)
    low_large, high_large = wilson_interval(100, 200)
    assert high_large - low_large < high_small - low_small


@pytest.mark.parametrize(("successes", "n"), [(1, 0), (-1, 10), (11, 10)])
def test_wilson_rejects_invalid_counts(successes, n):
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


def test_summarize_groups_by_setting():
    episodes = pd.DataFrame(
        {
            "policy": ["smolvla"] * 4,
            "corruption": ["fog", "fog", "fog", "low_light"],
            "severity": [1, 1, 3, 1],
            "camera": ["both"] * 4,
            "success": [1, 0, 1, 0],
        }
    )
    summary = summarize(episodes).set_index(["corruption", "severity"])
    assert len(summary) == 3
    assert summary.loc[("fog", 1), "n"] == 2
    assert summary.loc[("fog", 1), "success_rate"] == 0.5
    assert summary.loc[("low_light", 1), "successes"] == 0

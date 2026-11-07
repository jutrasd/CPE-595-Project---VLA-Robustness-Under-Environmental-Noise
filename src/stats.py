"""Success rates and Wilson 95% confidence intervals per setting (M3)."""

import math

import pandas as pd

Z_95 = 1.959964
GROUP_COLUMNS = ["policy", "corruption", "severity", "camera"]


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion, as fractions.

    Unlike the normal approximation, it stays inside [0, 1] and is not zero-width at 0 or n
    successes, which are common outcomes with 20 episodes per setting.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in 0..{n}, got {successes}.")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    # Exact bounds at the extremes, since rounding can leave them a hair inside [0, 1].
    low = 0.0 if successes == 0 else center - margin
    high = 1.0 if successes == n else center + margin
    return low, high


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    """Return n, successes, success_rate, ci_low and ci_high per setting."""
    rows = []
    for key, group in episodes.groupby(GROUP_COLUMNS, sort=True):
        n, successes = len(group), int(group["success"].sum())
        low, high = wilson_interval(successes, n)
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, key)),
                "n": n,
                "successes": successes,
                "success_rate": successes / n,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)

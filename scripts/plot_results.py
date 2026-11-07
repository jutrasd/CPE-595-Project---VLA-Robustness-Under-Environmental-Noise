"""Plot degradation curves from episodes.csv and write the summary CSV (M3)."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Runs headless.
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Running a script puts scripts/ on the path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import summarize  # noqa: E402

# Fixed per corruption, so a corruption keeps its color and marker in every plot.
STYLES = {
    "motion_blur": ("#2a78d6", "o", "Motion blur"),
    "defocus_blur": ("#eb6834", "s", "Defocus blur"),
    "low_light": ("#1baf7a", "^", "Low light"),
    "fog": ("#eda100", "D", "Fog"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANEL_WIDTH = 3.3


def plot(summary: pd.DataFrame, clean: pd.Series, output: Path, n: int) -> None:
    """Draw one panel per corruption, since four overlapping CI bands can't be read on one plot."""
    fig, axes = plt.subplots(1, len(STYLES), figsize=(PANEL_WIDTH * len(STYLES), 3.8), sharey=True)
    for ax, (corruption, (color, marker, label)) in zip(axes, STYLES.items()):
        rows = summary[summary["corruption"] == corruption].sort_values("severity")
        x = [0, *rows["severity"]]
        y = [clean["success_rate"], *rows["success_rate"]]
        low = [clean["ci_low"], *rows["ci_low"]]
        high = [clean["ci_high"], *rows["ci_high"]]
        ax.fill_between(x, [v * 100 for v in low], [v * 100 for v in high], color=color, alpha=0.18, lw=0)
        ax.plot(x, [v * 100 for v in y], color=color, marker=marker, markersize=7, lw=2, clip_on=False, zorder=3)
        ax.axhline(clean["success_rate"] * 100 / 2, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_title(label, color=INK, fontsize=12, loc="left")
        ax.set_xlabel("Severity", color=MUTED)
        ax.set_xticks(range(6))
        ax.set_xlim(-0.2, 5.2)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Task success (%)", color=MUTED)
    axes[0].text(5.2, clean["success_rate"] * 100 / 2 + 2, "half of clean", color=MUTED, fontsize=8, ha="right")
    fig.suptitle(
        f"SmolVLA success vs. camera corruption ({n} episodes per point; bands are 95% Wilson CIs)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    print(f"Wrote {output}")


def main() -> None:
    """Read the sweep and baseline, write the summary, plot it, and print the numbers."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, default=Path("results/mvp_sweep/episodes.csv"))
    p.add_argument("--baseline", type=Path, default=Path("results/clean_baseline/episodes.csv"))
    p.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    p.add_argument("--output", type=Path, default=Path("results/plots/smolvla_degradation.png"))
    args = p.parse_args()

    sweep, baseline = pd.read_csv(args.sweep), pd.read_csv(args.baseline)
    # Severity 0 is only comparable if the baseline ran the same episodes.
    pair = ["task_id", "init_state", "seed"]
    if set(map(tuple, sweep[pair].values)) != set(map(tuple, baseline[pair].values)):
        sys.exit("The sweep and baseline did not run the same task/init-state/seed episodes.")

    summary = summarize(pd.concat([baseline, sweep]))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False, float_format="%.4f")
    print(f"Wrote {args.summary}")

    clean = summary[summary["corruption"] == "none"].iloc[0]
    plot(summary[summary["corruption"] != "none"], clean, args.output, int(clean["n"]))
    print(
        summary[["corruption", "severity", "n", "successes", "success_rate", "ci_low", "ci_high"]]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

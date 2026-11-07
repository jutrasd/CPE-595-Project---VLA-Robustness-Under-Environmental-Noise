"""CLI wrapper around src/evaluate.py: run a sweep from a YAML file (M2)."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# Running a script puts scripts/ on the path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import RunConfig, expand_settings, run_sweep  # noqa: E402

# Memory grows about 0.12 GB per episode (cause not traced), and at episode 36 that got the first
# M3 sweep OOM-killed along with VS Code. Results are saved per episode and every episode seeds
# itself, so running in short-lived worker processes frees the memory without changing results.
EPISODES_PER_WORKER = 20
EXIT_MORE_TO_DO = 75


def run_workers() -> None:
    """Rerun this script as a worker process until the sweep is complete or a worker fails."""
    while True:
        command = [sys.executable, *sys.argv, "--max-episodes", str(EPISODES_PER_WORKER)]
        code = subprocess.run(command).returncode
        if code != EXIT_MORE_TO_DO:
            sys.exit(code)


def main() -> None:
    """Read the sweep's YAML, apply any overrides, and run it."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--sweep-name", help="override the YAML's sweep_name")
    p.add_argument("--suite", help="override the YAML's suite")
    p.add_argument("--checkpoint", help="override the YAML's checkpoint")
    p.add_argument("--task-ids", type=int, nargs="+", help="override the YAML's task_ids")
    p.add_argument("--init-states", type=int, nargs="+", help="override the YAML's init_states")
    p.add_argument("--max-episodes", type=int, help="be a worker: stop after this many episodes")
    args = p.parse_args()
    if args.max_episodes is None:
        run_workers()

    cfg = yaml.safe_load(args.config.read_text())
    run_config = RunConfig(
        checkpoint=args.checkpoint or cfg["checkpoint"],
        suite=args.suite or cfg["suite"],
        task_ids=tuple(args.task_ids or cfg["task_ids"]),
        init_states=tuple(args.init_states or cfg["init_states"]),
        seed=cfg["seed"],
        n_action_steps=cfg["n_action_steps"],
    )
    settings = expand_settings(cfg["corruptions"], cfg["severities"], cfg["cameras"])
    out_dir = Path("results") / (args.sweep_name or cfg["sweep_name"])
    if not run_sweep(run_config, settings, out_dir, args.max_episodes):
        sys.exit(EXIT_MORE_TO_DO)


if __name__ == "__main__":
    main()

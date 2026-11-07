"""Tests for the parts of src/evaluate.py that need no simulator (M2)."""

import pytest

from src.evaluate import (
    COLUMNS,
    RunConfig,
    Setting,
    append_row,
    check_run_config,
    episode_seed,
    expand_settings,
    read_completed,
)


def make_row(task_id=0, severity=1, success=1) -> dict:
    """Return a complete CSV row."""
    return {
        "policy": "smolvla",
        "checkpoint": "ckpt",
        "suite": "libero_object",
        "task_id": task_id,
        "init_state": 0,
        "seed": 1000,
        "corruption": "fog",
        "severity": severity,
        "camera": "both",
        "success": success,
        "steps": 50,
        "video_path": "",
        "timestamp": "2026-11-07T00:00:00+00:00",
    }


def make_config(n_action_steps=10) -> RunConfig:
    """Return a small run config."""
    return RunConfig("ckpt", "libero_object", (0,), (0,), 1000, n_action_steps)


def test_expand_settings_covers_every_combination():
    settings = expand_settings(["fog", "low_light"], [1, 3], ["both"])
    assert len(settings) == 4
    assert Setting("low_light", 3, "both") in settings


def test_expand_settings_rejects_unknown_camera():
    with pytest.raises(ValueError):
        expand_settings(["fog"], [1], ["front"])


def test_seed_varies_by_episode_and_never_collides():
    seeds = {episode_seed(1000, task, init) for task in range(10) for init in range(50)}
    assert len(seeds) == 500


def test_missing_csv_means_nothing_completed(tmp_path):
    assert read_completed(tmp_path / "episodes.csv") == set()


def test_resume_finds_written_rows_regardless_of_outcome(tmp_path):
    path = tmp_path / "episodes.csv"
    append_row(path, make_row(task_id=0, success=1))
    append_row(path, make_row(task_id=1, success=0))
    completed = read_completed(path)
    assert len(completed) == 2
    # Keys come back as strings from the CSV, so a new row's key must match them.
    assert tuple(str(make_row(task_id=1, success=1)[c]) for c in COLUMNS[:9]) in completed


def test_header_is_written_once(tmp_path):
    path = tmp_path / "episodes.csv"
    append_row(path, make_row(task_id=0))
    append_row(path, make_row(task_id=1))
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(COLUMNS)
    assert len(lines) == 3


def test_resuming_with_the_same_config_is_allowed(tmp_path):
    check_run_config(tmp_path, make_config())
    check_run_config(tmp_path, make_config())


def test_resuming_with_a_different_config_is_refused(tmp_path):
    check_run_config(tmp_path, make_config(n_action_steps=10))
    with pytest.raises(SystemExit):
        check_run_config(tmp_path, make_config(n_action_steps=1))

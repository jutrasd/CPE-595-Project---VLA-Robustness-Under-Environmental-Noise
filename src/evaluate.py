"""Run episodes, append one CSV row per episode, and resume from the CSV (M2)."""

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

from src.corruptions import corrupt

# Must be set before anything imports MuJoCo. Respect an explicit override (e.g. osmesa).
os.environ.setdefault("MUJOCO_GL", "egl")

COLUMNS = [
    "policy",
    "checkpoint",
    "suite",
    "task_id",
    "init_state",
    "seed",
    "corruption",
    "severity",
    "camera",
    "success",
    "steps",
    "video_path",
    "timestamp",
]
# Everything that identifies an episode. success, steps, video_path and timestamp are outcomes.
KEY_COLUMNS = COLUMNS[:9]
CAMERAS = ("both", "agentview", "wrist")


@dataclass(frozen=True)
class Setting:
    """One corruption condition."""

    corruption: str
    severity: int
    camera: str


@dataclass(frozen=True)
class RunConfig:
    """What a sweep runs. Episodes in one CSV must not mix these."""

    checkpoint: str
    suite: str
    task_ids: tuple[int, ...]
    init_states: tuple[int, ...]
    seed: int
    n_action_steps: int
    image_size: int = 256
    device: str = "cuda"


def ensure_libero_config() -> None:
    """Create LIBERO's config file, because importing LIBERO without one blocks on input()."""
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    if not (config_dir / "config.yaml").exists():
        subprocess.run(
            [sys.executable, "-c", "import libero.libero"], input="N\n", text=True, check=True
        )


def pick_egl_device() -> None:
    """Point robosuite at the first EGL device that initializes.

    robosuite only tries device 0, which fails on this laptop (NVIDIA, under PRIME on-demand).
    Device indexes differ by machine, so probe instead of hardcoding one.
    """
    if os.environ["MUJOCO_GL"] != "egl" or "MUJOCO_EGL_DEVICE_ID" in os.environ:
        return
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mujoco.egl import egl_ext as EGL
    from OpenGL import error

    for index, device in enumerate(EGL.eglQueryDevicesEXT()):
        display = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
        try:
            initialized = EGL.eglInitialize(display, None, None)
        except error.GLError:
            continue
        if initialized == EGL.EGL_TRUE:
            EGL.eglTerminate(display)
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(index)
            print(f"EGL device for rendering: {index}")
            return
    sys.exit("No EGL device initialized. Try MUJOCO_GL=osmesa.")


def expand_settings(corruptions: list[str], severities: list[int], cameras: list[str]) -> list[Setting]:
    """Return every corruption x severity x camera combination."""
    for camera in cameras:
        if camera not in CAMERAS:
            raise ValueError(f"Unknown camera {camera!r}. Expected one of {CAMERAS}.")
    return [Setting(c, s, cam) for c, s, cam in product(corruptions, severities, cameras)]


def episode_seed(base_seed: int, task_id: int, init_state: int) -> int:
    """Return a seed that varies by episode but not by condition, so comparisons stay paired."""
    return base_seed + 1000 * task_id + init_state


def _key(row: dict) -> tuple[str, ...]:
    """Return an episode's identity as strings, so CSV rows and new rows compare equal."""
    return tuple(str(row[column]) for column in KEY_COLUMNS)


def read_completed(csv_path: Path) -> set[tuple[str, ...]]:
    """Return the keys of episodes already in the CSV. A missing file means none."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as f:
        return {_key(row) for row in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict) -> None:
    """Append one row and force it to disk, so a crash loses at most the current episode."""
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def check_run_config(out_dir: Path, config: RunConfig) -> None:
    """Record the config, and refuse to resume with a different one.

    The CSV contract has no column for settings like n_action_steps, so without this a resumed
    sweep could silently mix them.
    """
    path = out_dir / "run_config.json"
    current = json.loads(json.dumps(asdict(config)))
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != current:
            sys.exit(f"{path} differs from this run.\nSaved: {saved}\nNow:   {current}")
    else:
        path.write_text(json.dumps(current, indent=2) + "\n")


def _load(config: RunConfig):
    """Build the policy, processors and per-task environments. Imports LeRobot late on purpose."""
    ensure_libero_config()
    pick_egl_device()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_policy, make_pre_post_processors

    if config.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available. Check `uname -r` and docs/SETUP_NOTES.md.")

    policy_cfg = PreTrainedConfig.from_pretrained(config.checkpoint)
    policy_cfg.pretrained_path = config.checkpoint
    policy_cfg.device = config.device
    policy_cfg.n_action_steps = config.n_action_steps

    # Checkpoints name their cameras differently. Input order is agentview, then wrist.
    cameras = [k.split(".images.")[1] for k in policy_cfg.input_features if ".images." in k]
    env_cfg = LiberoEnv(
        task=config.suite,
        task_ids=list(config.task_ids),
        observation_height=config.image_size,
        observation_width=config.image_size,
        camera_name_mapping={"agentview_image": cameras[0], "robot0_eye_in_hand_image": cameras[1]},
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)[config.suite]
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=config.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": config.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg)
    return policy_cfg, env_cfg, envs, policy, pre, post, env_pre, env_post, cameras


def run_episode(env, policy, processors, cameras, setting, init_state, seed, record):
    """Run one episode and return (success, steps, frames). Frames are empty unless record is True."""
    import torch
    from lerobot.envs import preprocess_observation
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.utils.constants import ACTION
    from lerobot.utils.random_utils import set_seed

    pre, post, env_pre, env_post = processors
    corrupted = {"both": cameras, "agentview": cameras[:1], "wrist": cameras[1:]}[setting.camera]
    rng = np.random.default_rng(seed)
    set_seed(seed)

    # reset() advances init_state_id, so set it before every episode.
    env.envs[0].unwrapped.init_state_id = init_state
    policy.reset()
    obs, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    task = env.call("task_description")[0]
    max_steps = env.call("_max_episode_steps")[0]

    frames, success, step = [], False, 0
    while step < max_steps:
        # Only the policy's copy is corrupted: the simulator's state is unaffected by this.
        for camera in corrupted:
            obs["pixels"][camera] = corrupt(
                obs["pixels"][camera][0], setting.corruption, setting.severity, rng
            )[None]
        if record:
            # Rotate each view as the policy sees it. Flipping the joined image would swap panels.
            frames.append(np.hstack([obs["pixels"][c][0][::-1, ::-1] for c in cameras]))

        batch = preprocess_observation(obs)
        batch["task"] = [task]
        batch = pre(env_pre(batch))
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = env_post({ACTION: post(action)})[ACTION]
        obs, _, terminated, truncated, info = env.step(action.to("cpu").numpy())
        step += 1

        final = info.get("final_info")
        source = final if isinstance(final, dict) else info
        success = bool(source.get("is_success", [False])[0])
        if terminated[0] or truncated[0]:
            break
    return success, step, frames


def run_sweep(
    config: RunConfig, settings: list[Setting], out_dir: Path, max_episodes: int | None = None
) -> bool:
    """Run every setting x task x init state, skipping episodes already in the CSV.

    Stops after max_episodes new episodes, if given. Returns True once the sweep is complete.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    check_run_config(out_dir, config)
    csv_path = out_dir / "episodes.csv"
    completed = read_completed(csv_path)

    policy_cfg, env_cfg, envs, policy, pre, post, env_pre, env_post, cameras = _load(config)
    import imageio.v2 as imageio

    processors = (pre, post, env_pre, env_post)
    total = len(settings) * len(config.task_ids) * len(config.init_states)
    print(f"{len(completed)} episodes already in {csv_path}; sweep has {total}.")
    done_now, start = 0, time.perf_counter()

    for setting, task_id in product(settings, config.task_ids):
        env = envs[task_id]
        for index, init_state in enumerate(config.init_states):
            seed = episode_seed(config.seed, task_id, init_state)
            row = {
                "policy": policy_cfg.type,
                "checkpoint": config.checkpoint,
                "suite": config.suite,
                "task_id": task_id,
                "init_state": init_state,
                "seed": seed,
                "corruption": setting.corruption,
                "severity": setting.severity,
                "camera": setting.camera,
            }
            if _key(row) in completed:
                continue

            # The first episode of a setting, on the first task.
            record = task_id == config.task_ids[0] and index == 0
            success, steps, frames = run_episode(
                env, policy, processors, cameras, setting, init_state, seed, record
            )
            video_path = ""
            if frames:
                name = f"{setting.corruption}_s{setting.severity}_{setting.camera}.mp4"
                video = out_dir / "videos" / name
                video.parent.mkdir(exist_ok=True)
                imageio.mimsave(video, frames, fps=env_cfg.fps, macro_block_size=1)
                video_path = str(video)

            row.update(
                success=int(success),
                steps=steps,
                video_path=video_path,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            append_row(csv_path, row)
            done_now += 1

            elapsed = time.perf_counter() - start
            remaining = total - len(completed) - done_now
            print(
                f"[{len(completed) + done_now}/{total}] {setting.corruption} s{setting.severity} "
                f"{setting.camera} task {task_id} init {init_state}: "
                f"{'SUCCESS' if success else 'fail'} in {steps} steps | "
                f"{elapsed / done_now:.0f} s/episode, about {elapsed / done_now * remaining / 60:.0f} min left",
                flush=True,
            )
            if done_now == max_episodes:
                env.close()
                return False
        # Free the renderer before moving on; the next reset rebuilds it.
        env.close()
    return True

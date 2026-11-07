"""Run one clean SmolVLA episode on one LIBERO task, save an mp4, print success and steps (M0)."""

import argparse
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# Must be set before anything imports MuJoCo. Respect an explicit override (e.g. osmesa).
os.environ.setdefault("MUJOCO_GL", "egl")


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--amp", action="store_true", help="run inference under torch autocast")
    p.add_argument("--max-steps", type=int, help="stop early; the result is then not a real score")
    p.add_argument("--n-action-steps", type=int, help="override the checkpoint's replanning interval")
    p.add_argument("--image-size", type=int, default=256, help="the checkpoints declare 256x256 inputs")
    p.add_argument("--out", type=Path, default=Path("results/smoke_test/smoke_test.mp4"))
    return p.parse_args()


def main() -> None:
    """Load the policy, roll out one episode, and report what the policy received."""
    args = parse_args()
    ensure_libero_config()
    pick_egl_device()

    # Imported late so MUJOCO_GL, the EGL device, and the LIBERO config are in place first.
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION
    from lerobot.utils.random_utils import set_seed

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available. Fix that, or pass --device cpu for a pipeline check.")
    set_seed(args.seed)

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.pretrained_path = args.checkpoint
    policy_cfg.device = args.device
    if args.n_action_steps is not None:
        policy_cfg.n_action_steps = args.n_action_steps

    # Checkpoints name their cameras differently. Input order is agentview, then wrist.
    image_keys = [k for k in policy_cfg.input_features if ".images." in k]
    cameras = [k.split(".images.")[1] for k in image_keys]
    env_cfg = LiberoEnv(
        task=args.suite,
        task_ids=[args.task_id],
        observation_height=args.image_size,
        observation_width=args.image_size,
        camera_name_mapping={"agentview_image": cameras[0], "robot0_eye_in_hand_image": cameras[1]},
    )

    env = make_env(env_cfg, n_envs=1, use_async_envs=False)[args.suite][args.task_id]
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)
    amp = torch.autocast(device_type=args.device) if args.amp else nullcontext()

    task = env.call("task_description")[0]
    max_steps = env.call("_max_episode_steps")[0]
    limit = min(max_steps, args.max_steps) if args.max_steps else max_steps
    print(f"checkpoint: {args.checkpoint}\nsuite: {args.suite} task {args.task_id}: {task}")
    print(f"max steps: {max_steps} | n_action_steps: {policy_cfg.n_action_steps}")

    policy.reset()
    obs, _ = env.reset(seed=[args.seed], options={NEW_ROLLOUT_OPTION: True})
    print(f"raw observation keys: {list(obs)}")
    for name, image in obs["pixels"].items():
        print(f"  pixels/{name}: shape {image.shape}, dtype {image.dtype}")

    frames, success, step = [], False, 0
    start = time.perf_counter()
    while step < limit:
        # LIBERO renders upside down; LiberoProcessorStep rotates 180 degrees for the policy.
        # Rotate each view separately: flipping the joined image would also swap the panels.
        frames.append(np.hstack([obs["pixels"][c][0][::-1, ::-1] for c in cameras]))
        batch = preprocess_observation(obs)
        batch["task"] = [task]
        batch = preprocessor(env_preprocessor(batch))
        if step == 0:
            print("policy input:")
            for key, value in batch.items():
                if hasattr(value, "shape"):
                    print(f"  {key}: shape {tuple(value.shape)}, dtype {value.dtype}")

        with torch.inference_mode(), amp:
            action = policy.select_action(batch)
        action = env_postprocessor({ACTION: postprocessor(action)})[ACTION]
        obs, _, terminated, truncated, info = env.step(action.to("cpu").numpy())
        step += 1

        final = info.get("final_info")
        source = final if isinstance(final, dict) else info
        success = bool(source.get("is_success", [False])[0])
        if terminated[0] or truncated[0]:
            break
    elapsed = time.perf_counter() - start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=env_cfg.fps, macro_block_size=1)
    env.close()

    print(f"success: {success}\nsteps: {step}")
    if step < max_steps and not success and not (terminated[0] or truncated[0]):
        print("NOTE: stopped early by --max-steps, so this is not a real score.")
    print(f"time: {elapsed:.1f}s ({elapsed / step:.2f}s per step)")
    if args.device == "cuda":
        # Reserved is what counts against the card's limit; allocated is what tensors actually used.
        print(
            f"peak VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB reserved, "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB allocated"
        )
    print(f"video: {args.out} ({len(frames)} frames; agentview left, wrist right)")


if __name__ == "__main__":
    main()

# Setup notes

Verified facts for this project. Last updated 2026-11-07 (M0 done, M1 awaiting the team's look at the grid).

## Status

All M0 checks pass:

| M0 check | State |
|---|---|
| Repo skeleton | Done |
| LeRobot + LIBERO installed | Done, CUDA torch |
| Checkpoint found, published rates recorded | Done (the checkpoint choice is provisional, see below) |
| `torch.cuda.is_available()` is True | True, RTX 3050 Ti Laptop GPU, 4096 MiB |
| One clean episode on GPU, video saved | Success at step 74, `results/smoke_test/smoke_test.mp4` |
| Camera keys and image shape recorded | Done |

## Boot the right kernel

**The GPU only works under kernel `6.17.0-22-generic`.** The newest installed kernel, `7.0.0-38`, has no NVIDIA module, and it is what GRUB boots by default. After a normal reboot, `nvidia-smi` fails and `torch.cuda.is_available()` is False.

To select the right kernel, hold Shift or tap Esc at boot, then choose "Advanced options". Check with `uname -r`. Userspace driver: `580.126.09`, `nvidia-driver-580-open`.

The alternative, `sudo apt install linux-modules-nvidia-580-open-7.0.0-38-generic`, has not been applied. A dry run shows it also upgrades the driver to `580.178.04`, adds kernel `7.0.0-42`, and **removes** the `6.17.0-22` NVIDIA module.

## Environment

Conda env `vla`, Python 3.12.16 (LeRobot 0.6.1 requires >=3.12).

```bash
conda create -y -n vla python=3.12
conda activate vla
pip install -r requirements.txt
```

The env was actually built in two steps: CPU torch first, then `pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128`. `requirements.txt` was checked with `pip install --dry-run` against that env, not with a fresh install. The bundled CUDA runtime is 12.8; no system CUDA is needed.

| Package | Version |
|---|---|
| lerobot | 0.6.1 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| transformers | 5.5.4 |
| hf_libero | 0.1.4 |
| robosuite | 1.4.0 |
| mujoco | 3.8.1 |
| gymnasium | 1.4.0 |
| numpy | 2.2.6 |

**Run commands with ROS removed from the environment.** This machine sources ROS Jazzy, which sets `PYTHONPATH` and `LD_LIBRARY_PATH` to system locations:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH ~/anaconda3/envs/vla/bin/python scripts/smoke_test.py
```

The command uses all defaults: `HuggingFaceVLA/smolvla_libero`, `libero_spatial` task 0, seed 1000, GPU.

## Checkpoints

Two official-looking SmolVLA LIBERO checkpoints exist. Neither model card publishes a success rate.

| | `HuggingFaceVLA/smolvla_libero` | `lerobot/smolvla_libero` |
|---|---|---|
| Commit | `6721902bc4d6` | `a24f1f380461` |
| Weights file | 1.2 GB | 0.9 GB (mostly BF16) |
| Image keys | `image`, `image2` | `camera1`, `camera2` |
| `n_action_steps` | 1 | 50 |
| VLM | `SmolVLM2-500M-Instruct` | `SmolVLM2-500M-Video-Instruct`, 16 layers |
| Training info | No `train_config.json` | `lerobot/libero`, 25k steps, batch 32 |
| Known score | 79/100 on Spatial, community report ([discussion](https://huggingface.co/HuggingFaceVLA/smolvla_libero/discussions/1)) | None found; used in LeRobot CI as a smoke test |
| Loads and steps in 0.6.1 | Yes | Yes |

`smoke_test.py` defaults to `HuggingFaceVLA/smolvla_libero` because it is the only one with any reported score. **This choice is provisional.** M2 should measure clean success to settle it.

`n_action_steps` matters. The SmolVLA paper's Table 13 gives LIBERO averages of 80.3% at 1 step, 82.8% at 10, and 51.8% at 50, so `lerobot/smolvla_libero`'s default of 50 may score low unless overridden. On the other hand, 1 step means running inference at every sim step, which is the slowest setting.

## Published success rates

SmolVLA paper ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)), Table 2, 10 trials per task:

| Model | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| SmolVLA (0.24B) | 87 | 93 | 88 | 63 | 82.75 |
| **SmolVLA (0.45B)** | 90 | 96 | 92 | 71 | 87.3 |
| SmolVLA (2.25B) | 93 | 94 | 91 | 77 | 88.75 |

These are the authors' own trained models, **not** scores for either Hub checkpoint. Other measurements run lower: the CRT paper noted in `CLAUDE.md` saw 43% on LIBERO-10, and the community run above saw 79% on Spatial. Compare M2 baselines with those in mind.

## Observations

Measured from `scripts/smoke_test.py` on `libero_spatial` task 0.

- Raw env keys: `pixels`, `robot_state`.
- Cameras: `agentview_image` and `robot0_eye_in_hand_image`, shown to the policy as `observation.images.image` and `observation.images.image2` (or `camera1`, `camera2`, set through `camera_name_mapping`).
- Raw frames: `(1, 256, 256, 3)` uint8. Policy input: `(1, 3, 256, 256)` float32.
- `observation.state`: `(1, 8)`, end-effector position, axis-angle, gripper.
- Action: 7-dim in [-1, 1].
- The policy resizes images to 512x512 internally.
- Max steps: spatial 280, object 280, goal 300, `libero_10` 520, `libero_90` 400.

## Gotchas

- **robosuite only tries EGL device 0, which fails here.** The devices are 0 NVIDIA (fails with `EGL_NOT_INITIALIZED`), 1 Intel UHD (works), 2 llvmpipe (works, software). `smoke_test.py` picks the first that initializes, which is Intel, so simulator rendering uses no NVIDIA VRAM. **M2's `evaluate.py` needs the same logic.** Why NVIDIA EGL fails is not diagnosed, since it isn't needed. Pixels could differ slightly on another renderer such as Colab's, so don't compare individual episodes across machines.
- **`torchcodec` fails to import** because this machine has no FFmpeg shared libraries. Nothing in M0 needs it. **M5 reads `lerobot/libero` video, so it will hit this locally.** Try `--dataset.video_backend=pyav` first (`av` is installed); this is untested.
- **Set the image size explicitly.** LeRobot's `LiberoEnv` config defaults to 360x360, but both checkpoints declare 256x256, and so does the `lerobot/libero` dataset. `smoke_test.py` passes 256.
- **Raw frames are upside down.** `LiberoProcessorStep` rotates them 180 degrees before the policy sees them. M2 should corrupt the raw frame at the point where `corrupt()` is applied, and saved videos need the same rotation.
- **Flip each view separately** when building a side-by-side video. Flipping the joined image also swaps the panels.
- **LIBERO prompts on first import** if `~/.libero/config.yaml` is missing, which hangs or crashes a non-interactive run. `smoke_test.py` creates it first. Colab will hit this too.
- **Assets download on first use:** about 400 MB to `~/.cache/libero`. The Hub cache is about 3.9 GB with both checkpoints.
- **`opencv-python` and `opencv-python-headless` are both installed**, at different versions, through LeRobot's dependencies. Nothing has broken so far; check this first if `cv2` misbehaves in M1.

## Corruptions (M1)

Kernel sizes are in pixels for the 256x256 frames. No scaling was needed: on the grid, the plan's blur and fog values meet its rule that severity 1 is barely noticeable and severity 5 is hard but not blank.

**`low_light` differs from the plan**, because the plan's values failed that rule. Severity 1 halved mean brightness (119 to 57), and severity 5 was nearly blank (mean 7, contrast about 5, noise sigma 10 stronger than the remaining signal). The fixed gamma 1.2 made both ends worse.

| Severity | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Plan: brightness / noise sigma (gamma 1.2) | 0.60 / 2 | 0.40 / 4 | 0.25 / 6 | 0.15 / 8 | 0.08 / 10 |
| **Used:** brightness | 0.85 | 0.65 | 0.45 | 0.30 | 0.20 |
| **Used:** gamma | 1.04 | 1.08 | 1.12 | 1.16 | 1.20 |
| **Used:** noise sigma | 1 | 2 | 3 | 4 | 6 |

For M2, `corrupt()` is called on every step, and it costs under 8 ms per frame at severity 5, so it adds little to the 0.44 s/step:

- **Make one new `rng` per episode, seeded from the episode's seed.** `motion_blur` draws its angle from a copy of `rng` without advancing it, so the angle stays fixed for as long as the same `rng` is reused. This meets the plan's "one angle per episode" without changing the contract.
- **Both views share a blur angle** if they share an `rng`. Low-light noise still differs between views and frames, because it does advance `rng`.

Sample frames in `app/assets/` are lossless PNGs from the `libero_spatial` task 0 reset (seed 1000), rotated as the policy sees them. Generate the grid with `python scripts/frame_grid.py`; pass `--input app/assets/sample_wrist.png --output <path>` for the wrist view.

## Timing and VRAM

`HuggingFaceVLA/smolvla_libero` at `n_action_steps=1`, batch size 1, no half precision:

| | s/step | Worst case, 280 steps | Peak VRAM |
|---|---|---|---|
| GPU | 0.44 | about 2 min | 1.24 GB reserved by torch, about 1.4 GB in `nvidia-smi` |
| CPU | 4.35 | about 20 min | n/a |

**It fits in 4 GB without half precision**, so `--amp` is not needed and M0-M2 can stay local.

**M3 sweep estimate:** 240 episodes at up to about 2 minutes each is up to about 8 hours if most fail, which is likely at high severity. That is over the 2-hour ask-first limit, so plan it with the team. Raising `n_action_steps` would cut this, but it changes the policy's behavior, so it would have to be fixed before the baseline.

## Smoke test results

`libero_spatial` task 0, seed 1000, init state 0. One episode says nothing about the success rate; these show that the pipeline and success detection work.

| Run | Result | Time | Video |
|---|---|---|---|
| GPU | Success at step 74 | 32 s | `results/smoke_test/smoke_test.mp4` |
| CPU | Success at step 73 | 318 s | `results/smoke_test/smoke_test_cpu.mp4` |

Both videos are gitignored. The CPU video predates the panel fix, so its panels are swapped: wrist is on the left. **The same seed gave different step counts on CPU and GPU**, so don't expect exact reproduction across devices. The M2 severity-0 check should compare runs on the same device.

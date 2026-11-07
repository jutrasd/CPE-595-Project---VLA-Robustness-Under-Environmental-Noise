# Setup notes

Verified facts for this project. Last updated 2026-11-07 (M0 and M2 done; M1 and M3 awaiting the team's review).

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

**Using `HuggingFaceVLA/smolvla_libero`.** On the same 20 `libero_object` episodes at `n_action_steps=10`, it scored 17/20 and `lerobot/smolvla_libero` scored 18/20. That is well within noise, so there was no reason to switch.

**`n_action_steps=10` for all evaluation (team decision, 2026-11-07).** This overrides the checkpoint's default of 1 and is about 3x faster. The SmolVLA paper's Table 13 gives LIBERO averages of 80.3% at 1 step, 82.8% at 10, and 51.8% at 50. Every comparison must use the same value. The CSV has no column for it, so each sweep saves a `run_config.json`, and resuming with a different value is refused. `smoke_test.py` still uses the checkpoint's default unless you pass `--n-action-steps`.

## Published success rates

SmolVLA paper ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)), Table 2, 10 trials per task:

| Model | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| SmolVLA (0.24B) | 87 | 93 | 88 | 63 | 82.75 |
| **SmolVLA (0.45B)** | 90 | 96 | 92 | 71 | 87.3 |
| SmolVLA (2.25B) | 93 | 94 | 91 | 77 | 88.75 |

These are the authors' own trained models, **not** scores for either Hub checkpoint. Other measurements run lower: the CRT paper noted in `CLAUDE.md` saw 43% on LIBERO-10, and the community run above saw 79% on Spatial. Compare M2 baselines with those in mind.

## Clean baselines (M2)

`HuggingFaceVLA/smolvla_libero`, `n_action_steps=10`, all 10 tasks x init states 0 and 1, base seed 1000:

| Suite | Clean success | 95% Wilson CI | Paper, 0.45B model |
|---|---|---|---|
| **`libero_object` (chosen)** | **17/20 = 85%** | 64-95% | 96 |
| `libero_spatial` | 11/20 = 55% | 34-74% | 90 |
| `libero_goal` | 11/20 = 55% | 34-74% | 92 |
| `libero_10` | 9/20 = 45% | 26-66% | 71 |

**The suite is `libero_object`**, per the rule to use the highest measured clean success. Its 85% is a reasonable margin from the paper's 96%, given the wide CI and that the paper tested its own model. `libero_10` agrees with the 43% in the CRT paper. The other three suites sit well below the paper, which has not been investigated, since the MVP doesn't use them. The community's 79% on Spatial used `n_action_steps=1` and a different harness, so 55% may reflect the setting or the small sample.

Results: `results/clean_baseline/` is `libero_object`; the others are in `results/clean_baseline_libero_<suite>/`.

**Severity-0 check passed exactly.** Rerunning 8 episodes (tasks 0, 5, 8, 2) as `fog` at severity 0 reproduced every outcome and step count, including the failures. On the same device with the same seed, evaluation is deterministic, so a change in a sweep comes from the corruption rather than sampling noise.

**Pairing:** each episode's seed is `base + 1000 * task_id + init_state`, independent of the condition. A different seed changes about 4% of pixels by more than 10 levels even with the same init state, so never compare episodes across seeds.

## First sweep results (M3)

`libero_object`, `HuggingFaceVLA/smolvla_libero`, both cameras corrupted, same 20 episodes per point. Clean is 17/20 = 85% (CI 64-95%). "Half of clean" is 42.5%.

| Corruption | Severity 1 | Severity 3 | Severity 5 | First below half of clean |
|---|---|---|---|---|
| Motion blur | 8/20 = 40% | 3/20 = 15% | 0/20 | 1 |
| Defocus blur | 15/20 = 75% | 5/20 = 25% | 0/20 | 3 |
| Low light | 18/20 = 90% | 16/20 = 80% | 0/20 | 5 |
| Fog | 16/20 = 80% | 13/20 = 65% | 6/20 = 30% | 5 |

Paired exact McNemar p-values against clean, uncorrected. With 12 tests, the Bonferroni threshold is about 0.004:

| Corruption | Severity 1 | Severity 3 | Severity 5 |
|---|---|---|---|
| Motion blur | 0.012 | 0.0005 | <0.0001 |
| Defocus blur | 0.63 | 0.0018 | <0.0001 |
| Low light | 1.0 | 1.0 | <0.0001 |
| Fog | 1.0 | 0.22 | 0.0010 |

How to read these:

- **Motion blur at severity 1 is the surprise.** It is hard to see, yet it lost 10 clean successes and gained 1. Its CI (22-61%) does not overlap clean's, but p=0.012 does not survive Bonferroni, so call it strong evidence pending more episodes. The kernel check found no bug: blur centroids shift at most 0.24 px over 1500 kernels, kernels are normalized, and severity 0 reproduces clean.
- **Possible direction effect (exploratory).** At motion blur severity 1, episodes with a near-vertical blur (60-120 degrees) succeeded 7/11 times, against 1/9 otherwise. Fisher p is about 0.03, but the cutoff was chosen after seeing the data, and angle is tied to seed. This is a hypothesis to test, not a result. Severity-1 defocus changes pixels about as much and lowers sharpness more, yet lost only 3 episodes, which fits a direction effect.
- **Low light looks like a cliff, but severities 2 and 4 were not run.** The lines between 3 and 5 are interpolation. Its position also depends on the values tuned in M1, so report it as "tolerant through severity 3", not as an absolute threshold.
- **All of this is `libero_object` and 20 episodes.** Differences between neighboring points mostly fall inside the CIs.

Results: `results/mvp_sweep/episodes.csv`, `results/summary.csv`, `results/plots/smolvla_degradation.png`.

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

- **The eval loop grows about 0.12 GB per episode** with the policy loaded. The first M3 sweep reached 7.1 GB at episode 36 and was OOM-killed along with VS Code. The simulator alone does not leak; the exact source was not traced. `run_sweep.py` therefore restarts a worker process every 20 episodes, which is safe because results are saved per episode and each episode seeds itself (verified: step counts match across restarts).
- **Run long sweeps outside VS Code's process group** so neither can kill the other. Use the command under "Running a long sweep".
- **EGL tracebacks at process exit are harmless.** All baseline runs exited 0 with complete CSVs.
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

At `n_action_steps=10`, a 20-episode `libero_object` baseline took about 5 minutes (15 s per episode). A failed episode takes about 30 s, so the 240-episode M3 sweep takes at most about 2 hours.

## Running a long sweep

```bash
R=$PWD
systemd-run --user --unit=vla-sweep --collect -p MemoryMax=8G -p MemorySwapMax=2G \
  -p WorkingDirectory="$R" \
  -p StandardOutput=append:"$R/results/mvp_sweep/sweep.log" -p StandardError=append:"$R/results/mvp_sweep/sweep.log" \
  /usr/bin/env -u PYTHONPATH -u LD_LIBRARY_PATH ~/anaconda3/envs/vla/bin/python scripts/run_sweep.py configs/mvp_sweep.yaml
```

Check it with `systemctl --user is-active vla-sweep` and `tail results/mvp_sweep/sweep.log`. Stop it with `systemctl --user stop vla-sweep`. Rerunning the same command resumes from the CSV.

## Smoke test results

`libero_spatial` task 0, seed 1000, init state 0. One episode says nothing about the success rate; these show that the pipeline and success detection work.

| Run | Result | Time | Video |
|---|---|---|---|
| GPU | Success at step 74 | 32 s | `results/smoke_test/smoke_test.mp4` |
| CPU | Success at step 73 | 318 s | `results/smoke_test/smoke_test_cpu.mp4` |

Both videos are gitignored. The CPU video predates the panel fix, so its panels are swapped: wrist is on the left. **The same seed gave different step counts on CPU and GPU**, so don't expect exact reproduction across devices. The M2 severity-0 check should compare runs on the same device.

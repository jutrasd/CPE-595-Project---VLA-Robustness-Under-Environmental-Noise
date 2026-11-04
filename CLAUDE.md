# CLAUDE.md — VLA Robustness Under Visual Degradation (CPE 595)

Read this first every session. The step-by-step build plan is `docs/MVP_PLAN.md`; work through it in order and don't skip ahead.

## What this project is

A course project for CPE 595 Applied Machine Learning (Dr. Tao Han, Stevens). Three-person team:
- **Danny:** environment setup, evaluation loop, ACT training
- **William:** corruption functions, sweeps, statistics and plots
- **Jagger:** Streamlit app, deployment, documentation

**Research question:** How do robot manipulation policies fail as camera images degrade (motion blur, defocus blur, low light, fog)? Does a language-conditioned VLA fail differently from a vision-only policy? Does training with corruption augmentation help on corruption types the model never saw?

**Setup:** LIBERO simulation (MuJoCo) through LeRobot.
- **SmolVLA** (~450M params): the VLA under test. Pretrained LIBERO checkpoint, evaluation only.
- **ACT** (~52M params): the vision-only baseline. Small enough that we train it ourselves; this is the project's ML training component.

**Metric:** task success rate (LIBERO checks the goal condition automatically), with 95% Wilson confidence intervals, plotted against corruption severity 0–5.

**Course requirements this must satisfy:** public GitHub repo (source, README with setup and the AI tools used, requirements.txt), a deployable app with a live URL (or a 2–5 min video as backup), and documentation including an architecture diagram. Grading weights: technical execution 30%, presentation/report 30%, problem definition 20%, innovation 10%, live demo 10%.

## Decisions already made, and why (do not undo without asking)

- **No trained image-restoration model.** A 2026 paper (CRT, arXiv 2602.01158) already put a restoration network in front of SmolVLA/π0.5 on LIBERO through LeRobot. Training took ~1.3 days on a Quadro 6000, and it *lowered* SmolVLA's clean success (43% → 33% on LIBERO-10). Our professor also flagged restoration as risky. Classical fixes (CLAHE, unsharp masking) may appear only as cheap baselines, later.
- **Our novelty is diagnosis, not "VLAs break."** Many benchmarks (LIBERO-Plus, VLA-Arena, STRONG-VLA, VLA-Gauge) already show VLAs are brittle. We add: graded, matched severities across a VLA and a vision-only policy; per-camera-view isolation (which view's corruption causes the failure); and augmentation training tested on held-out corruption types.
- **Expect surprises and report them.** One study found VLAs fairly tolerant to brightness shifts but very sensitive to small camera rotations. If low light barely hurts, that is a finding, not a bug.
- **No CARLA or driving.** Not feasible on our GPU; it's "future work" only.
- **Simulation only, no physical robot.** State it as a limitation; don't work around it.
- **The deployed app shows precomputed results.** The simulator does not run in the hosted app.

## Hard constraints

- **Local GPU:** RTX 3050 Ti Laptop, **4 GB VRAM**, on Ubuntu (dual boot). Anything that won't fit goes to free Google Colab (T4; sessions disconnect).
- Batch size 1 for SmolVLA inference; use half precision if the installed LeRobot supports it for that policy; never load two policies at once; keep the checkpoint's native image resolution.
- **Runs must be resumable:** append one results row per episode immediately; on restart, skip combinations already in the CSV. On Colab, write results and checkpoints to Google Drive.
- Headless rendering needs `MUJOCO_GL=egl` (or `osmesa` as a fallback). Set it in scripts, not by hand.
- **Jetson Orin Nano:** out of scope. Don't use it.

## Simplicity rules

- Smallest thing that produces a correct, plottable result. No frameworks, registries, plugin systems, or config hierarchies. Plain argparse CLIs; one small YAML per sweep at most.
- Use LeRobot's existing LIBERO environment, policies, eval, and training entry points. Don't reimplement policies, the simulator, or the training loop.
- Corruptions use NumPy + OpenCV only.
- One LIBERO suite for the MVP: whichever suite has the highest clean SmolVLA success with the checkpoint we use. Measure it; don't assume.
- Don't build anything the current milestone in `docs/MVP_PLAN.md` doesn't need.

## Shared contracts (teammates code against these; don't change them silently)

```python
# src/corruptions.py
def corrupt(frame: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Return a degraded copy of an HxWx3 uint8 RGB frame.

    kind: "none" | "motion_blur" | "defocus_blur" | "low_light" | "fog"
    severity: 0..5. Severity 0 (or kind "none") MUST return the input unchanged.
    Output has the same shape and dtype as the input.
    """
```

Results: one row per episode in `results/<sweep_name>/episodes.csv`, columns:

`policy, checkpoint, suite, task_id, init_state, seed, corruption, severity, camera, success, steps, video_path, timestamp`

- `camera` is `both`, `agentview`, or `wrist` (which camera views were corrupted).
- `video_path` may be empty; record video for only the first episode of each setting.
- Corrupt only the image passed to the policy. Saved videos show the corrupted view, so we can see what the policy saw.

## Code conventions

- Python 3.10+, PyTorch, LeRobot, NumPy, OpenCV, pandas, matplotlib, Streamlit, pytest.
- Every public function has a short docstring. Comments explain *why* or non-obvious behavior, not what the line does.
- Seed everything (NumPy, torch, env reset). Use the same seed and init-state list for every condition so comparisons are paired.
- Small modules in `src/`, thin CLI scripts in `scripts/`, tests in `tests/`. Pin versions in `requirements.txt` once the environment works.
- Gitignore videos, checkpoints, and raw logs. Commit summary CSVs, plots, and a few short sample videos for the app.

## Verification habits (the course grades this; see the diagnostics lecture)

1. **Clean baseline first.** If clean success is far below the published number, stop and debug setup. Robustness numbers are meaningless otherwise.
2. **Severity 0 reproduces clean.** Same seeds, same outcomes (or within noise if the policy samples actions).
3. **Look at the inputs.** Generate a frame grid (every corruption × every severity) and check it before any sweep.
4. **Report CIs.** With 20 episodes, a ±20-point swing can be noise.
5. **ACT training:** check training loss first (high = optimization problem), then the train/validation gap (overfitting). Choose settings on validation; evaluate held-out corruptions once.

## Stop and ask the human before

- Installing GPU drivers or CUDA, or anything needing sudo.
- Downloads over ~5 GB.
- Proceeding when a checkpoint or dataset doesn't exist, or its clean score is unexpectedly low.
- Changing the shared contracts above.
- Starting anything expected to take over ~2 hours of GPU time.

## Verify, don't assume

Hugging Face repo IDs for the SmolVLA LIBERO checkpoint and LIBERO datasets, the LeRobot LIBERO env and eval/train CLI for the installed version, camera key names in observations, and whether a pretrained ACT LIBERO checkpoint exists. Look these up in the LeRobot source/docs and HF Hub, and record what you found (versions, commands, IDs, gotchas) in `docs/SETUP_NOTES.md`.

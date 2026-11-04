# MVP Plan — build order for Claude Code

Goal: undeniable, presentable progress for the midterm progress presentation, with nothing extra. Work milestone by milestone. Each one ends with a concrete artifact and a check. Context, constraints, and contracts are in `CLAUDE.md`.

**Midterm-ready means:** M0–M4 done and M5 started. That gives a working pipeline, clean baselines, a first SmolVLA degradation plot, ACT trained on clean data, and a deployed app showing it.

---

## Target repo layout

```
vla-robustness/
├── CLAUDE.md
├── README.md                 # setup, usage, AI tools used (course requirement)
├── requirements.txt
├── .gitignore                # results/**/videos, checkpoints/, *.mp4 except app samples
├── docs/
│   ├── MVP_PLAN.md           # this file
│   └── SETUP_NOTES.md        # verified versions, commands, checkpoint IDs, gotchas
├── src/
│   ├── corruptions.py        # corrupt() + per-kind helpers
│   ├── evaluate.py           # run episodes, append CSV rows, resumable
│   └── stats.py              # success rate + Wilson CI, aggregation
├── scripts/
│   ├── smoke_test.py         # one clean episode -> video
│   ├── frame_grid.py         # corruption x severity image grid
│   ├── run_sweep.py          # CLI wrapper around evaluate.py
│   └── plot_results.py       # degradation curves from episodes.csv
├── configs/
│   └── mvp_sweep.yaml
├── tests/
│   └── test_corruptions.py
├── notebooks/
│   └── colab_runner.ipynb    # mount Drive, install, run_sweep / train ACT
├── app/
│   └── streamlit_app.py
└── results/                  # summary CSVs + plots committed; videos gitignored
```

---

## M0 — Environment and smoke test (owner: Danny)

1. Create the repo skeleton above (empty modules with docstrings are fine).
2. Install LeRobot with LIBERO support per its docs; record exact versions in `docs/SETUP_NOTES.md`.
3. Find the SmolVLA LIBERO checkpoint on HF Hub and its published per-suite success rates. Record them.
4. `scripts/smoke_test.py`: load SmolVLA, run **one** clean episode on one task, save an mp4, print success and steps. Also print the observation keys and image shapes.

**Done when:** the video exists and plays, `torch.cuda.is_available()` is True, and SETUP_NOTES lists the camera keys, image shape, and checkpoint ID.
**If it fails:** if SmolVLA won't fit in 4 GB even at batch size 1, move M0–M2 to Colab and note it. Don't spend more than a day fighting the local install.

## M1 — Corruption module and frame grid (owner: William; can run in parallel with M0)

Implement `src/corruptions.py` with the contract in CLAUDE.md. Starting parameters below. They're deliberately simple, and you tune them by looking at the frame grid, not by theory. Scale kernel sizes to the actual frame resolution once M0 reports it.

| Severity | motion_blur (line kernel length, px) | defocus_blur (disk radius, px) | low_light (brightness × / noise σ in 0–255) | fog (transmission t, haze A=200) |
|---|---|---|---|---|
| 0 | none | none | 1.0 / 0 | 1.0 |
| 1 | 3 | 1 | 0.60 / 2 | 0.85 |
| 2 | 5 | 2 | 0.40 / 4 | 0.70 |
| 3 | 9 | 3 | 0.25 / 6 | 0.55 |
| 4 | 13 | 5 | 0.15 / 8 | 0.40 |
| 5 | 19 | 7 | 0.08 / 10 | 0.25 |

- **motion_blur:** convolve with a normalized line kernel; draw the angle once per episode from `rng` so it's consistent within an episode.
- **defocus_blur:** convolve with a normalized disk kernel.
- **low_light:** scale toward black, apply gamma 1.2, add Gaussian noise *after* darkening (mimics sensor noise in the dark), clip to uint8.
- **fog:** `out = frame * t + A * (1 - t)`, uniform haze. (No depth map; note this as a simplification.)
- Draw any randomness from the passed `rng` only, so runs are reproducible.

`tests/test_corruptions.py`: severity 0 returns an identical array; shape and dtype are preserved for every kind and severity; the same seed gives the same output; mean brightness decreases with low_light severity.

`scripts/frame_grid.py`: take one saved frame (from M0, or any image until then) and write `results/frame_grid.png` (rows = corruption, columns = severity 0–5).

**Done when:** tests pass and the team has eyeballed the grid. Severity 1 should be barely noticeable, and severity 5 hard but not blank.

## M2 — Evaluation loop and clean baselines (owner: Danny)

`src/evaluate.py`, wrapped by `scripts/run_sweep.py`:
- Inputs: policy checkpoint, suite, task IDs, init states, seeds, a list of (corruption, severity, camera) settings, output dir.
- For each setting × task × init state: reset the env with a fixed seed, and each step corrupt the selected camera image(s) with `corrupt()` before the policy sees them. Run until success or the suite's max steps.
- Append one CSV row per episode immediately. On start, read the existing CSV and skip completed rows.
- Record video only for the first episode of each setting.
- Print a time-per-episode estimate after the first few episodes.

Run clean baselines for SmolVLA on the chosen suite: all 10 tasks × 2 init states = 20 episodes (more if time allows).

**Done when:** `results/clean_baseline/episodes.csv` exists, clean success is within a reasonable margin of the published number (record both in SETUP_NOTES), and a severity-0 rerun of a few episodes matches.
**If clean success is far off:** stop and debug (wrong suite, wrong image key or flip, wrong normalization, wrong max steps). Don't start sweeps.

## M3 — First SmolVLA sweep and plots (owners: William runs and analyzes, Danny supports)

`configs/mvp_sweep.yaml`: SmolVLA, 4 corruptions × severities {1, 3, 5}, camera `both`, the same 20 task/init-state pairs as the baseline. That's 12 settings × 20 = 240 episodes. Use the per-episode time from M2 to decide local vs. Colab. Severities 2 and 4 come after the midterm.

`src/stats.py`: success rate and Wilson 95% CI per (policy, corruption, severity, camera).
`scripts/plot_results.py`: one line per corruption, x = severity (0 from the clean baseline), y = success rate with CI band. Save `results/plots/smolvla_degradation.png` and a `results/summary.csv`.

**Done when:** the plot exists and the team can state, in one sentence each, which corruption hurt most and at what severity success drops below half of clean.

## M4 — Streamlit app v1, deployed (owner: Jagger; start once M1 lands, add real results after M3)

`app/streamlit_app.py`, reading only committed files:
1. **Corruption explorer (interactive):** choose a sample LIBERO frame, pick a corruption and drag a severity slider, and see the corrupted frame live using `src/corruptions.py`. This works without any GPU or simulator and is the live-demo hook.
2. **Results:** load `results/summary.csv` and show the degradation plot, with a filter by corruption.
3. **Rollouts:** side-by-side clean vs. corrupted sample videos (a few short mp4s committed under `app/assets/`).
4. **About:** problem, method, and limitations (simulation only; synthetic corruptions).

Deploy to Streamlit Community Cloud or Hugging Face Spaces. The app's requirements must not include LeRobot or MuJoCo; keep a separate `app/requirements.txt`.

**Done when:** the public URL loads on a phone and the slider works.

## M5 — ACT baseline trained on clean data (owner: Danny; start in parallel with M3)

1. Check whether a pretrained ACT checkpoint exists for our suite. If yes, evaluate it clean and skip training for now.
2. Otherwise train ACT with LeRobot's training entry point on the LIBERO demonstration dataset for the same suite, on Colab, checkpointing to Drive. Log train and validation loss.
3. Evaluate clean on the same 20 task/init-state pairs.

**Midterm target:** training curves and a clean success number, even if still improving.
**If ACT clean success is very low** (it may be, since ACT isn't designed for multitask language-conditioned suites): try training on a single task or a few tasks and compare SmolVLA on that same subset. Record the decision in SETUP_NOTES.

---

## After the midterm (not MVP; don't start early)

- Add severities 2 and 4; run the same sweep for ACT, and plot VLA vs. vision-only.
- Per-camera isolation: one corruption at severity 3, corrupting `agentview` only vs. `wrist` only.
- Augmentation experiment: train ACT with random corruptions from {motion_blur, low_light}; test on held-out {defocus_blur, fog}. Compare against clean-trained ACT on identical seeds.
- Optional: CLAHE or unsharp-mask baseline at test time.
- App v2 adds the comparison plots; then the final report, video, and pitch.

## Midterm presentation checklist (what these milestones produce)

- [ ] Pipeline diagram: frame → corrupt → policy → action → sim → success
- [ ] Clean SmolVLA baseline vs. published number
- [ ] Frame grid image
- [ ] First degradation plot with CIs and a one-line finding
- [ ] Short clean vs. corrupted rollout clip
- [ ] ACT training curve and clean result
- [ ] Live app URL
- [ ] What's next, plus problems hit and how AI tools helped

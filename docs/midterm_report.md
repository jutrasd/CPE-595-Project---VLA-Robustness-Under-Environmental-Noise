# Midterm progress presentation: source document

Source for building the slide deck. Each section is one slide. Build only from what is written here.

**Stats verification:** every rate and 95% Wilson CI was recomputed from `results/clean_baseline/episodes.csv` and `results/mvp_sweep/episodes.csv` with `src/stats.py`. **Mismatches with `results/summary.csv`: 0** across all 13 settings.

**Claim rule:** a difference is called real only if the 95% CIs do not overlap. Otherwise the wording is "suggests" or "not distinguishable from clean".

**Scope:** M0-M3 are done. M4 (app) and M5 (ACT) are not done and appear only as next steps.

---

## Slide 1: How Robot Policies Fail as Camera Images Degrade

- Project: VLA Robustness Under Visual Degradation
- Team: Danny, William, Jagger
- CPE 595 Applied Machine Learning, Dr. Tao Han, Stevens
- Midterm progress presentation, November 8, 2026

**Visual:** `app/assets/sample_agentview.png` (the robot's view of the simulated task)

**Speaker:** Danny. We measure how a robot manipulation policy fails as its camera images degrade. Today covers the refined problem, the pipeline, and our first results.

---

## Slide 2: We Changed Focus from Restoring Images to Diagnosing Failures

- Proposal idea: train an image-restoration network in front of the policy
- Already published: CRT (arXiv 2602.01158) restores inputs for SmolVLA and π0.5 on LIBERO
- Costly: 1d 8h 14m of LIBERO training on one Quadro 6000
- It hurt clean performance: SmolVLA fell from 43.0% to 33.0% clean on LIBERO-10
- New focus: diagnose which degradations matter. Our professor also flagged restoration as risky, and the second note, on research basis, is already addressed

**Visual:** Two-box diagram. Left box, struck through: "Proposal: train a restoration model". Arrow to right box: "Now: measure which degradations break the policy, and how".

**Speaker:** Danny. Restoration is already published, it was expensive to train, and it lowered SmolVLA's clean success. So we moved to a diagnosis question we can answer on our hardware.

---

## Slide 3: Only the Policy's View Is Corrupted

- Corruption is applied on every step, before the policy sees the frame
- The simulator's state is untouched, so only perception changes
- Policy inputs: 2 camera views, 8-dim robot state, the task's language instruction
- Output: 7-dim action in [-1, 1]; the policy replans every 10 steps
- An episode ends on LIBERO's success check or at 280 steps (`libero_object`)

**Visual:** Draw a left-to-right loop:
`LIBERO sim` → `camera frames (agentview, wrist; 256×256 RGB)` → `corrupt(frame, kind, severity, rng)` → `SmolVLA (+ language instruction, robot state)` → `7-dim action` → back to `LIBERO sim`.
A branch from `LIBERO sim` goes to `success / fail + steps` → `one CSV row per episode`.

**Speaker:** Danny. The physics never changes; only what the policy sees does. Any drop in success therefore comes from perception.

---

## Slide 4: Setup: One Suite, Paired Episodes, One Laptop GPU

- Suite: `libero_object`, the highest measured clean success of 4 suites
- Checkpoint: `HuggingFaceVLA/smolvla_libero` (commit `6721902bc4d6`), `n_action_steps=10`
- Software: LeRobot 0.6.1, PyTorch 2.11.0 (CUDA 12.8), MuJoCo 3.8.1, robosuite 1.4.0
- Hardware: RTX 3050 Ti Laptop, 4 GB VRAM; all M0-M3 runs were local, and Colab is planned for M5
- 20 episodes per setting (10 tasks × 2 init states); same seeds in every condition; 12 settings = 240 sweep episodes

**Visual:** Small table.

| Item | Value |
|---|---|
| Base seed | 1000 |
| Episode seed | base + 1000 × task_id + init_state |
| Init states | 0, 1 |
| Severities swept | 1, 3, 5 (0 from the clean baseline) |
| Cameras corrupted | both |
| Peak VRAM, smoke test at `n_action_steps=1` | about 1.4 GB |

**Speaker:** Danny. Every condition runs the same episodes with the same seeds, so comparisons are paired. We replan every 10 steps instead of the checkpoint's default of every step, which is about 3× faster.

---

## Slide 5: Clean Baseline: 85% Clean, and Severity 0 Reproduces It Exactly

- Clean success on `libero_object`: 17/20 = 85.0%, 95% CI [64.0, 94.8]
- Published for SmolVLA 0.45B on LIBERO-Object: 96% (SmolVLA paper, Table 2)
- 96% sits just above our CI's upper bound; the paper tested its own model, not this checkpoint
- Severity-0 sanity check: 8 of 8 rerun episodes matched in outcome and step count, including 3 failures
- Same device and seed gives the same result, so within an episode, a changed outcome comes from the corruption

**Visual:** Bar chart with error bar: "Ours, 85.0% [64.0, 94.8]" beside a marker for "Published, 96%". Label the published value "different model".

**Speaker:** Danny. Our baseline comes in below the published number, and we report that gap plainly. The exact severity-0 match tells us the evaluation is deterministic, so an episode's outcome changes only because of the corruption. The CIs still matter, because 20 episodes are only a small sample of possible episodes.

---

## Slide 6: Four Corruptions at Graded Severities 1-5

- Motion blur: normalized line kernel; one random angle per episode
- Defocus blur: normalized disk kernel
- Low light: darken with gamma, then add Gaussian noise after darkening
- Fog: uniform haze, `out = frame × t + 200 × (1 − t)`, with no depth map
- 132 tests pass in `tests/test_corruptions.py`; low-light values were retuned from the plan after inspecting the grid

| Severity | Motion: length (px) | Defocus: radius (px) | Low light: brightness / gamma / noise σ | Fog: transmission t |
|---|---|---|---|---|
| 1 | 3 | 1 | 0.85 / 1.04 / 1 | 0.85 |
| 2 | 5 | 2 | 0.65 / 1.08 / 2 | 0.70 |
| 3 | 9 | 3 | 0.45 / 1.12 / 3 | 0.55 |
| 4 | 13 | 5 | 0.30 / 1.16 / 4 | 0.40 |
| 5 | 19 | 7 | 0.20 / 1.20 / 6 | 0.25 |

**Visual:** `results/frame_grid.png`

**Speaker:** William. These use only NumPy and OpenCV, and Claude Code implemented them at Danny's direction. The plan's low-light values were clearly visible at severity 1 and nearly blank at severity 5, so we retuned them after looking at this grid.

---

## Slide 7: Success Falls with Severity, at Different Rates per Corruption

- Clean: 17/20 = 85.0% [64.0, 94.8]; 20 episodes per point
- Below clean with non-overlapping CIs: motion blur at 1, 3, 5; defocus at 3, 5; low light at 5; fog at 5
- Not distinguishable from clean: defocus at 1; low light at 1, 3; fog at 1, 3

| Corruption | Severity 1 | Severity 3 | Severity 5 |
|---|---|---|---|
| Motion blur | 40.0% [21.9, 61.3] | 15.0% [5.2, 36.0] | 0.0% [0.0, 16.1] |
| Defocus blur | 75.0% [53.1, 88.8] | 25.0% [11.2, 46.9] | 0.0% [0.0, 16.1] |
| Low light | 90.0% [69.9, 97.2] | 80.0% [58.4, 91.9] | 0.0% [0.0, 16.1] |
| Fog | 80.0% [58.4, 91.9] | 65.0% [43.3, 81.9] | 30.0% [14.5, 51.9] |

**Visual:** `results/plots/smolvla_degradation.png`

**Rollout clips (same episode, same seed; task: "pick up the ketchup and place it in the basket"):**
- Clean, success in 145 steps: `app/assets/rollout_clean_object_task4_init0.mp4`
- Motion blur severity 1, failure at 280 steps: `app/assets/rollout_motion_blur_s1_object_task4_init0.mp4`

**Speaker:** William. Each panel is one corruption, and the band is the 95% CI. Severities 2 and 4 were not run, so the lines between points are interpolation. The clips show the same episode failing under a blur that is hard to see.

---

## Slide 8: Key Findings, Strongest Evidence First

- **Strongest:** severity 5 is below clean for all 4: 0% for motion blur, defocus and low light; 30.0% for fog
- **Strong:** motion blur is the only corruption below clean at severity 1: 40.0% [21.9, 61.3] vs. 85.0% [64.0, 94.8]
- **Strong:** low light is not distinguishable from clean through severity 3 (80.0% [58.4, 91.9]), but falls to 0% at severity 5
- **Suggests:** motion blur hurts most. It first drops below half of clean (42.5%) at severity 1, vs. severity 3 for defocus and severity 5 for low light and fog, but its CIs overlap defocus blur's at severities 1 and 3
- **Context:** low-light tolerance is consistent with prior work finding VLAs more robust to brightness than to camera rotation (arXiv 2609.22293); that paper used other models and additive brightness shifts, so this is not a replication

**Visual:** Same plot, cropped or highlighted to the motion blur and low light panels: `results/plots/smolvla_degradation.png`

**Speaker:** William. The surprise is that a blur you can barely see already falls below clean, while low light stays indistinguishable from clean through severity 3. All of this is one suite at 20 episodes per point, so the ranking between motion and defocus blur is only suggestive.

---

## Slide 9: Challenges, and How AI Tools Helped and Misled Us

- GPU missing: the booted kernel had no NVIDIA module. Claude Code diagnosed it read-only; booting kernel `6.17.0-22` fixed it
- Rendering: robosuite only tries EGL device 0, which failed. We now pick the first working device, so rendering runs on the Intel GPU
- Memory, not VRAM, was the limit: Claude Code's sweep loop grew about 0.12 GB per episode, and the out-of-memory killer took out the sweep and VS Code at episode 36. Fix: restart the worker every 20 episodes
- Checks that caught problems: the frame grid exposed bad low-light values, and inspecting a video frame exposed a Claude Code bug that swapped the camera panels
- Claude Code overstated results: it first called the motion blur ranking firm and the baseline gap "reasonable"; rechecking against the CIs corrected both

**Visual:** Two-column list: "Problem" and "How we caught it". No image needed.

**Speaker:** Jagger. AI tools helped with setup and debugging, but they also introduced a memory problem and a video bug and overstated results. Our checks and rereading the CIs caught these, so we verify rather than trust the output.

---

## Slide 10: Next Steps

- M4: Streamlit app with a corruption slider, degradation results and sample rollouts, deployed to a public URL
- M5: train ACT on clean LIBERO data on Colab, logging train and validation loss, then evaluate on the same 20 episodes
- ACT vs. SmolVLA: run the same corruption sweep and compare a vision-only policy with a VLA
- Per-camera isolation: corrupt agentview only vs. wrist only at one severity
- Augmentation: train ACT on motion blur + low light; test on held-out defocus blur + fog

**Visual:** Timeline: `M0-M3 done` → `M4 app` → `M5 ACT` → `ACT vs. SmolVLA` → `per-camera isolation` → `augmentation test`.

**Speaker:** Jagger. The app and ACT are the immediate work, and neither is finished yet. ACT unlocks the main research comparison, a VLA against a vision-only policy.

---

## Slide 11: Limitations

- Simulation only; no physical robot
- Synthetic corruptions: fog is uniform with no depth, and low-light values were tuned by eye
- One suite, chosen as the best of 4 on the same 20 clean episodes we report, so 85.0% may be optimistic
- 20 episodes per setting using only init states 0 and 1; CIs are wide
- Severities 2 and 4 not run; both cameras always corrupted together so far

**Visual:** None, or a single-column list.

**Speaker:** Jagger. These results describe one simulated suite with small samples, so they point to where to look rather than giving final numbers. More episodes and severities come after the midterm.

---

## Backup slides (optional)

### Backup A: Clean Success Across All 4 Suites

| Suite | Clean success | 95% CI | Paper, SmolVLA 0.45B |
|---|---|---|---|
| `libero_object` | 17/20 = 85.0% | [64.0, 94.8] | 96 |
| `libero_spatial` | 11/20 = 55.0% | [34.2, 74.2] | 90 |
| `libero_goal` | 11/20 = 55.0% | [34.2, 74.2] | 92 |
| `libero_10` | 9/20 = 45.0% | [25.8, 65.8] | 71 |

- The cause of the gaps from the paper has not been investigated
- `libero_10` is close to the 43.0% clean SmolVLA figure in the CRT paper
- `lerobot/smolvla_libero` scored 18/20 on the same `libero_object` episodes, which is not distinguishable from our checkpoint's 17/20

**Speaker:** Danny, if asked. These explain the suite choice; the sweep uses only `libero_object`.

### Backup B: Does Blur Direction Matter? (Exploratory Hypothesis)

- Motion blur severity 1, near-vertical angles (60-120°): 7/11 succeeded
- All other angles: 1/9 succeeded
- The cutoff was chosen after seeing the data, and angle is tied to each episode's seed, so this is a hypothesis to test and not a result

**Speaker:** William, if asked. This could become a follow-up experiment that fixes the angle per episode.

---

## Visual assets

All paths below were checked and exist in this repo.

| Slide | File |
|---|---|
| 1 | `app/assets/sample_agentview.png` |
| 6 | `results/frame_grid.png` |
| 7, 8 | `results/plots/smolvla_degradation.png` |
| 7 | `app/assets/rollout_clean_object_task4_init0.mp4` |
| 7 | `app/assets/rollout_motion_blur_s1_object_task4_init0.mp4` |

Diagrams to draw, with no file: slides 2, 3, 5, 9, 10.

## Rollout video for the results slide

- Clean: `app/assets/rollout_clean_object_task4_init0.mp4` (145 steps, 7.2 s, success)
- Corrupted: `app/assets/rollout_motion_blur_s1_object_task4_init0.mp4` (280 steps, 14.0 s, failure)
- Both show agentview on the left and wrist on the right, and the corrupted clip shows what the policy saw.
- Don't use the sweep videos in `results/mvp_sweep/videos/` as clean vs. corrupted examples. They record task 0, init state 0, which also fails when clean.

## Open TODOs

- TODO: team review of `results/frame_grid.png` (required to close M1)
- TODO: team agreement on the slide 8 findings (required to close M3)
- TODO: confirm the wording of the professor's second note on slide 2. The repo records only the restoration note; "research basis, already addressed" comes from Danny.
- TODO: the plan's midterm checklist expects a live app URL and an ACT training curve with a clean result. Neither exists. Slide 10 presents them as next steps.
- TODO: README lacks setup steps and the AI tools used (a course requirement), though not needed for this deck.

## Sources

- Our results: `results/clean_baseline/episodes.csv`, `results/mvp_sweep/episodes.csv`, `results/summary.csv`, `docs/SETUP_NOTES.md`, `configs/`, `src/`
- Slide 9's overstatement point comes from this project's working session with Claude Code; it is not recorded in a repo file.
- SmolVLA paper, arXiv 2506.01844, Table 2
- CRT, "Improving Robustness of Vision-Language-Action Models by Restoring Corrupted Visual Inputs", arXiv 2602.01158
- "Validating, Not Sampling: Region-Level Robustness of Vision-Language and Vision-Language-Action Models", arXiv 2609.22293v1

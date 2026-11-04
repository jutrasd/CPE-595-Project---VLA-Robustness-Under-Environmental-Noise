# CPE-595-Project---VLA-Robustness-Under-Environmental-Noise
We measure how robot manipulation policies fail as camera images degrade (motion blur, defocus blur, low light, fog) in the LIBERO simulator. We compare a vision-language-action model (SmolVLA) against a vision-only policy (ACT), then test whether training ACT with corruption augmentation helps on corruption types it never saw.

Why this version: after the proposal, we found that restoration in front of SmolVLA on LIBERO is already published, that it was expensive to train, and that it hurt SmolVLA's clean performance. So we dropped the restoration model. Our contribution is diagnosing which degradations matter and why, plus an augmentation-training experiment we can run on our hardware.

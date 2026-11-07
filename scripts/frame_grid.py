"""Write an image grid of every corruption at every severity, to check inputs before a sweep (M1)."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Running a script puts scripts/ on the path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruptions import KINDS, MAX_SEVERITY, corrupt  # noqa: E402

LABEL_WIDTH = 130
HEADER_HEIGHT = 30
WHITE = (255, 255, 255)


def build_grid(frame: np.ndarray, seed: int) -> np.ndarray:
    """Return a grid with rows = corruption and columns = severity 0-5."""
    height, width = frame.shape[:2]
    header = np.full((HEADER_HEIGHT, LABEL_WIDTH + width * (MAX_SEVERITY + 1), 3), 30, np.uint8)
    for severity in range(MAX_SEVERITY + 1):
        _label(header, f"severity {severity}", LABEL_WIDTH + width * severity + 8, 21)

    rows = [header]
    for kind in KINDS:
        if kind == "none":
            continue
        label = np.full((height, LABEL_WIDTH, 3), 30, np.uint8)
        _label(label, kind, 8, height // 2)
        # A fresh rng per cell means each column shows what one episode's first frame would.
        cells = [
            corrupt(frame, kind, severity, np.random.default_rng(seed))
            for severity in range(MAX_SEVERITY + 1)
        ]
        rows.append(np.hstack([label, *cells]))
    return np.vstack(rows)


def _label(image: np.ndarray, text: str, x: int, y: int) -> None:
    """Draw text on an image in place."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def main() -> None:
    """Read one frame, build the grid, and save it."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("app/assets/sample_agentview.png"))
    p.add_argument("--output", type=Path, default=Path("results/frame_grid.png"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"Could not read an image from {args.input}")
    grid = build_grid(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Wrote {args.output} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()

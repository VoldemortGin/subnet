from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.protocol import ForgeryMethod
from src.validator.forge import ForgeEngine

ROOT_DIR = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT_DIR / "data" / "clean"
PROBE_DIR = ROOT_DIR / "data" / "probes"

SEED = 42
CANVAS_H, CANVAS_W = 512, 512


def _draw_sample(index: int, rng: np.random.Generator) -> np.ndarray:
    bg_color = rng.integers(150, 240, size=3).tolist()
    canvas = np.full((CANVAS_H, CANVAS_W, 3), bg_color, dtype=np.uint8)

    n_shapes = rng.integers(3, 7)
    for _ in range(n_shapes):
        shape = rng.integers(0, 4)
        color = rng.integers(0, 200, size=3).tolist()
        thickness = int(rng.integers(2, 6))

        if shape == 0:
            center = (int(rng.integers(50, CANVAS_W - 50)), int(rng.integers(50, CANVAS_H - 50)))
            radius = int(rng.integers(20, 80))
            cv2.circle(canvas, center, radius, color, thickness)
        elif shape == 1:
            pt1 = (int(rng.integers(0, CANVAS_W)), int(rng.integers(0, CANVAS_H)))
            pt2 = (int(rng.integers(0, CANVAS_W)), int(rng.integers(0, CANVAS_H)))
            cv2.line(canvas, pt1, pt2, color, thickness)
        elif shape == 2:
            pt1 = (int(rng.integers(30, CANVAS_W - 100)), int(rng.integers(30, CANVAS_H - 100)))
            pt2 = (pt1[0] + int(rng.integers(40, 150)), pt1[1] + int(rng.integers(40, 150)))
            cv2.rectangle(canvas, pt1, pt2, color, thickness)
        else:
            org = (int(rng.integers(30, CANVAS_W - 150)), int(rng.integers(50, CANVAS_H - 30)))
            scale = rng.uniform(0.6, 1.5)
            cv2.putText(canvas, f"S{index}", org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    return canvas


def generate_clean_images(n: int = 5) -> list[Path]:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    paths: list[Path] = []
    for i in range(n):
        img = _draw_sample(i, rng)
        path = CLEAN_DIR / f"clean_{i:03d}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)
    print(f"Generated {n} clean images in {CLEAN_DIR}")
    return paths


def generate_probes(clean_paths: list[Path]) -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    engine = ForgeEngine(seed=SEED)

    methods = list(ForgeryMethod)
    usable = [m for m in methods if m != ForgeryMethod.METADATA]

    for i, path in enumerate(clean_paths):
        method = usable[i % len(usable)]
        probe = engine.generate_probe(path, PROBE_DIR, method=method)

        stem = probe.tampered_image_path.stem
        meta = {
            "verdict": probe.ground_truth.verdict.value,
            "method": probe.ground_truth.method.value if probe.ground_truth.method else None,
        }
        (PROBE_DIR / f"{stem}.json").write_text(json.dumps(meta))
        np.save(str(PROBE_DIR / f"{stem}_mask.npy"), probe.ground_truth.mask)

        print(
            f"  [{probe.ground_truth.method.value:>15s}] "
            f"{path.name} -> {probe.tampered_image_path.name}"
        )

    print(f"Generated {len(clean_paths)} probes in {PROBE_DIR}")


def main() -> None:
    clean_paths = generate_clean_images()
    generate_probes(clean_paths)


if __name__ == "__main__":
    main()

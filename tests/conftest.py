from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[50:150, 50:250] = [128, 128, 128]
    cv2.rectangle(img, (80, 80), (220, 130), (200, 100, 50), -1)
    cv2.circle(img, (150, 100), 30, (50, 200, 100), -1)
    return img


@pytest.fixture
def sample_image_path(tmp_path: Path, sample_image: np.ndarray) -> Path:
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), sample_image)
    return path


@pytest.fixture
def clean_images_dir(tmp_path: Path) -> Path:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    for i in range(3):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(clean_dir / f"img_{i}.png"), img)
    return clean_dir


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out

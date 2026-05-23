from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from src.miner.backends.base import DetectionBackend
from src.protocol import ForgeryMethod

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/grip-unina/TruFor.git"
MODEL_DIR = Path("models/trufor")
WEIGHTS_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"


def _ensure_repo() -> Path:
    repo_path = MODEL_DIR / "TruFor"
    if not repo_path.exists():
        logger.info("Cloning TruFor repo to %s", repo_path)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", REPO_URL, str(repo_path)])
    return repo_path


def _get_repo_path() -> Path | None:
    repo_path = MODEL_DIR / "TruFor"
    return repo_path if repo_path.exists() else None


def _find_weights() -> Path | None:
    repo_path = _get_repo_path()
    if repo_path is None:
        return None
    weights_dir = repo_path / "test_docker" / "weights"
    if weights_dir.exists() and any(weights_dir.glob("*.pth")):
        return weights_dir
    weights_dir = MODEL_DIR / "weights"
    if weights_dir.exists() and any(weights_dir.glob("*.pth")):
        return weights_dir
    return None


class TruForBackend(DetectionBackend):
    def __init__(self, confidence_threshold: float = 0.5, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def name(self) -> str:
        return "trufor"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return _get_repo_path() is not None and _find_weights() is not None

    @classmethod
    def gpu_required(cls) -> bool:
        return True

    @classmethod
    def estimated_vram_mb(cls) -> int:
        return 500

    def _load_model(self):
        if self._model is not None:
            return

        import torch

        repo_path = _ensure_repo()
        test_dir = repo_path / "test_docker"
        if str(test_dir) not in sys.path:
            sys.path.insert(0, str(test_dir))
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        weights_dir = _find_weights()
        if weights_dir is None:
            raise FileNotFoundError(
                f"TruFor weights not found. "
                "Run: python scripts/setup_models.py --install trufor"
            )

        try:
            from trufor import trufor_main  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "TruFor Python module not importable. "
                "Ensure the repo is properly set up: %s", repo_path
            )
            raise

        model = trufor_main(weights_dir=str(weights_dir), device=self.device)
        self._model = model

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        return rgb

    def detect(self, image: np.ndarray) -> tuple[bool, float, np.ndarray | None, ForgeryMethod | None]:
        import torch

        self._load_model()

        rgb = self._preprocess(image)
        img_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            result = self._model(img_tensor)

        if isinstance(result, dict):
            confidence = float(result.get("score", result.get("conf", 0.0)))
            loc_map = result.get("map", result.get("mask", None))
        elif isinstance(result, (tuple, list)):
            confidence = float(result[0]) if len(result) > 0 else 0.0
            loc_map = result[1] if len(result) > 1 else None
        else:
            confidence = 0.0
            loc_map = None

        mask = None
        if loc_map is not None:
            if isinstance(loc_map, torch.Tensor):
                loc_map = loc_map.squeeze().cpu().numpy()
            loc_map = np.clip(loc_map, 0.0, 1.0)
            loc_map = cv2.resize(loc_map, (image.shape[1], image.shape[0]))
            mask = (loc_map > self.confidence_threshold).astype(np.uint8) * 255

        is_tampered = confidence > self.confidence_threshold
        method = self._guess_method(mask) if is_tampered and mask is not None else None

        return is_tampered, round(confidence, 4), mask, method

    def _guess_method(self, mask: np.ndarray) -> ForgeryMethod | None:
        tampered_ratio = np.count_nonzero(mask) / mask.size
        if tampered_ratio < 0.005:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 2:
            return ForgeryMethod.COPY_MOVE
        if tampered_ratio > 0.05:
            return ForgeryMethod.SPLICING
        return ForgeryMethod.INPAINTING

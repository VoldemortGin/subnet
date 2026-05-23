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

REPO_URL = "https://github.com/scu-zjz/Mesorch.git"
MODEL_DIR = Path("models/mesorch")
WEIGHTS_FILENAMES = ["mesorch-98.pth", "mesorch_p-118.pth"]


def _ensure_repo() -> Path:
    repo_path = MODEL_DIR / "Mesorch"
    if not repo_path.exists():
        logger.info("Cloning Mesorch repo to %s", repo_path)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", REPO_URL, str(repo_path)])
    return repo_path


def _get_repo_path() -> Path | None:
    repo_path = MODEL_DIR / "Mesorch"
    return repo_path if repo_path.exists() else None


def _find_weights() -> Path | None:
    repo_path = _get_repo_path()
    if repo_path is None:
        return None
    for filename in WEIGHTS_FILENAMES:
        for candidate in [
            repo_path / "checkpoints" / filename,
            repo_path / filename,
            MODEL_DIR / "weights" / filename,
            MODEL_DIR / filename,
        ]:
            if candidate.exists():
                return candidate
    return None


class MesorchBackend(DetectionBackend):
    def __init__(self, confidence_threshold: float = 0.5, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def name(self) -> str:
        return "mesorch"

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
        return 800

    def _load_model(self):
        if self._model is not None:
            return

        import torch

        repo_path = _ensure_repo()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        weights_path = _find_weights()
        if weights_path is None:
            raise FileNotFoundError(
                "Mesorch weights not found. "
                "Run: python scripts/setup_models.py --install mesorch"
            )

        try:
            from models.mesorch import Mesorch  # type: ignore[import-untyped]
            model = Mesorch()
        except ImportError:
            try:
                from model import Mesorch  # type: ignore[import-untyped]
                model = Mesorch()
            except ImportError:
                logger.warning(
                    "Mesorch model module not importable. "
                    "Ensure repo is set up: %s", repo_path
                )
                raise

        checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        self._model = model

    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        import torch

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (512, 512))

        img_float = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = (img_float - mean) / std

        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

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

    def detect(self, image: np.ndarray) -> tuple[bool, float, np.ndarray | None, ForgeryMethod | None]:
        import torch

        self._load_model()

        tensor = self._preprocess(image)
        with torch.no_grad():
            output = self._model(tensor)

        if isinstance(output, (tuple, list)):
            pred = output[-1]
        elif isinstance(output, dict):
            pred = output.get("mask", output.get("pred", list(output.values())[-1]))
        else:
            pred = output

        if isinstance(pred, torch.Tensor):
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        pred = np.clip(pred, 0.0, 1.0)

        if pred.ndim == 0:
            confidence = float(pred)
            mask = None
        else:
            pred_resized = cv2.resize(pred, (image.shape[1], image.shape[0]))
            confidence = float(np.max(pred_resized))
            mask = (pred_resized > self.confidence_threshold).astype(np.uint8) * 255

        is_tampered = confidence > self.confidence_threshold
        method = self._guess_method(mask) if is_tampered and mask is not None else None

        return is_tampered, round(confidence, 4), mask, method

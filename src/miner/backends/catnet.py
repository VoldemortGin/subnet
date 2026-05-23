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

REPO_URL = "https://github.com/mjkwon2021/CAT-Net.git"
MODEL_DIR = Path("models/catnet")
WEIGHTS_FILENAME = "CAT_full_v2.pth.tar"


def _ensure_repo() -> Path:
    repo_path = MODEL_DIR / "CAT-Net"
    if not repo_path.exists():
        logger.info("Cloning CAT-Net repo to %s", repo_path)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", REPO_URL, str(repo_path)])
    return repo_path


def _get_repo_path() -> Path | None:
    repo_path = MODEL_DIR / "CAT-Net"
    return repo_path if repo_path.exists() else None


def _find_weights() -> Path | None:
    repo_path = _get_repo_path()
    if repo_path is None:
        return None
    for candidate in [
        repo_path / "Weights" / WEIGHTS_FILENAME,
        MODEL_DIR / "weights" / WEIGHTS_FILENAME,
        MODEL_DIR / WEIGHTS_FILENAME,
    ]:
        if candidate.exists():
            return candidate
    return None


class CATNetBackend(DetectionBackend):
    def __init__(self, confidence_threshold: float = 0.5, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def name(self) -> str:
        return "catnet"

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
        return 400

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
                "CAT-Net weights not found. "
                "Run: python scripts/setup_models.py --install catnet"
            )

        checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=False)
        try:
            from model import get_model  # type: ignore[import-untyped]
            model = get_model()
            model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        except ImportError:
            logger.warning(
                "CAT-Net model module not importable. "
                "Ensure repo is set up: %s", repo_path
            )
            raise

        model.to(self.device)
        model.eval()
        self._model = model

    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        import torch

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        if new_h != h or new_w != w:
            rgb = cv2.resize(rgb, (new_w, new_h))

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
        # CAT-Net specializes in JPEG artifacts — bias toward compression/splicing
        if tampered_ratio > 0.05:
            return ForgeryMethod.SPLICING
        return ForgeryMethod.COMPRESSION

    def detect(self, image: np.ndarray) -> tuple[bool, float, np.ndarray | None, ForgeryMethod | None]:
        import torch

        self._load_model()

        tensor = self._preprocess(image)
        with torch.no_grad():
            output = self._model(tensor)

        if isinstance(output, (tuple, list)):
            pred = output[0]
        else:
            pred = output

        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().cpu().numpy()

        if pred.ndim == 0:
            confidence = float(pred)
            mask = None
        elif pred.ndim == 2:
            pred = np.clip(pred, 0.0, 1.0)
            confidence = float(np.max(pred))
            pred_resized = cv2.resize(pred, (image.shape[1], image.shape[0]))
            mask = (pred_resized > self.confidence_threshold).astype(np.uint8) * 255
        else:
            # Multi-channel output — take channel with highest activation
            if pred.shape[0] <= pred.shape[-1]:
                pred = pred[1] if pred.shape[0] > 1 else pred[0]
            else:
                pred = pred[:, :, 1] if pred.shape[-1] > 1 else pred[:, :, 0]
            pred = np.clip(pred, 0.0, 1.0)
            confidence = float(np.max(pred))
            pred_resized = cv2.resize(pred, (image.shape[1], image.shape[0]))
            mask = (pred_resized > self.confidence_threshold).astype(np.uint8) * 255

        is_tampered = confidence > self.confidence_threshold
        method = self._guess_method(mask) if is_tampered and mask is not None else None

        return is_tampered, round(confidence, 4), mask, method

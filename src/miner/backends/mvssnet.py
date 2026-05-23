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

REPO_URL = "https://github.com/dong03/MVSS-Net.git"
MODEL_DIR = Path("models/mvssnet")
WEIGHTS_FILENAME = "mvssnet_casia.pt"


def _ensure_repo() -> Path:
    repo_path = MODEL_DIR / "MVSS-Net"
    if not repo_path.exists():
        logger.info("Cloning MVSS-Net repo to %s", repo_path)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", REPO_URL, str(repo_path)])
    return repo_path


def _get_repo_path() -> Path | None:
    repo_path = MODEL_DIR / "MVSS-Net"
    return repo_path if repo_path.exists() else None


def _find_weights() -> Path | None:
    repo_path = _get_repo_path()
    if repo_path is None:
        return None
    for candidate in [
        repo_path / WEIGHTS_FILENAME,
        repo_path / "ckpt" / WEIGHTS_FILENAME,
        MODEL_DIR / "weights" / WEIGHTS_FILENAME,
        MODEL_DIR / WEIGHTS_FILENAME,
    ]:
        if candidate.exists():
            return candidate
    return None


class MVSSNetBackend(DetectionBackend):
    def __init__(self, confidence_threshold: float = 0.5, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def name(self) -> str:
        return "mvssnet"

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
        return 600

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
                "MVSS-Net weights not found. "
                "Run: python scripts/setup_models.py --install mvssnet"
            )

        try:
            from models.mvssnet import get_mvss  # type: ignore[import-untyped]
            model = get_mvss(backbone="resnet50", pretrained_base=False)
        except ImportError:
            logger.warning(
                "MVSS-Net model module not importable. "
                "Ensure repo is set up: %s", repo_path
            )
            raise

        checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
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
            cls_out = output[0] if len(output) > 0 else None
            seg_out = output[1] if len(output) > 1 else None
        elif isinstance(output, dict):
            cls_out = output.get("cls", None)
            seg_out = output.get("seg", output.get("mask", None))
        else:
            cls_out = None
            seg_out = output

        if cls_out is not None and isinstance(cls_out, torch.Tensor):
            cls_out = cls_out.squeeze().cpu().numpy()
            confidence = float(cls_out) if cls_out.ndim == 0 else float(cls_out.max())
        else:
            confidence = 0.0

        mask = None
        if seg_out is not None:
            if isinstance(seg_out, torch.Tensor):
                seg_out = torch.sigmoid(seg_out).squeeze().cpu().numpy()
            seg_out = np.clip(seg_out, 0.0, 1.0)
            if confidence == 0.0:
                confidence = float(np.max(seg_out))
            seg_resized = cv2.resize(seg_out, (image.shape[1], image.shape[0]))
            mask = (seg_resized > self.confidence_threshold).astype(np.uint8) * 255

        is_tampered = confidence > self.confidence_threshold
        method = self._guess_method(mask) if is_tampered and mask is not None else None

        return is_tampered, round(confidence, 4), mask, method

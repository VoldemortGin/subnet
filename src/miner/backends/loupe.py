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

REPO_URL = "https://github.com/Kamichanw/Loupe.git"
MODEL_DIR = Path("models/loupe")


def _ensure_repo() -> Path:
    repo_path = MODEL_DIR / "Loupe"
    if not repo_path.exists():
        logger.info("Cloning Loupe repo to %s", repo_path)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", REPO_URL, str(repo_path)])
    return repo_path


def _get_repo_path() -> Path | None:
    repo_path = MODEL_DIR / "Loupe"
    return repo_path if repo_path.exists() else None


def _find_weights() -> Path | None:
    repo_path = _get_repo_path()
    if repo_path is None:
        return None
    for candidate in [
        repo_path / "checkpoints",
        repo_path / "weights",
        MODEL_DIR / "weights",
    ]:
        if candidate.exists() and any(
            candidate.glob("*.pth") or candidate.glob("*.pt") or candidate.glob("*.bin")
        ):
            return candidate
    # Loupe uses HuggingFace Perception Encoder; check if repo itself is set up
    config_path = repo_path / "configs"
    if config_path.exists():
        return repo_path
    return None


class LoupeBackend(DetectionBackend):
    def __init__(self, confidence_threshold: float = 0.5, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def name(self) -> str:
        return "loupe"

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
                "Loupe weights/configs not found. "
                "Run: python scripts/setup_models.py --install loupe"
            )

        try:
            from models.loupe import Loupe  # type: ignore[import-untyped]
            model = Loupe()
        except ImportError:
            try:
                from model import Loupe  # type: ignore[import-untyped]
                model = Loupe()
            except ImportError:
                logger.warning(
                    "Loupe model module not importable. "
                    "Ensure repo is set up: %s", repo_path
                )
                raise

        if weights_path.is_file():
            checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        elif weights_path.is_dir():
            ckpt_files = list(weights_path.glob("*.pth")) + list(weights_path.glob("*.pt"))
            if ckpt_files:
                checkpoint = torch.load(str(ckpt_files[0]), map_location=self.device, weights_only=False)
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

        # Loupe may return (classification, segmentation) or dict with cls + mask
        if isinstance(output, dict):
            cls_score = output.get("cls", output.get("score", None))
            seg_pred = output.get("mask", output.get("seg", None))
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            cls_score = output[0]
            seg_pred = output[1]
        elif isinstance(output, (tuple, list)):
            cls_score = None
            seg_pred = output[-1]
        else:
            cls_score = None
            seg_pred = output

        if cls_score is not None:
            if isinstance(cls_score, torch.Tensor):
                cls_score = torch.sigmoid(cls_score).squeeze().cpu().numpy()
            confidence = float(cls_score)
        else:
            confidence = 0.0

        mask = None
        if seg_pred is not None:
            if isinstance(seg_pred, torch.Tensor):
                seg_pred = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
            seg_pred = np.clip(seg_pred, 0.0, 1.0)
            if seg_pred.ndim >= 2:
                pred_resized = cv2.resize(seg_pred, (image.shape[1], image.shape[0]))
                if confidence == 0.0:
                    confidence = float(np.max(pred_resized))
                mask = (pred_resized > self.confidence_threshold).astype(np.uint8) * 255
            elif confidence == 0.0:
                confidence = float(seg_pred)

        is_tampered = confidence > self.confidence_threshold
        method = self._guess_method(mask) if is_tampered and mask is not None else None

        return is_tampered, round(confidence, 4), mask, method

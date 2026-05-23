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

REPO_URL = "https://github.com/RonyAbecidan/ManTraNet-pytorch.git"
_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # project root
MODEL_DIR = _ROOT / "models" / "mantranet"
# The repo stores weights inside MantraNet/ subdirectory
WEIGHTS_FILENAME = "MantraNetv4.pt"


def _ensure_repo() -> Path:
    repo_path = MODEL_DIR / "ManTraNet-pytorch"
    if not repo_path.exists():
        logger.info("Cloning ManTraNet repo to %s", repo_path)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", REPO_URL, str(repo_path)])
    return repo_path


def _get_repo_path() -> Path | None:
    repo_path = MODEL_DIR / "ManTraNet-pytorch"
    return repo_path if repo_path.exists() else None


class ManTraNetBackend(DetectionBackend):
    def __init__(self, confidence_threshold: float = 0.3, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def name(self) -> str:
        return "mantranet"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        repo_path = _get_repo_path()
        if repo_path is None:
            return False
        weights_path = repo_path / "MantraNet" / WEIGHTS_FILENAME
        return weights_path.exists()

    @classmethod
    def gpu_required(cls) -> bool:
        return False

    @classmethod
    def estimated_vram_mb(cls) -> int:
        return 70

    def _load_model(self):
        if self._model is not None:
            return

        import torch

        repo_path = _ensure_repo()
        mantra_pkg = repo_path / "MantraNet"
        if str(mantra_pkg) not in sys.path:
            sys.path.insert(0, str(mantra_pkg))

        # Save/restore cwd because the model loads SRM weights
        # relative to cwd inside its __init__
        import os
        old_cwd = os.getcwd()
        os.chdir(str(mantra_pkg))
        try:
            from mantranet import MantraNet  # type: ignore[import-untyped]

            weights_path = mantra_pkg / WEIGHTS_FILENAME
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"ManTraNet weights not found at {weights_path}. "
                    "Run: python scripts/setup_models.py --install mantranet"
                )
            model = MantraNet(device=torch.device(self.device))
            model.load_state_dict(
                torch.load(str(weights_path), map_location=self.device, weights_only=False)
            )
            model.to(self.device)
            model.eval()
            self._model = model
        finally:
            os.chdir(old_cwd)

    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        import torch

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # MantraNet normalizes internally (x / 255 * 2 - 1), so pass raw float values
        tensor = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
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

        prob_map = output.squeeze().cpu().numpy()
        prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]))
        prob_map = np.clip(prob_map, 0.0, 1.0)

        confidence = float(np.max(prob_map))
        binary_mask = (prob_map > self.confidence_threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        is_tampered = confidence > self.confidence_threshold
        method = self._guess_method(binary_mask) if is_tampered else None

        return is_tampered, round(confidence, 4), binary_mask, method

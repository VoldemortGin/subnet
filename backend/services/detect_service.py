from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from src.miner.backends.ela import ELABackend
from src.miner.detector import ForgeryDetector

from backend.db.store import ImageRecord, Store


class DetectService:
    def __init__(self, store: Store, data_dir: Path):
        self.store = store
        self.data_dir = data_dir
        self.uploads_dir = data_dir / "uploads"
        self.viz_dir = data_dir / "viz"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.backend = ELABackend()
        self.detector = ForgeryDetector(backend=self.backend)

    def analyze_image(self, image_path: Path, image_id: str | None = None) -> ImageRecord:
        if image_id is None:
            image_id = str(uuid.uuid4())[:8]

        task_id = f"analysis_{image_id}"
        response = self.detector.detect_from_path(str(image_path), task_id)

        mask_path: str | None = None
        viz_path: str | None = None

        if response.mask is not None:
            mask_filename = f"{image_id}_mask.png"
            mask_full_path = self.viz_dir / mask_filename
            cv2.imwrite(str(mask_full_path), response.mask)
            mask_path = str(mask_full_path)

            viz_filename = f"{image_id}_overlay.png"
            viz_full_path = self.viz_dir / viz_filename
            self._create_overlay(image_path, response.mask, viz_full_path)
            viz_path = str(viz_full_path)

        record = ImageRecord(
            id=image_id,
            filename=image_path.name,
            path=str(image_path),
            verdict=response.verdict.value,
            confidence=response.confidence,
            method=response.method.value if response.method else None,
            mask_path=mask_path,
            visualization_path=viz_path,
            status="analyzed",
        )
        self.store.images[image_id] = record
        return record

    def _create_overlay(self, image_path: Path, mask: np.ndarray, output_path: Path):
        original = cv2.imread(str(image_path))
        if original is None:
            return

        if mask.shape[:2] != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        overlay = original.copy()
        red_tint = np.zeros_like(original)
        red_tint[:, :, 2] = 255  # Red channel in BGR

        mask_bool = mask > 0
        alpha = 0.4
        overlay[mask_bool] = cv2.addWeighted(
            original[mask_bool], 1 - alpha,
            red_tint[mask_bool], alpha, 0
        )

        cv2.imwrite(str(output_path), overlay)

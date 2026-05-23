from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle

from src.miner.backends.base import DetectionBackend
from src.protocol import ForgeryMethod


class ELABackend(DetectionBackend):
    def __init__(
        self,
        ela_quality: int = 90,
        ela_threshold: int = 25,
        noise_weight: float = 0.1,
        noise_var_threshold: float = 2.0,
        confidence_threshold: float = 0.5,
        min_tampered_ratio: float = 0.005,
    ):
        self.ela_quality = ela_quality
        self.ela_threshold = ela_threshold
        self.noise_weight = noise_weight
        self.noise_var_threshold = noise_var_threshold
        self.confidence_threshold = confidence_threshold
        self.min_tampered_ratio = min_tampered_ratio

    def name(self) -> str:
        return "ela"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def gpu_required(cls) -> bool:
        return False

    @classmethod
    def estimated_vram_mb(cls) -> int:
        return 0

    def _run_ela(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.ela_quality)
        buf.seek(0)
        resaved = np.array(Image.open(buf))
        resaved = cv2.cvtColor(resaved, cv2.COLOR_RGB2BGR)

        diff = cv2.absdiff(image, resaved).astype(np.float32)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        diff_scaled = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, mask = cv2.threshold(diff_scaled, self.ela_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        tampered_ratio = np.count_nonzero(mask) / mask.size
        mean_diff = float(np.mean(diff_scaled[mask > 0])) / 255.0 if np.any(mask) else 0.0
        confidence = min(1.0, tampered_ratio * 10 + mean_diff * 0.5)

        return confidence, mask

    def _run_noise_analysis(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        image_float = image.astype(np.float64) / 255.0
        denoised = denoise_tv_chambolle(image_float, weight=self.noise_weight, channel_axis=-1)
        noise_residual = image_float - denoised
        noise_gray = np.mean(np.abs(noise_residual), axis=2)

        block_size = 16
        h, w = noise_gray.shape
        var_map = np.zeros_like(noise_gray)

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = noise_gray[y : y + block_size, x : x + block_size]
                var_map[y : y + block_size, x : x + block_size] = np.var(block)

        global_var = np.var(noise_gray)
        if global_var < 1e-10:
            return 0.0, np.zeros((h, w), dtype=np.uint8)

        deviation = np.abs(var_map - global_var) / (global_var + 1e-10)
        mask = (deviation > self.noise_var_threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        tampered_ratio = np.count_nonzero(mask) / mask.size
        confidence = min(1.0, tampered_ratio * 5 + float(np.mean(deviation)) * 0.3)

        return confidence, mask

    def _guess_method(self, ela_mask: np.ndarray, noise_mask: np.ndarray) -> ForgeryMethod | None:
        ela_ratio = np.count_nonzero(ela_mask) / ela_mask.size
        noise_ratio = np.count_nonzero(noise_mask) / noise_mask.size

        if ela_ratio < self.min_tampered_ratio and noise_ratio < self.min_tampered_ratio:
            return None

        if noise_ratio > 0.05:
            return ForgeryMethod.SPLICING

        if ela_ratio > 0.01:
            contours, _ = cv2.findContours(ela_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) >= 2:
                return ForgeryMethod.COPY_MOVE
            return ForgeryMethod.INPAINTING

        return ForgeryMethod.COMPRESSION

    def detect(self, image: np.ndarray) -> tuple[bool, float, np.ndarray | None, ForgeryMethod | None]:
        ela_conf, ela_mask = self._run_ela(image)
        noise_conf, noise_mask = self._run_noise_analysis(image)

        if ela_mask.shape != noise_mask.shape:
            noise_mask = cv2.resize(
                noise_mask, (ela_mask.shape[1], ela_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        combined_mask = cv2.bitwise_or(ela_mask, noise_mask)
        combined_confidence = min(1.0, (ela_conf + noise_conf) / 2.0)

        is_tampered = combined_confidence > self.confidence_threshold
        method = self._guess_method(ela_mask, noise_mask) if is_tampered else None

        return is_tampered, round(combined_confidence, 4), combined_mask, method

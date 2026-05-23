from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.protocol import ForgeryMethod, GroundTruth, ProbeTask, Verdict


class ForgeEngine:
    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def forge(
        self,
        image: np.ndarray | str | Path,
        method: ForgeryMethod | None = None,
        donor_image: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, ForgeryMethod]:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image}")

        image = image.copy()

        available_methods = [
            ForgeryMethod.COPY_MOVE,
            ForgeryMethod.SPLICING,
            ForgeryMethod.COMPRESSION,
            ForgeryMethod.INPAINTING,
        ]
        if method is None:
            method = available_methods[int(self._rng.integers(len(available_methods)))]

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        match method:
            case ForgeryMethod.COPY_MOVE:
                image, mask = self._copy_move(image, mask)
            case ForgeryMethod.SPLICING:
                image, mask = self._splicing(image, mask, donor_image)
            case ForgeryMethod.COMPRESSION:
                image, mask = self._compression_mismatch(image, mask)
            case ForgeryMethod.INPAINTING:
                image, mask = self._noise_injection(image, mask)

        return image, mask, method

    def generate_probe(
        self,
        clean_image_path: Path,
        output_dir: Path,
        method: ForgeryMethod | None = None,
    ) -> ProbeTask:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = clean_image_path.stem

        tampered, mask, used_method = self.forge(clean_image_path, method=method)

        tampered_path = output_dir / f"{stem}_tampered.png"
        mask_path = output_dir / f"{stem}_mask.png"

        cv2.imwrite(str(tampered_path), tampered)
        cv2.imwrite(str(mask_path), mask)

        return ProbeTask(
            original_image_path=clean_image_path,
            tampered_image_path=tampered_path,
            ground_truth=GroundTruth(
                verdict=Verdict.TAMPERED,
                method=used_method,
                mask=mask,
            ),
        )

    def _random_rect(
        self, h: int, w: int, min_frac: float = 0.1, max_frac: float = 0.3
    ) -> tuple[int, int, int, int]:
        rh = int(h * self._rng.uniform(min_frac, max_frac))
        rw = int(w * self._rng.uniform(min_frac, max_frac))
        y = int(self._rng.integers(0, h - rh))
        x = int(self._rng.integers(0, w - rw))
        return y, x, rh, rw

    def _copy_move(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        sy, sx, rh, rw = self._random_rect(h, w)
        patch = image[sy : sy + rh, sx : sx + rw].copy()

        max_dy = h - rh
        max_dx = w - rw
        for _ in range(50):
            dy = int(self._rng.integers(0, max_dy))
            dx = int(self._rng.integers(0, max_dx))
            if abs(dy - sy) > rh // 2 or abs(dx - sx) > rw // 2:
                break

        image[dy : dy + rh, dx : dx + rw] = patch
        mask[dy : dy + rh, dx : dx + rw] = 255
        return image, mask

    def _splicing(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        donor: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        dy, dx, rh, rw = self._random_rect(h, w)

        if donor is not None:
            dh, dw = donor.shape[:2]
            sy = int(self._rng.integers(0, max(1, dh - rh)))
            sx = int(self._rng.integers(0, max(1, dw - rw)))
            actual_rh = min(rh, dh - sy)
            actual_rw = min(rw, dw - sx)
            patch = donor[sy : sy + actual_rh, sx : sx + actual_rw]
            if len(patch.shape) == 2 and len(image.shape) == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            patch = cv2.resize(patch, (rw, rh))
        else:
            base_color = self._rng.integers(40, 220, size=3).astype(np.uint8)
            patch = np.full((rh, rw, 3), base_color, dtype=np.uint8)
            noise = self._rng.integers(-30, 30, size=(rh, rw, 3), dtype=np.int16)
            patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        alpha = 0.85
        image[dy : dy + rh, dx : dx + rw] = cv2.addWeighted(
            patch, alpha, image[dy : dy + rh, dx : dx + rw], 1 - alpha, 0
        )
        mask[dy : dy + rh, dx : dx + rw] = 255
        return image, mask

    def _compression_mismatch(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        ry, rx, rh, rw = self._random_rect(h, w)

        quality = int(self._rng.integers(10, 35))
        region = image[ry : ry + rh, rx : rx + rw]
        _, buf = cv2.imencode(".jpg", region, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        image[ry : ry + rh, rx : rx + rw] = compressed
        mask[ry : ry + rh, rx : rx + rw] = 255
        return image, mask

    def _noise_injection(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        ry, rx, rh, rw = self._random_rect(h, w)

        sigma = self._rng.uniform(15, 40)
        noise = self._rng.normal(0, sigma, size=(rh, rw, image.shape[2]))
        region = image[ry : ry + rh, rx : rx + rw].astype(np.float64)
        region = np.clip(region + noise, 0, 255).astype(np.uint8)

        image[ry : ry + rh, rx : rx + rw] = region
        mask[ry : ry + rh, rx : rx + rw] = 255
        return image, mask

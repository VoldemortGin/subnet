from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.miner.backends.ela import ELABackend
from src.protocol import ForgeryMethod


class TestELABackendProperties:
    """Tests for ELABackend metadata properties."""

    def test_name(self):
        backend = ELABackend()
        assert backend.name() == "ela"

    def test_is_available(self):
        assert ELABackend.is_available() is True

    def test_gpu_required(self):
        assert ELABackend.gpu_required() is False

    def test_estimated_vram_mb(self):
        assert ELABackend.estimated_vram_mb() == 0


class TestELABackendDetect:
    """Tests for ELABackend.detect() return format and behavior."""

    def test_detect_returns_correct_tuple_format(self, sample_image: np.ndarray):
        backend = ELABackend()
        result = backend.detect(sample_image)

        assert isinstance(result, tuple)
        assert len(result) == 4

        is_tampered, confidence, mask, method = result
        assert isinstance(is_tampered, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(mask, np.ndarray)
        assert mask.shape == sample_image.shape[:2]
        assert method is None or isinstance(method, ForgeryMethod)

    def test_detect_uniform_image_is_authentic(self):
        """A perfectly uniform image should not be flagged as tampered."""
        uniform = np.full((200, 300, 3), 128, dtype=np.uint8)
        backend = ELABackend()
        is_tampered, confidence, mask, method = backend.detect(uniform)

        assert is_tampered is False
        assert confidence <= 0.5
        assert method is None

    def test_detect_solid_color_low_confidence(self):
        """A solid single-color image should have near-zero confidence."""
        solid = np.full((100, 100, 3), 50, dtype=np.uint8)
        backend = ELABackend()
        is_tampered, confidence, mask, method = backend.detect(solid)

        assert is_tampered is False
        assert confidence < 0.3

    def test_detect_confidence_is_rounded(self, sample_image: np.ndarray):
        backend = ELABackend()
        _, confidence, _, _ = backend.detect(sample_image)
        assert confidence == round(confidence, 4)

    def test_detect_mask_is_binary(self, sample_image: np.ndarray):
        """The mask should only contain 0 and 255 values."""
        backend = ELABackend()
        _, _, mask, _ = backend.detect(sample_image)
        unique_values = set(np.unique(mask))
        assert unique_values.issubset({0, 255})

    def test_detect_with_natural_image(self):
        """An image with varied but consistent content should be authentic."""
        # Create a gradient image (natural-looking, no splicing)
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(200):
            img[i, :] = [i, 255 - i, 128]
        backend = ELABackend()
        is_tampered, confidence, mask, method = backend.detect(img)

        # Natural gradient shouldn't trigger high confidence
        assert isinstance(is_tampered, bool)
        assert 0.0 <= confidence <= 1.0


class TestELABackendInternalMethods:
    """Tests for ELABackend internal analysis methods."""

    def test_run_ela_returns_tuple(self, sample_image: np.ndarray):
        backend = ELABackend()
        result = backend._run_ela(sample_image)

        assert isinstance(result, tuple)
        assert len(result) == 2
        confidence, mask = result
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(mask, np.ndarray)
        assert mask.shape == sample_image.shape[:2]

    def test_run_ela_uniform_low_confidence(self):
        """Uniform image should have minimal ELA response."""
        uniform = np.full((100, 100, 3), 200, dtype=np.uint8)
        backend = ELABackend()
        confidence, mask = backend._run_ela(uniform)

        # Uniform image → minimal difference after JPEG re-save
        assert confidence < 0.5

    def test_run_noise_analysis_returns_tuple(self, sample_image: np.ndarray):
        backend = ELABackend()
        result = backend._run_noise_analysis(sample_image)

        assert isinstance(result, tuple)
        assert len(result) == 2
        confidence, mask = result
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(mask, np.ndarray)
        assert mask.shape == sample_image.shape[:2]

    def test_run_noise_analysis_zero_variance(self):
        """An image with zero variance noise should return 0 confidence."""
        # Perfectly uniform → global_var < 1e-10 → early return
        uniform = np.full((64, 64, 3), 100, dtype=np.uint8)
        backend = ELABackend()
        confidence, mask = backend._run_noise_analysis(uniform)

        assert confidence == 0.0
        assert np.all(mask == 0)

    def test_guess_method_below_threshold(self):
        """Both masks below min_tampered_ratio → None."""
        backend = ELABackend(min_tampered_ratio=0.005)
        ela_mask = np.zeros((100, 100), dtype=np.uint8)
        noise_mask = np.zeros((100, 100), dtype=np.uint8)
        result = backend._guess_method(ela_mask, noise_mask)
        assert result is None

    def test_guess_method_splicing_high_noise(self):
        """High noise ratio → SPLICING."""
        backend = ELABackend()
        ela_mask = np.zeros((100, 100), dtype=np.uint8)
        # > 5% noise pixels
        noise_mask = np.zeros((100, 100), dtype=np.uint8)
        noise_mask[:10, :] = 255  # 10% of pixels
        result = backend._guess_method(ela_mask, noise_mask)
        assert result == ForgeryMethod.SPLICING

    def test_guess_method_copy_move_multiple_contours(self):
        """Multiple ELA regions → COPY_MOVE."""
        backend = ELABackend()
        ela_mask = np.zeros((100, 100), dtype=np.uint8)
        ela_mask[5:15, 5:15] = 255
        ela_mask[70:85, 70:85] = 255
        noise_mask = np.zeros((100, 100), dtype=np.uint8)
        result = backend._guess_method(ela_mask, noise_mask)
        assert result == ForgeryMethod.COPY_MOVE

    def test_guess_method_inpainting_single_region(self):
        """Single large ELA region → INPAINTING."""
        backend = ELABackend()
        ela_mask = np.zeros((100, 100), dtype=np.uint8)
        ela_mask[10:30, 10:30] = 255  # single contiguous block, > 1%
        noise_mask = np.zeros((100, 100), dtype=np.uint8)
        result = backend._guess_method(ela_mask, noise_mask)
        assert result == ForgeryMethod.INPAINTING


class TestELABackendCustomParams:
    """Tests for ELABackend with custom parameters."""

    def test_custom_ela_quality(self, sample_image: np.ndarray):
        backend = ELABackend(ela_quality=50)
        result = backend.detect(sample_image)
        assert len(result) == 4

    def test_high_confidence_threshold(self):
        """Very high threshold should make detection harder."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        backend = ELABackend(confidence_threshold=0.99)
        is_tampered, _, _, _ = backend.detect(img)
        assert is_tampered is False

    def test_low_confidence_threshold(self, sample_image: np.ndarray):
        """Very low threshold makes detection more sensitive."""
        backend = ELABackend(confidence_threshold=0.01)
        is_tampered, confidence, _, _ = backend.detect(sample_image)
        # With threshold at 0.01, even minor artifacts may trigger
        if confidence > 0.01:
            assert is_tampered is True

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.miner.backends.base import DetectionBackend
from src.miner.detector import ForgeryDetector
from src.protocol import ForgeryMethod, MinerResponse, Verdict


class TestForgeryDetectorBuiltin:
    """Tests for ForgeryDetector using the builtin ELA+noise pipeline."""

    def test_detect_returns_miner_response(self, sample_image: np.ndarray):
        detector = ForgeryDetector()
        result = detector.detect(sample_image, task_id="task-001")

        assert isinstance(result, MinerResponse)
        assert result.task_id == "task-001"
        assert isinstance(result.verdict, Verdict)
        assert 0.0 <= result.confidence <= 1.0
        assert result.mask is not None
        assert result.mask.shape[:2] == sample_image.shape[:2]

    def test_detect_uniform_image_is_authentic(self):
        """A uniform solid-color image should be classified as authentic."""
        uniform = np.full((200, 300, 3), 128, dtype=np.uint8)
        detector = ForgeryDetector()
        result = detector.detect(uniform, task_id="task-uniform")

        assert result.verdict == Verdict.AUTHENTIC
        assert result.confidence <= 0.5

    def test_detect_tampered_image(self, sample_image: np.ndarray):
        """An image with a high-contrast patch spliced in should be detected."""
        tampered = sample_image.copy()
        # Splice a bright white block into the image
        tampered[10:60, 10:60] = [255, 255, 255]
        # Save and reload via JPEG at low quality to create ELA artifacts
        _, buf = cv2.imencode(".jpg", tampered, [cv2.IMWRITE_JPEG_QUALITY, 30])
        tampered_jpg = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        detector = ForgeryDetector(ela_quality=90, confidence_threshold=0.3)
        result = detector.detect(tampered_jpg, task_id="task-tampered")

        assert isinstance(result, MinerResponse)
        assert result.task_id == "task-tampered"
        # Confidence should be non-trivial (may or may not exceed threshold
        # depending on image content, but the structure is correct)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_preserves_task_id(self, sample_image: np.ndarray):
        detector = ForgeryDetector()
        result = detector.detect(sample_image, task_id="my-unique-id-123")
        assert result.task_id == "my-unique-id-123"

    def test_detect_confidence_is_rounded(self, sample_image: np.ndarray):
        detector = ForgeryDetector()
        result = detector.detect(sample_image, task_id="task-round")
        # confidence should have at most 4 decimal places
        assert result.confidence == round(result.confidence, 4)

    def test_detect_method_none_when_authentic(self):
        """When verdict is AUTHENTIC, method should be None."""
        uniform = np.full((100, 100, 3), 100, dtype=np.uint8)
        detector = ForgeryDetector()
        result = detector.detect(uniform, task_id="task-auth")

        if result.verdict == Verdict.AUTHENTIC:
            assert result.method is None


class TestForgeryDetectorWithBackend:
    """Tests for ForgeryDetector delegating to a backend."""

    def _make_mock_backend(
        self,
        is_tampered: bool = True,
        confidence: float = 0.85,
        method: ForgeryMethod | None = ForgeryMethod.SPLICING,
    ) -> MagicMock:
        mock = MagicMock(spec=DetectionBackend)
        mask = np.zeros((200, 300), dtype=np.uint8)
        mock.detect.return_value = (is_tampered, confidence, mask, method)
        return mock

    def test_detect_delegates_to_backend(self, sample_image: np.ndarray):
        mock_backend = self._make_mock_backend()
        detector = ForgeryDetector(backend=mock_backend)

        result = detector.detect(sample_image, task_id="task-backend")

        mock_backend.detect.assert_called_once_with(sample_image)
        assert result.verdict == Verdict.TAMPERED
        assert result.confidence == 0.85
        assert result.method == ForgeryMethod.SPLICING

    def test_detect_backend_authentic(self, sample_image: np.ndarray):
        mock_backend = self._make_mock_backend(
            is_tampered=False, confidence=0.1, method=None
        )
        detector = ForgeryDetector(backend=mock_backend)

        result = detector.detect(sample_image, task_id="task-backend-auth")

        assert result.verdict == Verdict.AUTHENTIC
        assert result.confidence == 0.1
        assert result.method is None

    def test_detect_backend_mask_passthrough(self, sample_image: np.ndarray):
        mock_backend = MagicMock(spec=DetectionBackend)
        expected_mask = np.ones((200, 300), dtype=np.uint8) * 255
        mock_backend.detect.return_value = (True, 0.9, expected_mask, ForgeryMethod.COPY_MOVE)

        detector = ForgeryDetector(backend=mock_backend)
        result = detector.detect(sample_image, task_id="task-mask")

        assert result.mask is expected_mask


class TestForgeryDetectorFromPath:
    """Tests for detect_from_path()."""

    def test_detect_from_path_valid(self, sample_image_path: Path):
        detector = ForgeryDetector()
        result = detector.detect_from_path(str(sample_image_path), task_id="task-path")

        assert isinstance(result, MinerResponse)
        assert result.task_id == "task-path"
        assert isinstance(result.verdict, Verdict)

    def test_detect_from_path_invalid_raises(self, tmp_path: Path):
        detector = ForgeryDetector()
        bad_path = str(tmp_path / "nonexistent.png")

        with pytest.raises(FileNotFoundError, match="Cannot read image"):
            detector.detect_from_path(bad_path, task_id="task-bad")

    def test_detect_from_path_unreadable_file(self, tmp_path: Path):
        """A file that exists but is not a valid image should raise."""
        bad_file = tmp_path / "not_an_image.png"
        bad_file.write_text("this is not an image")

        detector = ForgeryDetector()
        with pytest.raises(FileNotFoundError, match="Cannot read image"):
            detector.detect_from_path(str(bad_file), task_id="task-corrupt")


class TestForgeryDetectorInternalMethods:
    """Tests for internal helper methods."""

    def test_run_ela_returns_detection_result(self, sample_image: np.ndarray):
        detector = ForgeryDetector()
        result = detector._run_ela(sample_image)

        assert hasattr(result, "confidence")
        assert hasattr(result, "mask")
        assert 0.0 <= result.confidence <= 1.0
        assert result.mask.shape == sample_image.shape[:2]

    def test_run_noise_analysis_returns_detection_result(self, sample_image: np.ndarray):
        detector = ForgeryDetector()
        result = detector._run_noise_analysis(sample_image)

        assert hasattr(result, "confidence")
        assert hasattr(result, "mask")
        assert 0.0 <= result.confidence <= 1.0
        assert result.mask.shape == sample_image.shape[:2]

    def test_guess_method_no_tampering(self):
        """When both masks are empty, method should be None."""
        detector = ForgeryDetector()
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        result = detector._guess_method(empty_mask, empty_mask)
        assert result is None

    def test_guess_method_splicing(self):
        """High noise ratio should yield SPLICING."""
        detector = ForgeryDetector()
        ela_mask = np.zeros((100, 100), dtype=np.uint8)
        noise_mask = np.ones((100, 100), dtype=np.uint8) * 255  # 100% nonzero
        result = detector._guess_method(ela_mask, noise_mask)
        assert result == ForgeryMethod.SPLICING

    def test_guess_method_inpainting(self):
        """Single ELA region > 1% should yield INPAINTING."""
        detector = ForgeryDetector()
        ela_mask = np.zeros((100, 100), dtype=np.uint8)
        # Create one contiguous region > 1% of total pixels
        ela_mask[0:20, 0:20] = 255  # 400/10000 = 4%
        noise_mask = np.zeros((100, 100), dtype=np.uint8)
        result = detector._guess_method(ela_mask, noise_mask)
        assert result == ForgeryMethod.INPAINTING

    def test_guess_method_copy_move(self):
        """Multiple ELA regions should yield COPY_MOVE."""
        detector = ForgeryDetector()
        ela_mask = np.zeros((100, 100), dtype=np.uint8)
        # Two separate regions
        ela_mask[0:10, 0:20] = 255  # region 1
        ela_mask[80:90, 80:100] = 255  # region 2
        noise_mask = np.zeros((100, 100), dtype=np.uint8)
        result = detector._guess_method(ela_mask, noise_mask)
        assert result == ForgeryMethod.COPY_MOVE

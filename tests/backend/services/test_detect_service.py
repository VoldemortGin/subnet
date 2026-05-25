from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.protocol import ForgeryMethod, MinerResponse, Verdict

from backend.db.store import Store
from backend.services.detect_service import DetectService


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "uploads").mkdir()
    (tmp_path / "viz").mkdir()
    return tmp_path


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def svc(store: Store, data_dir: Path) -> DetectService:
    return DetectService(store, data_dir)


def _make_mock_response(
    task_id: str = "test",
    verdict: Verdict = Verdict.TAMPERED,
    confidence: float = 0.85,
    method: ForgeryMethod | None = ForgeryMethod.COPY_MOVE,
    mask_shape: tuple[int, int] = (100, 100),
) -> MinerResponse:
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask[20:80, 20:80] = 255
    return MinerResponse(
        task_id=task_id,
        verdict=verdict,
        confidence=confidence,
        method=method,
        mask=mask,
    )


def _make_authentic_response(task_id: str = "test") -> MinerResponse:
    return MinerResponse(
        task_id=task_id,
        verdict=Verdict.AUTHENTIC,
        confidence=0.2,
        method=None,
        mask=None,
    )


class TestAnalyzeImage:
    def test_returns_image_record_tampered(
        self, svc: DetectService, store: Store, sample_image_path: Path
    ):
        mock_resp = _make_mock_response(mask_shape=(100, 100))
        with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
            record = svc.analyze_image(sample_image_path)
        assert record.filename == sample_image_path.name
        assert record.path == str(sample_image_path)
        assert record.status == "analyzed"
        assert record.verdict == "tampered"
        assert record.confidence == 0.85
        assert record.method == "copy_move"

    def test_returns_image_record_authentic(
        self, svc: DetectService, sample_image_path: Path
    ):
        mock_resp = _make_authentic_response()
        with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
            record = svc.analyze_image(sample_image_path)
        assert record.verdict == "authentic"
        assert record.confidence == 0.2
        assert record.method is None
        assert record.mask_path is None
        assert record.visualization_path is None

    def test_stores_record_in_store(
        self, svc: DetectService, store: Store, sample_image_path: Path
    ):
        mock_resp = _make_authentic_response()
        with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
            record = svc.analyze_image(sample_image_path)
        assert record.id in store.images
        assert store.images[record.id] is record

    def test_custom_image_id(self, svc: DetectService, sample_image_path: Path):
        mock_resp = _make_authentic_response()
        with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
            record = svc.analyze_image(sample_image_path, image_id="custom_123")
        assert record.id == "custom_123"

    def test_auto_generated_id(self, svc: DetectService, sample_image_path: Path):
        mock_resp = _make_authentic_response()
        with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
            record = svc.analyze_image(sample_image_path)
        assert len(record.id) == 8

    def test_mask_saved_when_present(
        self, svc: DetectService, sample_image_path: Path, data_dir: Path
    ):
        mock_resp = _make_mock_response(mask_shape=(100, 100))
        with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
            record = svc.analyze_image(sample_image_path)
        assert record.mask_path is not None
        assert Path(record.mask_path).exists()
        mask_img = cv2.imread(record.mask_path, cv2.IMREAD_GRAYSCALE)
        assert mask_img is not None
        assert mask_img.shape == (100, 100)

    def test_visualization_saved_when_mask_present(
        self, svc: DetectService, sample_image_path: Path, data_dir: Path
    ):
        mock_resp = _make_mock_response(mask_shape=(100, 100))
        with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
            record = svc.analyze_image(sample_image_path)
        assert record.visualization_path is not None
        assert Path(record.visualization_path).exists()

    def test_no_mask_no_viz(self, svc: DetectService, sample_image_path: Path):
        mock_resp = _make_authentic_response()
        with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
            record = svc.analyze_image(sample_image_path)
        assert record.mask_path is None
        assert record.visualization_path is None

    def test_method_values(self, svc: DetectService, sample_image_path: Path):
        for method in ForgeryMethod:
            mock_resp = _make_mock_response(method=method, mask_shape=(100, 100))
            with patch.object(svc.detector, "detect_from_path", return_value=mock_resp):
                record = svc.analyze_image(sample_image_path, image_id=f"id_{method.value}")
            assert record.method == method.value


class TestCreateOverlay:
    def test_creates_output_file(self, svc: DetectService, tmp_path: Path):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "source.png"
        cv2.imwrite(str(img_path), img)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        output_path = tmp_path / "overlay.png"
        svc._create_overlay(img_path, mask, output_path)
        assert output_path.exists()

    def test_overlay_has_same_dimensions(self, svc: DetectService, tmp_path: Path):
        img = np.full((150, 200, 3), 100, dtype=np.uint8)
        img_path = tmp_path / "source.png"
        cv2.imwrite(str(img_path), img)

        mask = np.zeros((150, 200), dtype=np.uint8)
        mask[30:120, 40:160] = 255

        output_path = tmp_path / "overlay.png"
        svc._create_overlay(img_path, mask, output_path)
        overlay = cv2.imread(str(output_path))
        assert overlay.shape == (150, 200, 3)

    def test_overlay_differs_in_masked_region(self, svc: DetectService, tmp_path: Path):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "source.png"
        cv2.imwrite(str(img_path), img)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        output_path = tmp_path / "overlay.png"
        svc._create_overlay(img_path, mask, output_path)

        overlay = cv2.imread(str(output_path))
        original = cv2.imread(str(img_path))
        # Masked region should have red tint (higher red channel in BGR)
        masked_region_overlay = overlay[40, 40]
        masked_region_original = original[40, 40]
        assert masked_region_overlay[2] > masked_region_original[2]

    def test_mask_resized_if_different_dimensions(
        self, svc: DetectService, tmp_path: Path
    ):
        img = np.full((200, 300, 3), 100, dtype=np.uint8)
        img_path = tmp_path / "source.png"
        cv2.imwrite(str(img_path), img)

        # Mask with different size
        mask = np.zeros((100, 150), dtype=np.uint8)
        mask[25:75, 25:125] = 255

        output_path = tmp_path / "overlay.png"
        svc._create_overlay(img_path, mask, output_path)
        assert output_path.exists()
        overlay = cv2.imread(str(output_path))
        assert overlay.shape == (200, 300, 3)

    def test_no_output_for_unreadable_image(self, svc: DetectService, tmp_path: Path):
        bad_path = tmp_path / "nonexistent.png"
        mask = np.zeros((50, 50), dtype=np.uint8)
        output_path = tmp_path / "overlay.png"
        svc._create_overlay(bad_path, mask, output_path)
        # Should not create output if source is unreadable
        assert not output_path.exists()

    def test_unmasked_region_unchanged(self, svc: DetectService, tmp_path: Path):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "source.png"
        cv2.imwrite(str(img_path), img)

        # Only mask a small corner
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[90:100, 90:100] = 255

        output_path = tmp_path / "overlay.png"
        svc._create_overlay(img_path, mask, output_path)
        overlay = cv2.imread(str(output_path))
        original = cv2.imread(str(img_path))
        # An unmasked region should be identical
        np.testing.assert_array_equal(overlay[0:10, 0:10], original[0:10, 0:10])

    def test_full_mask_all_tinted(self, svc: DetectService, tmp_path: Path):
        img = np.full((80, 80, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "source.png"
        cv2.imwrite(str(img_path), img)

        mask = np.full((80, 80), 255, dtype=np.uint8)  # All non-zero
        output_path = tmp_path / "overlay.png"
        svc._create_overlay(img_path, mask, output_path)
        assert output_path.exists()
        overlay = cv2.imread(str(output_path))
        original = cv2.imread(str(img_path))
        # Every pixel should be tinted red
        assert overlay[40, 40, 2] > original[40, 40, 2]

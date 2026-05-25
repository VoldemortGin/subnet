from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.protocol import ForgeryMethod, Verdict
from src.validator.forge import ForgeEngine


class TestForgeEngineInit:
    def test_deterministic_with_seed(self, sample_image: np.ndarray):
        engine1 = ForgeEngine(seed=42)
        engine2 = ForgeEngine(seed=42)
        t1, m1, method1 = engine1.forge(sample_image)
        t2, m2, method2 = engine2.forge(sample_image)
        assert method1 == method2
        assert np.array_equal(t1, t2)
        assert np.array_equal(m1, m2)

    def test_different_seeds_differ(self, sample_image: np.ndarray):
        engine1 = ForgeEngine(seed=1)
        engine2 = ForgeEngine(seed=99)
        _, m1, _ = engine1.forge(sample_image)
        _, m2, _ = engine2.forge(sample_image)
        # Very unlikely to produce identical masks with different seeds
        assert not np.array_equal(m1, m2)

    def test_no_seed_runs_without_error(self, sample_image: np.ndarray):
        engine = ForgeEngine(seed=None)
        tampered, mask, method = engine.forge(sample_image)
        assert tampered.shape == sample_image.shape


class TestForgeMethod:
    """Test each tampering method independently."""

    @pytest.fixture
    def engine(self) -> ForgeEngine:
        return ForgeEngine(seed=123)

    def _validate_output(
        self, original: np.ndarray, tampered: np.ndarray, mask: np.ndarray
    ):
        """Common assertions for all forge methods."""
        assert tampered.shape == original.shape, "Tampered shape must match input"
        h, w = original.shape[:2]
        assert mask.shape == (h, w), "Mask must be 2D with (h, w)"
        assert mask.dtype == np.uint8
        assert np.any(mask > 0), "Mask must have non-zero pixels"
        unique_vals = set(np.unique(mask))
        assert unique_vals <= {0, 255}, f"Mask must only contain 0 or 255, got {unique_vals}"

    def test_copy_move(self, engine: ForgeEngine, sample_image: np.ndarray):
        tampered, mask, method = engine.forge(sample_image, method=ForgeryMethod.COPY_MOVE)
        assert method == ForgeryMethod.COPY_MOVE
        self._validate_output(sample_image, tampered, mask)

    def test_splicing_no_donor(self, engine: ForgeEngine, sample_image: np.ndarray):
        tampered, mask, method = engine.forge(sample_image, method=ForgeryMethod.SPLICING)
        assert method == ForgeryMethod.SPLICING
        self._validate_output(sample_image, tampered, mask)

    def test_splicing_with_donor(self, engine: ForgeEngine, sample_image: np.ndarray):
        donor = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
        tampered, mask, method = engine.forge(
            sample_image, method=ForgeryMethod.SPLICING, donor_image=donor
        )
        assert method == ForgeryMethod.SPLICING
        self._validate_output(sample_image, tampered, mask)

    def test_compression_mismatch(self, engine: ForgeEngine, sample_image: np.ndarray):
        tampered, mask, method = engine.forge(
            sample_image, method=ForgeryMethod.COMPRESSION
        )
        assert method == ForgeryMethod.COMPRESSION
        self._validate_output(sample_image, tampered, mask)

    def test_noise_injection(self, engine: ForgeEngine, sample_image: np.ndarray):
        tampered, mask, method = engine.forge(
            sample_image, method=ForgeryMethod.INPAINTING
        )
        assert method == ForgeryMethod.INPAINTING
        self._validate_output(sample_image, tampered, mask)

    def test_pixels_are_modified(self, engine: ForgeEngine, sample_image: np.ndarray):
        """Tampered region should differ from the original."""
        tampered, mask, _ = engine.forge(sample_image, method=ForgeryMethod.COMPRESSION)
        mask_bool = mask.astype(bool)
        original_region = sample_image[mask_bool]
        tampered_region = tampered[mask_bool]
        # Compression changes pixel values
        assert not np.array_equal(original_region, tampered_region)


class TestForgeInputTypes:
    @pytest.fixture
    def engine(self) -> ForgeEngine:
        return ForgeEngine(seed=7)

    def test_accepts_ndarray(self, engine: ForgeEngine, sample_image: np.ndarray):
        tampered, mask, _ = engine.forge(sample_image)
        assert tampered.shape == sample_image.shape

    def test_accepts_path(self, engine: ForgeEngine, sample_image_path: Path):
        tampered, mask, _ = engine.forge(sample_image_path)
        assert tampered is not None
        assert mask is not None

    def test_accepts_str_path(self, engine: ForgeEngine, sample_image_path: Path):
        tampered, mask, _ = engine.forge(str(sample_image_path))
        assert tampered is not None

    def test_invalid_path_raises(self, engine: ForgeEngine):
        with pytest.raises(FileNotFoundError):
            engine.forge("/nonexistent/path/to/image.png")

    def test_does_not_mutate_input(self, engine: ForgeEngine, sample_image: np.ndarray):
        original_copy = sample_image.copy()
        engine.forge(sample_image, method=ForgeryMethod.COPY_MOVE)
        assert np.array_equal(sample_image, original_copy)


class TestRandomRect:
    def test_within_bounds(self):
        engine = ForgeEngine(seed=0)
        h, w = 200, 300
        for _ in range(100):
            y, x, rh, rw = engine._random_rect(h, w)
            assert 0 <= y < h
            assert 0 <= x < w
            assert y + rh <= h
            assert x + rw <= w

    def test_respects_fraction_bounds(self):
        engine = ForgeEngine(seed=5)
        h, w = 1000, 1000
        for _ in range(50):
            y, x, rh, rw = engine._random_rect(h, w, min_frac=0.1, max_frac=0.3)
            assert rh >= int(h * 0.1) - 1  # allow int rounding
            assert rh <= int(h * 0.3) + 1
            assert rw >= int(w * 0.1) - 1
            assert rw <= int(w * 0.3) + 1

    def test_custom_fractions(self):
        engine = ForgeEngine(seed=10)
        h, w = 500, 500
        y, x, rh, rw = engine._random_rect(h, w, min_frac=0.4, max_frac=0.5)
        assert rh >= 190  # 500 * 0.4 = 200, with rounding tolerance
        assert rh <= 255  # 500 * 0.5 = 250, with rounding tolerance


class TestGenerateProbe:
    def test_generates_probe_task(self, sample_image_path: Path, output_dir: Path):
        engine = ForgeEngine(seed=42)
        probe = engine.generate_probe(sample_image_path, output_dir)
        assert probe.original_image_path == sample_image_path
        assert probe.tampered_image_path.exists()
        assert probe.ground_truth.verdict == Verdict.TAMPERED
        assert probe.ground_truth.method is not None
        assert probe.ground_truth.mask is not None

    def test_output_files_are_valid_images(
        self, sample_image_path: Path, output_dir: Path
    ):
        engine = ForgeEngine(seed=42)
        probe = engine.generate_probe(sample_image_path, output_dir)
        tampered_img = cv2.imread(str(probe.tampered_image_path))
        assert tampered_img is not None
        assert tampered_img.shape[0] > 0

        mask_path = output_dir / f"{sample_image_path.stem}_mask.png"
        assert mask_path.exists()
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask_img is not None

    def test_specific_method(self, sample_image_path: Path, output_dir: Path):
        engine = ForgeEngine(seed=42)
        probe = engine.generate_probe(
            sample_image_path, output_dir, method=ForgeryMethod.COPY_MOVE
        )
        assert probe.ground_truth.method == ForgeryMethod.COPY_MOVE

    def test_creates_output_dir_if_missing(self, sample_image_path: Path, tmp_path: Path):
        new_dir = tmp_path / "nested" / "output"
        engine = ForgeEngine(seed=42)
        probe = engine.generate_probe(sample_image_path, new_dir)
        assert new_dir.exists()
        assert probe.tampered_image_path.exists()

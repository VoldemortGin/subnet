from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.db.store import Store
from backend.services.forge_service import ForgeService


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "clean").mkdir()
    (tmp_path / "probes").mkdir()
    return tmp_path


@pytest.fixture
def svc(store: Store, data_dir: Path) -> ForgeService:
    return ForgeService(store, data_dir)


class TestListCleanImages:
    def test_empty_dir(self, svc: ForgeService, data_dir: Path):
        # clean dir exists but is empty
        assert svc.list_clean_images() == []

    def test_returns_png_files(self, svc: ForgeService, data_dir: Path):
        clean_dir = data_dir / "clean"
        for i in range(3):
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(clean_dir / f"img_{i}.png"), img)
        result = svc.list_clean_images()
        assert len(result) == 3
        assert all(p.suffix == ".png" for p in result)

    def test_ignores_non_png_files(self, svc: ForgeService, data_dir: Path):
        clean_dir = data_dir / "clean"
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(clean_dir / "good.png"), img)
        cv2.imwrite(str(clean_dir / "bad.jpg"), img)
        (clean_dir / "readme.txt").write_text("ignore me")
        result = svc.list_clean_images()
        assert len(result) == 1
        assert result[0].name == "good.png"

    def test_sorted_output(self, svc: ForgeService, data_dir: Path):
        clean_dir = data_dir / "clean"
        for name in ["c.png", "a.png", "b.png"]:
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(clean_dir / name), img)
        result = svc.list_clean_images()
        assert [p.name for p in result] == ["a.png", "b.png", "c.png"]

    def test_missing_clean_dir(self, store: Store, tmp_path: Path):
        # data_dir without clean/ subdirectory
        svc = ForgeService(store, tmp_path / "nonexistent")
        assert svc.list_clean_images() == []


class TestGenerateProbe:
    def test_creates_task_record(self, svc: ForgeService, store: Store, data_dir: Path):
        clean_dir = data_dir / "clean"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        clean_path = clean_dir / "test_img.png"
        cv2.imwrite(str(clean_path), img)

        task = svc.generate_probe(clean_path)
        assert task.task_type == "probe"
        assert task.status == "pending"
        assert task.ground_truth_verdict == "tampered"
        assert task.ground_truth_method is not None
        assert task.id in store.tasks

    def test_creates_image_record(self, svc: ForgeService, store: Store, data_dir: Path):
        clean_dir = data_dir / "clean"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        clean_path = clean_dir / "test_img.png"
        cv2.imwrite(str(clean_path), img)

        task = svc.generate_probe(clean_path)
        image_record = store.images[task.image_id]
        assert image_record.verdict == "tampered"
        assert image_record.confidence == 1.0
        assert image_record.status == "probe"
        assert image_record.filename.endswith(".png")

    def test_creates_output_files(self, svc: ForgeService, data_dir: Path):
        clean_dir = data_dir / "clean"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        clean_path = clean_dir / "probe_src.png"
        cv2.imwrite(str(clean_path), img)

        svc.generate_probe(clean_path)
        probes_dir = data_dir / "probes"
        tampered_files = list(probes_dir.glob("*_tampered.png"))
        mask_files = list(probes_dir.glob("*_mask.png"))
        assert len(tampered_files) >= 1
        assert len(mask_files) >= 1

    def test_initializes_submission_list(
        self, svc: ForgeService, store: Store, data_dir: Path
    ):
        clean_dir = data_dir / "clean"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        clean_path = clean_dir / "test_img.png"
        cv2.imwrite(str(clean_path), img)

        task = svc.generate_probe(clean_path)
        assert task.id in store.submissions
        assert store.submissions[task.id] == []

    def test_probes_dir_created_if_missing(self, store: Store, tmp_path: Path):
        data_dir = tmp_path / "fresh_data"
        data_dir.mkdir()
        (data_dir / "clean").mkdir()
        # probes dir does not exist yet
        svc = ForgeService(store, data_dir)
        img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        clean_path = data_dir / "clean" / "x.png"
        cv2.imwrite(str(clean_path), img)

        task = svc.generate_probe(clean_path)
        assert (data_dir / "probes").exists()
        assert task.task_type == "probe"

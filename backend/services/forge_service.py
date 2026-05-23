from __future__ import annotations

import uuid
from pathlib import Path

from src.validator.forge import ForgeEngine

from backend.db.store import Store, TaskRecord


class ForgeService:
    def __init__(self, store: Store, data_dir: Path):
        self.store = store
        self.data_dir = data_dir
        self.probes_dir = data_dir / "probes"
        self.clean_dir = data_dir / "clean"
        self.engine = ForgeEngine()

    def generate_probe(self, clean_image_path: Path) -> TaskRecord:
        self.probes_dir.mkdir(parents=True, exist_ok=True)

        probe = self.engine.generate_probe(clean_image_path, self.probes_dir)

        image_id = str(uuid.uuid4())[:8]
        task_id = str(uuid.uuid4())[:8]

        from backend.db.store import ImageRecord
        image_record = ImageRecord(
            id=image_id,
            filename=probe.tampered_image_path.name,
            path=str(probe.tampered_image_path),
            verdict="tampered",
            confidence=1.0,
            method=probe.ground_truth.method.value if probe.ground_truth.method else None,
            status="probe",
        )
        self.store.images[image_id] = image_record

        gt = probe.ground_truth
        mask_path = str(self.probes_dir / f"{clean_image_path.stem}_mask.png")

        task_record = TaskRecord(
            id=task_id,
            image_id=image_id,
            task_type="probe",
            ground_truth_verdict=gt.verdict.value,
            ground_truth_method=gt.method.value if gt.method else None,
            ground_truth_mask_path=mask_path,
            status="pending",
        )
        self.store.tasks[task_id] = task_record
        self.store.submissions.setdefault(task_id, [])

        return task_record

    def list_clean_images(self) -> list[Path]:
        if not self.clean_dir.exists():
            return []
        return sorted(self.clean_dir.glob("*.png"))

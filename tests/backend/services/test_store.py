from __future__ import annotations

from datetime import datetime

from backend.db.store import (
    ImageRecord,
    MinerRecord,
    Store,
    SubmissionRecord,
    TaskRecord,
)


class TestMinerRecord:
    def test_defaults(self):
        record = MinerRecord(id="m1", name="Alice", backend_name="ela")
        assert record.id == "m1"
        assert record.name == "Alice"
        assert record.backend_name == "ela"
        assert record.probe_history == []
        assert record.probe_scores == []
        assert record.consensus_scores == []
        assert record.total_score == 0.0
        assert record.avg_latency_ms == 0.0
        assert isinstance(record.created_at, datetime)

    def test_mutable_defaults_are_independent(self):
        r1 = MinerRecord(id="m1", name="A", backend_name="ela")
        r2 = MinerRecord(id="m2", name="B", backend_name="ela")
        r1.probe_history.append(True)
        assert r2.probe_history == []


class TestImageRecord:
    def test_defaults(self):
        record = ImageRecord(id="i1", filename="test.png", path="/tmp/test.png")
        assert record.id == "i1"
        assert record.filename == "test.png"
        assert record.path == "/tmp/test.png"
        assert record.verdict is None
        assert record.confidence is None
        assert record.method is None
        assert record.mask_path is None
        assert record.visualization_path is None
        assert record.status == "pending"
        assert isinstance(record.upload_time, datetime)

    def test_custom_fields(self):
        record = ImageRecord(
            id="i2",
            filename="forged.png",
            path="/data/forged.png",
            verdict="tampered",
            confidence=0.95,
            method="copy_move",
            status="analyzed",
        )
        assert record.verdict == "tampered"
        assert record.confidence == 0.95
        assert record.method == "copy_move"
        assert record.status == "analyzed"


class TestTaskRecord:
    def test_defaults(self):
        record = TaskRecord(id="t1", image_id="i1", task_type="real")
        assert record.id == "t1"
        assert record.image_id == "i1"
        assert record.task_type == "real"
        assert record.ground_truth_verdict is None
        assert record.ground_truth_method is None
        assert record.ground_truth_mask_path is None
        assert record.status == "pending"
        assert isinstance(record.created_at, datetime)

    def test_probe_task(self):
        record = TaskRecord(
            id="t2",
            image_id="i2",
            task_type="probe",
            ground_truth_verdict="tampered",
            ground_truth_method="splicing",
            ground_truth_mask_path="/masks/t2_mask.png",
        )
        assert record.task_type == "probe"
        assert record.ground_truth_verdict == "tampered"
        assert record.ground_truth_method == "splicing"
        assert record.ground_truth_mask_path == "/masks/t2_mask.png"


class TestSubmissionRecord:
    def test_defaults(self):
        record = SubmissionRecord(
            miner_id="m1",
            task_id="t1",
            verdict="tampered",
            confidence=0.8,
        )
        assert record.miner_id == "m1"
        assert record.task_id == "t1"
        assert record.verdict == "tampered"
        assert record.confidence == 0.8
        assert record.mask_path is None
        assert record.latency_ms == 0.0
        assert record.score == 0.0
        assert record.committed_hash is None
        assert isinstance(record.created_at, datetime)


class TestStore:
    def test_empty_store(self):
        store = Store()
        assert store.images == {}
        assert store.tasks == {}
        assert store.miners == {}
        assert store.submissions == {}

    def test_add_miner(self):
        store = Store()
        miner = MinerRecord(id="m1", name="Miner1", backend_name="ela")
        store.miners["m1"] = miner
        assert "m1" in store.miners
        assert store.miners["m1"].name == "Miner1"

    def test_add_image(self):
        store = Store()
        img = ImageRecord(id="i1", filename="img.png", path="/tmp/img.png")
        store.images["i1"] = img
        assert store.images["i1"].filename == "img.png"

    def test_add_task(self):
        store = Store()
        task = TaskRecord(id="t1", image_id="i1", task_type="real")
        store.tasks["t1"] = task
        assert store.tasks["t1"].task_type == "real"

    def test_add_submissions(self):
        store = Store()
        store.submissions.setdefault("t1", [])
        sub = SubmissionRecord(
            miner_id="m1", task_id="t1", verdict="authentic", confidence=0.3
        )
        store.submissions["t1"].append(sub)
        assert len(store.submissions["t1"]) == 1
        assert store.submissions["t1"][0].verdict == "authentic"

    def test_multiple_submissions_per_task(self):
        store = Store()
        store.submissions.setdefault("t1", [])
        for i in range(5):
            sub = SubmissionRecord(
                miner_id=f"m{i}", task_id="t1", verdict="tampered", confidence=0.9
            )
            store.submissions["t1"].append(sub)
        assert len(store.submissions["t1"]) == 5

    def test_stores_are_independent(self):
        s1 = Store()
        s2 = Store()
        s1.miners["m1"] = MinerRecord(id="m1", name="A", backend_name="ela")
        assert "m1" not in s2.miners

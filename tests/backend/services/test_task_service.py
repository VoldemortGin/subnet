from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.db.store import Store, TaskRecord
from backend.services.miner_service import MinerService
from backend.services.task_service import TaskService


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def miner_svc(store: Store) -> MinerService:
    return MinerService(store)


@pytest.fixture
def forge_svc_mock():
    mock = MagicMock()
    mock.list_clean_images.return_value = []
    return mock


@pytest.fixture
def detect_svc_mock():
    return MagicMock()


@pytest.fixture
def task_svc(
    store: Store,
    forge_svc_mock: MagicMock,
    detect_svc_mock: MagicMock,
    miner_svc: MinerService,
) -> TaskService:
    return TaskService(store, forge_svc_mock, detect_svc_mock, miner_svc)


class TestCreateRealTask:
    def test_creates_task(self, task_svc: TaskService, store: Store):
        task = task_svc.create_real_task("img_001")
        assert task.image_id == "img_001"
        assert task.task_type == "real"
        assert task.status == "pending"
        assert len(task.id) == 8

    def test_stores_in_store(self, task_svc: TaskService, store: Store):
        task = task_svc.create_real_task("img_002")
        assert task.id in store.tasks
        assert store.tasks[task.id] is task

    def test_initializes_submission_list(self, task_svc: TaskService, store: Store):
        task = task_svc.create_real_task("img_003")
        assert task.id in store.submissions
        assert store.submissions[task.id] == []

    def test_no_ground_truth_for_real_task(self, task_svc: TaskService):
        task = task_svc.create_real_task("img_004")
        assert task.ground_truth_verdict is None
        assert task.ground_truth_method is None
        assert task.ground_truth_mask_path is None


class TestCreateProbeTask:
    def test_raises_when_no_clean_images(
        self, task_svc: TaskService, forge_svc_mock: MagicMock
    ):
        forge_svc_mock.list_clean_images.return_value = []
        with pytest.raises(RuntimeError, match="No clean images"):
            task_svc.create_probe_task()

    def test_calls_forge_service(
        self, task_svc: TaskService, forge_svc_mock: MagicMock
    ):
        fake_task = TaskRecord(
            id="probe1",
            image_id="img_forged",
            task_type="probe",
            ground_truth_verdict="tampered",
            ground_truth_method="copy_move",
        )
        forge_svc_mock.list_clean_images.return_value = [Path("/clean/a.png")]
        forge_svc_mock.generate_probe.return_value = fake_task
        result = task_svc.create_probe_task()
        assert result is fake_task
        forge_svc_mock.generate_probe.assert_called_once()

    def test_picks_from_available_images(
        self, task_svc: TaskService, forge_svc_mock: MagicMock
    ):
        paths = [Path(f"/clean/img_{i}.png") for i in range(5)]
        forge_svc_mock.list_clean_images.return_value = paths
        fake_task = TaskRecord(id="p1", image_id="x", task_type="probe")
        forge_svc_mock.generate_probe.return_value = fake_task
        task_svc.create_probe_task()
        call_arg = forge_svc_mock.generate_probe.call_args[0][0]
        assert call_arg in paths


class TestSubmitResult:
    def test_raises_for_unknown_task(self, task_svc: TaskService, miner_svc: MinerService):
        miner = miner_svc.register_miner("A", "ela")
        with pytest.raises(ValueError, match="Task .* not found"):
            task_svc.submit_result("bad_task", miner.id, "tampered", 0.9)

    def test_raises_for_unknown_miner(self, task_svc: TaskService, store: Store):
        task = TaskRecord(id="t1", image_id="i1", task_type="real", status="pending")
        store.tasks["t1"] = task
        with pytest.raises(ValueError, match="Miner .* not found"):
            task_svc.submit_result("t1", "bad_miner", "tampered", 0.9)

    def test_real_task_submission(
        self, task_svc: TaskService, store: Store, miner_svc: MinerService
    ):
        miner = miner_svc.register_miner("A", "ela")
        task = task_svc.create_real_task("img_x")
        sub = task_svc.submit_result(task.id, miner.id, "authentic", 0.7)
        assert sub.miner_id == miner.id
        assert sub.task_id == task.id
        assert sub.verdict == "authentic"
        assert sub.confidence == 0.7
        assert sub.score == 0.0  # No scoring for real tasks
        assert task.status == "completed"

    def test_real_task_does_not_record_probe_result(
        self, task_svc: TaskService, miner_svc: MinerService
    ):
        miner = miner_svc.register_miner("A", "ela")
        task = task_svc.create_real_task("img_x")
        task_svc.submit_result(task.id, miner.id, "tampered", 0.8)
        # No probe results recorded for real tasks
        assert miner.probe_history == []

    def test_probe_task_correct_verdict_scores_nonzero(
        self, task_svc: TaskService, store: Store, miner_svc: MinerService
    ):
        miner = miner_svc.register_miner("A", "ela")
        task = TaskRecord(
            id="pt1",
            image_id="img_probe",
            task_type="probe",
            ground_truth_verdict="tampered",
            ground_truth_method=None,
            status="pending",
        )
        store.tasks["pt1"] = task
        store.submissions.setdefault("pt1", [])
        sub = task_svc.submit_result("pt1", miner.id, "tampered", 1.0)
        # Correct verdict with high confidence should get a positive score
        assert sub.score > 0.0
        assert miner.probe_history == [True]

    def test_probe_task_wrong_verdict_scores_zero(
        self, task_svc: TaskService, store: Store, miner_svc: MinerService
    ):
        miner = miner_svc.register_miner("A", "ela")
        task = TaskRecord(
            id="pt2",
            image_id="img_probe",
            task_type="probe",
            ground_truth_verdict="tampered",
            status="pending",
        )
        store.tasks["pt2"] = task
        store.submissions.setdefault("pt2", [])
        sub = task_svc.submit_result("pt2", miner.id, "authentic", 0.9)
        assert sub.score == 0.0
        assert miner.probe_history == [False]

    def test_submission_stored(
        self, task_svc: TaskService, store: Store, miner_svc: MinerService
    ):
        miner = miner_svc.register_miner("A", "ela")
        task = task_svc.create_real_task("img_x")
        sub = task_svc.submit_result(task.id, miner.id, "tampered", 0.5)
        assert sub in store.submissions[task.id]

    def test_multiple_submissions_per_task(
        self, task_svc: TaskService, store: Store, miner_svc: MinerService
    ):
        m1 = miner_svc.register_miner("A", "ela")
        m2 = miner_svc.register_miner("B", "ela")
        task = task_svc.create_real_task("img_x")
        task_svc.submit_result(task.id, m1.id, "tampered", 0.9)
        task_svc.submit_result(task.id, m2.id, "authentic", 0.6)
        assert len(store.submissions[task.id]) == 2


class TestListTasks:
    def test_empty(self, task_svc: TaskService):
        assert task_svc.list_tasks() == []

    def test_returns_all_tasks(self, task_svc: TaskService):
        task_svc.create_real_task("i1")
        task_svc.create_real_task("i2")
        assert len(task_svc.list_tasks()) == 2

    def test_filter_by_task_type(self, task_svc: TaskService, store: Store):
        task_svc.create_real_task("i1")
        probe = TaskRecord(id="p1", image_id="i2", task_type="probe", status="pending")
        store.tasks["p1"] = probe
        real_tasks = task_svc.list_tasks(task_type="real")
        assert len(real_tasks) == 1
        assert real_tasks[0].task_type == "real"
        probe_tasks = task_svc.list_tasks(task_type="probe")
        assert len(probe_tasks) == 1
        assert probe_tasks[0].task_type == "probe"

    def test_filter_by_status(
        self, task_svc: TaskService, store: Store, miner_svc: MinerService
    ):
        miner = miner_svc.register_miner("A", "ela")
        t1 = task_svc.create_real_task("i1")
        t2 = task_svc.create_real_task("i2")
        task_svc.submit_result(t1.id, miner.id, "authentic", 0.5)
        pending = task_svc.list_tasks(status="pending")
        completed = task_svc.list_tasks(status="completed")
        assert len(pending) == 1
        assert pending[0].id == t2.id
        assert len(completed) == 1
        assert completed[0].id == t1.id

    def test_combined_filters(self, task_svc: TaskService, store: Store):
        task_svc.create_real_task("i1")
        probe = TaskRecord(
            id="p1", image_id="i2", task_type="probe", status="completed"
        )
        store.tasks["p1"] = probe
        result = task_svc.list_tasks(task_type="probe", status="completed")
        assert len(result) == 1
        assert result[0].id == "p1"

    def test_sorted_by_created_at_descending(self, task_svc: TaskService):
        import time
        t1 = task_svc.create_real_task("i1")
        time.sleep(0.01)
        t2 = task_svc.create_real_task("i2")
        tasks = task_svc.list_tasks()
        assert tasks[0].id == t2.id
        assert tasks[1].id == t1.id

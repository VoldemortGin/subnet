from __future__ import annotations

import hashlib

from backend.db.store import get_store
from backend.services.miner_service import MinerService


def _compute_commit_hash(
    verdict: str, confidence: float, method: str | None, nonce: str
) -> str:
    method_str = method or ""
    payload = f"{verdict}|{confidence}|{method_str}|{nonce}"
    return hashlib.sha256(payload.encode()).hexdigest()


class TestTaskList:
    def test_list_empty(self, client):
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_probe_creation(self, client, probe_task):
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == probe_task["id"]
        assert data[0]["task_type"] == "probe"

    def test_list_filter_by_task_type(self, client, probe_task):
        resp = client.get("/api/tasks", params={"task_type": "probe"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        resp = client.get("/api/tasks", params={"task_type": "real"})
        assert resp.status_code == 200
        assert len(resp.json()) == 0

    def test_list_filter_by_status(self, client, probe_task):
        resp = client.get("/api/tasks", params={"status": "pending"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        resp = client.get("/api/tasks", params={"status": "completed"})
        assert resp.status_code == 200
        assert len(resp.json()) == 0


class TestCreateProbe:
    def test_create_probe_success(self, client):
        resp = client.post("/api/tasks/probe")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_type"] == "probe"
        assert data["status"] == "pending"
        assert data["ground_truth_verdict"] == "tampered"
        assert data["ground_truth_method"] is not None
        assert "id" in data
        assert "image_id" in data
        assert "created_at" in data

    def test_create_probe_no_clean_images(self, client, monkeypatch, tmp_path):
        # Override DATA_DIR to a location with no clean images
        empty_data = tmp_path / "empty_data"
        empty_data.mkdir()
        (empty_data / "clean").mkdir()
        (empty_data / "probes").mkdir()
        monkeypatch.setattr("backend.api.routes.tasks.DATA_DIR", empty_data)

        resp = client.post("/api/tasks/probe")
        assert resp.status_code == 500
        assert "No clean images" in resp.json()["detail"]


class TestGetTask:
    def test_get_existing_task(self, client, probe_task):
        task_id = probe_task["id"]
        resp = client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"]["id"] == task_id
        assert data["submissions"] == []

    def test_get_nonexistent_task_returns_404(self, client):
        resp = client.get("/api/tasks/no-such-id")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Task not found"


class TestSubmitResult:
    def test_submit_success(self, client, registered_miner, probe_task):
        resp = client.post(f"/api/tasks/{probe_task['id']}/submit", json={
            "miner_id": registered_miner["id"],
            "verdict": "tampered",
            "confidence": 0.95,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["miner_id"] == registered_miner["id"]
        assert data["task_id"] == probe_task["id"]
        assert data["verdict"] == "tampered"
        assert data["confidence"] == 0.95
        assert data["score"] >= 0.0
        assert "created_at" in data

    def test_submit_nonexistent_task(self, client, registered_miner):
        resp = client.post("/api/tasks/fake-task/submit", json={
            "miner_id": registered_miner["id"],
            "verdict": "tampered",
            "confidence": 0.9,
        })
        assert resp.status_code == 404

    def test_submit_nonexistent_miner(self, client, probe_task):
        resp = client.post(f"/api/tasks/{probe_task['id']}/submit", json={
            "miner_id": "fake-miner",
            "verdict": "tampered",
            "confidence": 0.9,
        })
        assert resp.status_code == 404

    def test_submit_correct_verdict_gives_positive_score(
        self, client, registered_miner, probe_task
    ):
        # Probe tasks have ground_truth_verdict = "tampered"
        resp = client.post(f"/api/tasks/{probe_task['id']}/submit", json={
            "miner_id": registered_miner["id"],
            "verdict": "tampered",
            "confidence": 1.0,
        })
        assert resp.status_code == 200
        assert resp.json()["score"] > 0.0

    def test_submit_wrong_verdict_gives_zero_score(
        self, client, registered_miner, probe_task
    ):
        resp = client.post(f"/api/tasks/{probe_task['id']}/submit", json={
            "miner_id": registered_miner["id"],
            "verdict": "authentic",
            "confidence": 0.9,
        })
        assert resp.status_code == 200
        assert resp.json()["score"] == 0.0


class TestCommitReveal:
    def test_commit_success(self, client, registered_miner, probe_task):
        commit_hash = _compute_commit_hash("tampered", 0.95, None, "secret123")
        resp = client.post(f"/api/tasks/{probe_task['id']}/commit", json={
            "miner_id": registered_miner["id"],
            "hash": commit_hash,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["message"] == "Commitment recorded"

    def test_commit_nonexistent_task(self, client, registered_miner):
        resp = client.post("/api/tasks/fake/commit", json={
            "miner_id": registered_miner["id"],
            "hash": "abc123",
        })
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Task not found"

    def test_commit_nonexistent_miner(self, client, probe_task):
        resp = client.post(f"/api/tasks/{probe_task['id']}/commit", json={
            "miner_id": "fake-miner",
            "hash": "abc123",
        })
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Miner not found"

    def test_reveal_valid_hash(self, client, registered_miner, probe_task):
        task_id = probe_task["id"]
        miner_id = registered_miner["id"]
        verdict = "tampered"
        confidence = 0.95
        method = None
        nonce = "my-secret-nonce"

        commit_hash = _compute_commit_hash(verdict, confidence, method, nonce)
        client.post(f"/api/tasks/{task_id}/commit", json={
            "miner_id": miner_id,
            "hash": commit_hash,
        })

        resp = client.post(f"/api/tasks/{task_id}/reveal", json={
            "miner_id": miner_id,
            "verdict": verdict,
            "confidence": confidence,
            "method": method,
            "nonce": nonce,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hash_valid"] is True
        assert data["score"] > 0.0
        assert data["is_probe"] is True
        assert data["ground_truth"] is not None
        assert data["ground_truth"]["verdict"] == "tampered"
        assert "strike_status" in data
        assert isinstance(data["probe_history"], list)

    def test_reveal_invalid_hash(self, client, registered_miner, probe_task):
        task_id = probe_task["id"]
        miner_id = registered_miner["id"]

        # Commit with one hash
        commit_hash = _compute_commit_hash("tampered", 0.95, None, "nonce1")
        client.post(f"/api/tasks/{task_id}/commit", json={
            "miner_id": miner_id,
            "hash": commit_hash,
        })

        # Reveal with a different nonce -> hash mismatch
        resp = client.post(f"/api/tasks/{task_id}/reveal", json={
            "miner_id": miner_id,
            "verdict": "tampered",
            "confidence": 0.95,
            "method": None,
            "nonce": "wrong-nonce",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hash_valid"] is False
        assert data["score"] == 0.0

    def test_reveal_with_method(self, client, registered_miner, probe_task):
        task_id = probe_task["id"]
        miner_id = registered_miner["id"]
        verdict = "tampered"
        confidence = 0.85
        method = "copy_move"
        nonce = "nonce-with-method"

        commit_hash = _compute_commit_hash(verdict, confidence, method, nonce)
        client.post(f"/api/tasks/{task_id}/commit", json={
            "miner_id": miner_id,
            "hash": commit_hash,
        })

        resp = client.post(f"/api/tasks/{task_id}/reveal", json={
            "miner_id": miner_id,
            "verdict": verdict,
            "confidence": confidence,
            "method": method,
            "nonce": nonce,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hash_valid"] is True
        assert data["score"] > 0.0

    def test_reveal_nonexistent_task(self, client, registered_miner):
        resp = client.post("/api/tasks/fake/reveal", json={
            "miner_id": registered_miner["id"],
            "verdict": "tampered",
            "confidence": 0.9,
            "nonce": "x",
        })
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Task not found"

    def test_reveal_nonexistent_miner(self, client, probe_task):
        resp = client.post(f"/api/tasks/{probe_task['id']}/reveal", json={
            "miner_id": "fake-miner",
            "verdict": "tampered",
            "confidence": 0.9,
            "nonce": "x",
        })
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Miner not found"

    def test_reveal_no_commitment(self, client, registered_miner, probe_task):
        """Reveal without a prior commit returns 400."""
        resp = client.post(f"/api/tasks/{probe_task['id']}/reveal", json={
            "miner_id": registered_miner["id"],
            "verdict": "tampered",
            "confidence": 0.9,
            "nonce": "x",
        })
        assert resp.status_code == 400
        assert "No commitment found" in resp.json()["detail"]

    def test_full_commit_reveal_flow_updates_probe_history(
        self, client, registered_miner, probe_task
    ):
        """Full flow: commit -> reveal -> check miner's probe history updated."""
        task_id = probe_task["id"]
        miner_id = registered_miner["id"]
        verdict = "tampered"
        confidence = 1.0
        nonce = "flow-nonce"

        commit_hash = _compute_commit_hash(verdict, confidence, None, nonce)
        client.post(f"/api/tasks/{task_id}/commit", json={
            "miner_id": miner_id,
            "hash": commit_hash,
        })

        resp = client.post(f"/api/tasks/{task_id}/reveal", json={
            "miner_id": miner_id,
            "verdict": verdict,
            "confidence": confidence,
            "method": None,
            "nonce": nonce,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hash_valid"] is True
        # Miner got correct verdict on a probe, should have True in history
        assert True in data["probe_history"]

        # Verify miner state via GET
        miner_resp = client.get(f"/api/miners/{miner_id}")
        assert miner_resp.status_code == 200
        assert True in miner_resp.json()["probe_history"]

    def test_commit_changes_task_status_to_assigned(
        self, client, registered_miner, probe_task
    ):
        task_id = probe_task["id"]
        commit_hash = _compute_commit_hash("tampered", 0.9, None, "n")
        client.post(f"/api/tasks/{task_id}/commit", json={
            "miner_id": registered_miner["id"],
            "hash": commit_hash,
        })

        resp = client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["task"]["status"] == "assigned"

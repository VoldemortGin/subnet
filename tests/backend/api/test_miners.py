from __future__ import annotations

from backend.db.store import get_store
from backend.services.miner_service import MinerService


class TestMinerRegister:
    def test_register_success(self, client):
        resp = client.post("/api/miners/register", json={
            "name": "AlphaDetector",
            "backend_name": "ela",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "AlphaDetector"
        assert data["backend_name"] == "ela"
        assert data["probe_accuracy"] == 0.0
        assert data["probe_history"] == []
        assert data["total_score"] == 0.0
        assert data["strike_status"] == "normal"
        assert "id" in data
        assert "created_at" in data

    def test_register_default_backend(self, client):
        resp = client.post("/api/miners/register", json={"name": "MinimalMiner"})
        assert resp.status_code == 200
        assert resp.json()["backend_name"] == "ela"

    def test_register_multiple_miners(self, client):
        for name in ["M1", "M2", "M3"]:
            resp = client.post("/api/miners/register", json={"name": name})
            assert resp.status_code == 200

        store = get_store()
        assert len(store.miners) == 3


class TestMinerList:
    def test_list_empty(self, client):
        resp = client.get("/api/miners")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_with_miners(self, client, registered_miner):
        resp = client.get("/api/miners")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == registered_miner["id"]
        assert data[0]["name"] == "TestMiner"


class TestMinerGet:
    def test_get_existing_miner(self, client, registered_miner):
        miner_id = registered_miner["id"]
        resp = client.get(f"/api/miners/{miner_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == miner_id
        assert data["name"] == "TestMiner"

    def test_get_nonexistent_miner_returns_404(self, client):
        resp = client.get("/api/miners/no-such-id")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Miner not found"


class TestMinerLeaderboard:
    def test_leaderboard_empty(self, client):
        resp = client.get("/api/miners/leaderboard")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_leaderboard_ordering(self, client):
        store = get_store()
        service = MinerService(store)

        m1 = service.register_miner("HighScorer", "ela")
        m2 = service.register_miner("LowScorer", "ela")

        # Give m1 better scores
        for _ in range(5):
            service.record_probe_result(m1.id, True, 0.9)
        for _ in range(5):
            service.record_probe_result(m2.id, True, 0.3)

        resp = client.get("/api/miners/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["rank"] == 1
        assert data[1]["rank"] == 2
        assert data[0]["id"] == m1.id
        assert data[1]["id"] == m2.id
        assert data[0]["total_score"] > data[1]["total_score"]

    def test_leaderboard_includes_strike_status(self, client):
        store = get_store()
        service = MinerService(store)
        m = service.register_miner("StrikedMiner", "ela")
        service.record_probe_result(m.id, False, 0.0)

        resp = client.get("/api/miners/leaderboard")
        data = resp.json()
        assert data[0]["strike_status"] == "yellow_card"

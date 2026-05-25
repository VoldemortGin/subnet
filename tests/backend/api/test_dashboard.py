from __future__ import annotations

from backend.db.store import ImageRecord, get_store
from backend.services.miner_service import MinerService


class TestDashboardStats:
    def test_stats_empty(self, client):
        resp = client.get("/api/dashboard/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images_analyzed"] == 0
        assert data["total_tampered_detected"] == 0
        assert data["total_authentic"] == 0
        assert data["total_miners"] == 0
        assert data["total_probes"] == 0
        assert data["avg_accuracy"] == 0.0
        assert data["active_miners"] == 0

    def test_stats_with_miners(self, client, registered_miner):
        resp = client.get("/api/dashboard/stats")
        data = resp.json()
        assert data["total_miners"] == 1
        assert data["active_miners"] == 1

    def test_stats_counts_images(self, client):
        store = get_store()
        store.images["img1"] = ImageRecord(
            id="img1", filename="a.png", path="/tmp/a.png",
            verdict="tampered", confidence=0.9, status="analyzed",
        )
        store.images["img2"] = ImageRecord(
            id="img2", filename="b.png", path="/tmp/b.png",
            verdict="authentic", confidence=0.8, status="analyzed",
        )
        store.images["img3"] = ImageRecord(
            id="img3", filename="c.png", path="/tmp/c.png",
            verdict=None, confidence=None, status="pending",
        )

        resp = client.get("/api/dashboard/stats")
        data = resp.json()
        assert data["total_images_analyzed"] == 2
        assert data["total_tampered_detected"] == 1
        assert data["total_authentic"] == 1

    def test_stats_counts_probes(self, client, probe_task):
        resp = client.get("/api/dashboard/stats")
        data = resp.json()
        assert data["total_probes"] == 1

    def test_stats_avg_accuracy(self, client):
        store = get_store()
        service = MinerService(store)
        m = service.register_miner("M1", "ela")
        service.record_probe_result(m.id, True, 0.8)
        service.record_probe_result(m.id, True, 0.9)

        resp = client.get("/api/dashboard/stats")
        data = resp.json()
        assert data["avg_accuracy"] == 1.0  # 2/2 correct


    def test_stats_banned_miner_not_active(self, client):
        store = get_store()
        service = MinerService(store)
        m = service.register_miner("BadMiner", "ela")
        # 3 failures in the last window -> banned
        for _ in range(3):
            service.record_probe_result(m.id, False, 0.0)

        resp = client.get("/api/dashboard/stats")
        data = resp.json()
        assert data["total_miners"] == 1
        assert data["active_miners"] == 0

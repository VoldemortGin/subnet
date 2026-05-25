from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.db.store import Store
from backend.main import app


@pytest.fixture(autouse=True)
def reset_store(monkeypatch, tmp_path):
    """Reset the global store and data directory for each test."""
    fresh_store = Store()
    monkeypatch.setattr("backend.db.store._store", fresh_store)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "uploads").mkdir()
    (data_dir / "viz").mkdir()
    (data_dir / "clean").mkdir()
    (data_dir / "probes").mkdir()

    # Create clean images for probe generation
    for i in range(2):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / "clean" / f"clean_{i}.png"), img)

    monkeypatch.setattr("backend.main.DATA_DIR", data_dir)
    monkeypatch.setattr("backend.api.routes.tasks.DATA_DIR", data_dir)
    monkeypatch.setattr("backend.api.routes.images.DATA_DIR", data_dir)

    return fresh_store


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def registered_miner(client):
    """Register a miner and return the response JSON."""
    resp = client.post("/api/miners/register", json={
        "name": "TestMiner",
        "backend_name": "ela",
    })
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def probe_task(client):
    """Create a probe task and return the response JSON."""
    resp = client.post("/api/tasks/probe")
    assert resp.status_code == 200
    return resp.json()

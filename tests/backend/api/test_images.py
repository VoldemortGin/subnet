from __future__ import annotations

import io

import cv2
import numpy as np

from backend.db.store import ImageRecord, get_store


def _create_test_image_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a random PNG image and return its bytes."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


class TestImageUpload:
    def test_upload_success(self, client):
        img_bytes = _create_test_image_bytes()
        resp = client.post(
            "/api/images/upload",
            files={"file": ("test_image.png", io.BytesIO(img_bytes), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["filename"].endswith(".png")
        assert data["verdict"] in ("authentic", "tampered")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["image_url"].startswith("/data/")

    def test_upload_stores_in_db(self, client):
        img_bytes = _create_test_image_bytes()
        resp = client.post(
            "/api/images/upload",
            files={"file": ("photo.png", io.BytesIO(img_bytes), "image/png")},
        )
        assert resp.status_code == 200
        image_id = resp.json()["id"]

        store = get_store()
        assert image_id in store.images
        assert store.images[image_id].status == "analyzed"

    def test_upload_jpg(self, client):
        img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()

        resp = client.post(
            "/api/images/upload",
            files={"file": ("photo.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"].endswith(".jpg")


class TestImageGet:
    def test_get_existing_image(self, client):
        img_bytes = _create_test_image_bytes()
        upload_resp = client.post(
            "/api/images/upload",
            files={"file": ("test.png", io.BytesIO(img_bytes), "image/png")},
        )
        image_id = upload_resp.json()["id"]

        resp = client.get(f"/api/images/{image_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == image_id
        assert data["status"] == "analyzed"
        assert "upload_time" in data

    def test_get_nonexistent_image_returns_404(self, client):
        resp = client.get("/api/images/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Image not found"


class TestImageList:
    def test_list_empty(self, client):
        resp = client.get("/api/images")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_uploads(self, client):
        for i in range(3):
            img_bytes = _create_test_image_bytes()
            client.post(
                "/api/images/upload",
                files={"file": (f"img_{i}.png", io.BytesIO(img_bytes), "image/png")},
            )

        resp = client.get("/api/images")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        for item in data:
            assert "id" in item
            assert "filename" in item
            assert item["status"] == "analyzed"

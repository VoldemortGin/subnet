from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from backend.api.models import ImageDetail, ImageUploadResponse
from backend.db.store import get_store
from backend.services.detect_service import DetectService

router = APIRouter(prefix="/api/images", tags=["images"])

DATA_DIR = Path("data")


def _get_detect_service() -> DetectService:
    return DetectService(get_store(), DATA_DIR)


def _image_url(path: str) -> str:
    return f"/data/{Path(path).relative_to(DATA_DIR)}"


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile):
    uploads_dir = DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    image_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "image.png").suffix or ".png"
    filename = f"{image_id}{ext}"
    file_path = uploads_dir / filename

    content = await file.read()
    file_path.write_bytes(content)

    service = _get_detect_service()
    record = service.analyze_image(file_path, image_id)

    return ImageUploadResponse(
        id=record.id,
        filename=record.filename,
        verdict=record.verdict or "unknown",
        confidence=record.confidence or 0.0,
        method=record.method,
        image_url=_image_url(record.path),
        mask_url=_image_url(record.mask_path) if record.mask_path else None,
        visualization_url=_image_url(record.visualization_path) if record.visualization_path else None,
    )


@router.get("", response_model=list[ImageDetail])
async def list_images():
    store = get_store()
    results = []
    for img in store.images.values():
        results.append(ImageDetail(
            id=img.id,
            filename=img.filename,
            verdict=img.verdict,
            confidence=img.confidence,
            method=img.method,
            image_url=_image_url(img.path),
            mask_url=_image_url(img.mask_path) if img.mask_path else None,
            visualization_url=_image_url(img.visualization_path) if img.visualization_path else None,
            upload_time=img.upload_time,
            status=img.status,
        ))
    return results


@router.get("/{image_id}", response_model=ImageDetail)
async def get_image(image_id: str):
    store = get_store()
    img = store.images.get(image_id)
    if img is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageDetail(
        id=img.id,
        filename=img.filename,
        verdict=img.verdict,
        confidence=img.confidence,
        method=img.method,
        image_url=_image_url(img.path),
        mask_url=_image_url(img.mask_path) if img.mask_path else None,
        visualization_url=_image_url(img.visualization_path) if img.visualization_path else None,
        upload_time=img.upload_time,
        status=img.status,
    )

from __future__ import annotations

from fastapi import APIRouter

from backend.api.models import DashboardStats
from backend.db.store import get_store
from backend.services.miner_service import MinerService

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_stats():
    store = get_store()
    service = MinerService(store)

    total_images = len([
        img for img in store.images.values()
        if img.status == "analyzed"
    ])
    total_tampered = len([
        img for img in store.images.values()
        if img.verdict == "tampered" and img.status == "analyzed"
    ])
    total_authentic = len([
        img for img in store.images.values()
        if img.verdict == "authentic" and img.status == "analyzed"
    ])
    total_miners = len(store.miners)
    total_probes = len([
        t for t in store.tasks.values()
        if t.task_type == "probe"
    ])

    accuracies = [
        service.get_probe_accuracy(m.id)
        for m in store.miners.values()
        if m.probe_history
    ]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    active_miners = len([
        m for m in store.miners.values()
        if service.get_strike_status(m.id) not in ("banned",)
    ])

    return DashboardStats(
        total_images_analyzed=total_images,
        total_tampered_detected=total_tampered,
        total_authentic=total_authentic,
        total_miners=total_miners,
        total_probes=total_probes,
        avg_accuracy=round(avg_accuracy, 4),
        active_miners=active_miners,
    )

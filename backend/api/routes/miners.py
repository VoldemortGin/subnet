from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.models import LeaderboardEntry, MinerInfo, MinerRegisterRequest
from backend.db.store import get_store
from backend.services.miner_service import MinerService

router = APIRouter(prefix="/api/miners", tags=["miners"])


def _get_miner_service() -> MinerService:
    return MinerService(get_store())


def _miner_to_info(service: MinerService, miner) -> MinerInfo:
    return MinerInfo(
        id=miner.id,
        name=miner.name,
        backend_name=miner.backend_name,
        probe_accuracy=service.get_probe_accuracy(miner.id),
        probe_history=miner.probe_history,
        total_score=miner.total_score,
        strike_status=service.get_strike_status(miner.id),
        avg_latency_ms=miner.avg_latency_ms,
        created_at=miner.created_at,
    )


@router.post("/register", response_model=MinerInfo)
async def register_miner(req: MinerRegisterRequest):
    service = _get_miner_service()
    miner = service.register_miner(req.name, req.backend_name)
    return _miner_to_info(service, miner)


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard():
    service = _get_miner_service()
    miners = service.get_leaderboard()
    return [
        LeaderboardEntry(
            rank=i + 1,
            id=m.id,
            name=m.name,
            backend_name=m.backend_name,
            total_score=m.total_score,
            probe_accuracy=service.get_probe_accuracy(m.id),
            strike_status=service.get_strike_status(m.id),
        )
        for i, m in enumerate(miners)
    ]


@router.get("", response_model=list[MinerInfo])
async def list_miners():
    service = _get_miner_service()
    store = get_store()
    return [_miner_to_info(service, m) for m in store.miners.values()]


@router.get("/{miner_id}", response_model=MinerInfo)
async def get_miner(miner_id: str):
    service = _get_miner_service()
    store = get_store()
    miner = store.miners.get(miner_id)
    if miner is None:
        raise HTTPException(status_code=404, detail="Miner not found")
    return _miner_to_info(service, miner)

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.models import (
    CommitRequest,
    CommitResponse,
    RevealRequest,
    RevealResponse,
    SubmissionInfo,
    SubmissionRequest,
    TaskDetail,
    TaskInfo,
)
from backend.db.store import SubmissionRecord, get_store
from backend.services.detect_service import DetectService
from backend.services.forge_service import ForgeService
from backend.services.miner_service import MinerService
from backend.services.task_service import TaskService

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

DATA_DIR = Path("data")


def _get_task_service() -> TaskService:
    store = get_store()
    forge = ForgeService(store, DATA_DIR)
    detect = DetectService(store, DATA_DIR)
    miner_svc = MinerService(store)
    return TaskService(store, forge, detect, miner_svc)


def _task_to_info(task) -> TaskInfo:
    return TaskInfo(
        id=task.id,
        image_id=task.image_id,
        task_type=task.task_type,
        status=task.status,
        ground_truth_verdict=task.ground_truth_verdict,
        ground_truth_method=task.ground_truth_method,
        created_at=task.created_at,
    )


@router.get("", response_model=list[TaskInfo])
async def list_tasks(task_type: str | None = None, status: str | None = None):
    service = _get_task_service()
    tasks = service.list_tasks(task_type=task_type, status=status)
    return [_task_to_info(t) for t in tasks]


@router.post("/probe", response_model=TaskInfo)
async def create_probe():
    service = _get_task_service()
    try:
        task = service.create_probe_task()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _task_to_info(task)


@router.get("/{task_id}", response_model=TaskDetail)
async def get_task(task_id: str):
    store = get_store()
    task = store.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    subs = store.submissions.get(task_id, [])
    return TaskDetail(
        task=_task_to_info(task),
        submissions=[
            SubmissionInfo(
                miner_id=s.miner_id,
                task_id=s.task_id,
                verdict=s.verdict,
                confidence=s.confidence,
                score=s.score,
                latency_ms=s.latency_ms,
                created_at=s.created_at,
            )
            for s in subs
        ],
    )


@router.post("/{task_id}/submit", response_model=SubmissionInfo)
async def submit_result(task_id: str, req: SubmissionRequest):
    service = _get_task_service()
    try:
        sub = service.submit_result(task_id, req.miner_id, req.verdict, req.confidence)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return SubmissionInfo(
        miner_id=sub.miner_id,
        task_id=sub.task_id,
        verdict=sub.verdict,
        confidence=sub.confidence,
        score=sub.score,
        latency_ms=sub.latency_ms,
        created_at=sub.created_at,
    )


@router.post("/{task_id}/commit", response_model=CommitResponse)
async def commit_hash(task_id: str, req: CommitRequest):
    store = get_store()
    task = store.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if req.miner_id not in store.miners:
        raise HTTPException(status_code=404, detail="Miner not found")

    placeholder = SubmissionRecord(
        miner_id=req.miner_id,
        task_id=task_id,
        verdict="",
        confidence=0.0,
        committed_hash=req.hash,
    )
    store.submissions.setdefault(task_id, []).append(placeholder)
    task.status = "assigned"

    return CommitResponse(ok=True, message="Commitment recorded")


@router.post("/{task_id}/reveal", response_model=RevealResponse)
async def reveal_answer(task_id: str, req: RevealRequest):
    store = get_store()
    task = store.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    miner = store.miners.get(req.miner_id)
    if miner is None:
        raise HTTPException(status_code=404, detail="Miner not found")

    subs = store.submissions.get(task_id, [])
    committed = next(
        (s for s in subs if s.miner_id == req.miner_id and s.committed_hash),
        None,
    )
    if committed is None:
        raise HTTPException(status_code=400, detail="No commitment found for this miner")

    method_str = req.method or ""
    payload = f"{req.verdict}|{req.confidence}|{method_str}|{req.nonce}"
    computed_hash = hashlib.sha256(payload.encode()).hexdigest()

    if computed_hash != committed.committed_hash:
        miner_service = MinerService(store)
        return RevealResponse(
            hash_valid=False,
            score=0.0,
            is_probe=task.task_type == "probe",
            ground_truth=None,
            strike_status=miner_service.get_strike_status(req.miner_id),
            probe_history=miner.probe_history[-10:],
        )

    committed.verdict = req.verdict
    committed.confidence = req.confidence

    service = _get_task_service()
    try:
        sub = service.submit_result(task_id, req.miner_id, req.verdict, req.confidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    subs[:] = [s for s in subs if not (s.miner_id == req.miner_id and s.committed_hash and s.verdict == "")]

    miner_service = MinerService(store)
    ground_truth = None
    if task.task_type == "probe":
        ground_truth = {
            "verdict": task.ground_truth_verdict,
            "method": task.ground_truth_method,
        }

    return RevealResponse(
        hash_valid=True,
        score=sub.score,
        is_probe=task.task_type == "probe",
        ground_truth=ground_truth,
        strike_status=miner_service.get_strike_status(req.miner_id),
        probe_history=miner.probe_history[-10:],
    )

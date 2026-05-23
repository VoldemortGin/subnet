from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.models import SubmissionInfo, SubmissionRequest, TaskDetail, TaskInfo
from backend.db.store import get_store
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

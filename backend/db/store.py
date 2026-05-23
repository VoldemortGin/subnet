from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MinerRecord:
    id: str
    name: str
    backend_name: str
    probe_history: list[bool] = field(default_factory=list)
    probe_scores: list[float] = field(default_factory=list)
    consensus_scores: list[float] = field(default_factory=list)
    total_score: float = 0.0
    avg_latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImageRecord:
    id: str
    filename: str
    path: str
    upload_time: datetime = field(default_factory=datetime.now)
    verdict: str | None = None
    confidence: float | None = None
    method: str | None = None
    mask_path: str | None = None
    visualization_path: str | None = None
    status: str = "pending"


@dataclass
class TaskRecord:
    id: str
    image_id: str
    task_type: str  # "probe" or "real"
    ground_truth_verdict: str | None = None
    ground_truth_method: str | None = None
    ground_truth_mask_path: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"


@dataclass
class SubmissionRecord:
    miner_id: str
    task_id: str
    verdict: str
    confidence: float
    mask_path: str | None = None
    latency_ms: float = 0.0
    score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class Store:
    def __init__(self):
        self.images: dict[str, ImageRecord] = {}
        self.tasks: dict[str, TaskRecord] = {}
        self.miners: dict[str, MinerRecord] = {}
        self.submissions: dict[str, list[SubmissionRecord]] = {}


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store

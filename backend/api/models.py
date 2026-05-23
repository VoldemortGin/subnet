from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ImageUploadResponse(BaseModel):
    id: str
    filename: str
    verdict: str
    confidence: float
    method: str | None = None
    image_url: str
    mask_url: str | None = None
    visualization_url: str | None = None


class ImageDetail(BaseModel):
    id: str
    filename: str
    verdict: str | None = None
    confidence: float | None = None
    method: str | None = None
    image_url: str
    mask_url: str | None = None
    visualization_url: str | None = None
    upload_time: datetime
    status: str


class MinerRegisterRequest(BaseModel):
    name: str
    backend_name: str = "ela"


class MinerInfo(BaseModel):
    id: str
    name: str
    backend_name: str
    probe_accuracy: float
    probe_history: list[bool]
    total_score: float
    strike_status: str
    avg_latency_ms: float
    created_at: datetime


class TaskInfo(BaseModel):
    id: str
    image_id: str
    task_type: str
    status: str
    ground_truth_verdict: str | None = None
    ground_truth_method: str | None = None
    created_at: datetime


class SubmissionRequest(BaseModel):
    miner_id: str
    verdict: str
    confidence: float


class SubmissionInfo(BaseModel):
    miner_id: str
    task_id: str
    verdict: str
    confidence: float
    score: float
    latency_ms: float
    created_at: datetime


class TaskDetail(BaseModel):
    task: TaskInfo
    submissions: list[SubmissionInfo]


class DashboardStats(BaseModel):
    total_images_analyzed: int
    total_tampered_detected: int
    total_authentic: int
    total_miners: int
    total_probes: int
    avg_accuracy: float
    active_miners: int


class LeaderboardEntry(BaseModel):
    rank: int
    id: str
    name: str
    backend_name: str
    total_score: float
    probe_accuracy: float
    strike_status: str


class CommitRequest(BaseModel):
    miner_id: str
    hash: str


class CommitResponse(BaseModel):
    ok: bool
    message: str


class RevealRequest(BaseModel):
    miner_id: str
    verdict: str
    confidence: float
    method: str | None = None
    nonce: str


class RevealResponse(BaseModel):
    hash_valid: bool
    score: float
    is_probe: bool
    ground_truth: dict | None = None
    strike_status: str
    probe_history: list[bool]

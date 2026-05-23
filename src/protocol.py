from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class Verdict(str, Enum):
    AUTHENTIC = "authentic"
    TAMPERED = "tampered"


class ForgeryMethod(str, Enum):
    COPY_MOVE = "copy_move"
    SPLICING = "splicing"
    INPAINTING = "inpainting"
    COMPRESSION = "compression"
    METADATA = "metadata"


@dataclass
class TaskRequest:
    image_path: Path
    task_id: str
    timeout_ms: int = 30_000


@dataclass
class GroundTruth:
    verdict: Verdict
    method: ForgeryMethod | None = None
    mask: np.ndarray | None = None


@dataclass
class MinerResponse:
    task_id: str
    verdict: Verdict
    confidence: float
    method: ForgeryMethod | None = None
    mask: np.ndarray | None = None
    latency_ms: float = 0.0


@dataclass
class ProbeTask:
    original_image_path: Path
    tampered_image_path: Path
    ground_truth: GroundTruth


@dataclass
class ScoreResult:
    miner_id: str
    probe_score: float = 0.0
    consensus_score: float = 0.0
    latency_score: float = 0.0
    total_score: float = 0.0
    details: dict = field(default_factory=dict)

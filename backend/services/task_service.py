from __future__ import annotations

import random
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from src.protocol import GroundTruth, MinerResponse, Verdict
from src.validator.scorer import ProbeScorer

from backend.db.store import ImageRecord, Store, SubmissionRecord, TaskRecord
from backend.services.detect_service import DetectService
from backend.services.forge_service import ForgeService
from backend.services.miner_service import MinerService


class TaskService:
    def __init__(
        self,
        store: Store,
        forge_service: ForgeService,
        detect_service: DetectService,
        miner_service: MinerService,
    ):
        self.store = store
        self.forge_service = forge_service
        self.detect_service = detect_service
        self.miner_service = miner_service
        self.scorer = ProbeScorer()

    def create_real_task(self, image_id: str) -> TaskRecord:
        task_id = str(uuid.uuid4())[:8]
        task = TaskRecord(
            id=task_id,
            image_id=image_id,
            task_type="real",
            status="pending",
        )
        self.store.tasks[task_id] = task
        self.store.submissions.setdefault(task_id, [])
        return task

    def create_probe_task(self) -> TaskRecord:
        clean_images = self.forge_service.list_clean_images()
        if not clean_images:
            raise RuntimeError("No clean images available for probe generation")
        chosen = random.choice(clean_images)
        return self.forge_service.generate_probe(chosen)

    def submit_result(
        self,
        task_id: str,
        miner_id: str,
        verdict: str,
        confidence: float,
    ) -> SubmissionRecord:
        task = self.store.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        miner = self.store.miners.get(miner_id)
        if miner is None:
            raise ValueError(f"Miner {miner_id} not found")

        start_time = time.time()

        score = 0.0
        if task.task_type == "probe" and task.ground_truth_verdict:
            gt_mask = None
            if task.ground_truth_mask_path:
                loaded = cv2.imread(task.ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
                if loaded is not None:
                    gt_mask = loaded

            from src.protocol import ForgeryMethod
            gt_method = None
            if task.ground_truth_method:
                gt_method = ForgeryMethod(task.ground_truth_method)

            ground_truth = GroundTruth(
                verdict=Verdict(task.ground_truth_verdict),
                method=gt_method,
                mask=gt_mask,
            )
            miner_response = MinerResponse(
                task_id=task_id,
                verdict=Verdict(verdict),
                confidence=confidence,
                method=None,
                mask=None,
            )
            score = self.scorer.score_probe(miner_response, ground_truth)

            correct = miner_response.verdict == ground_truth.verdict
            self.miner_service.record_probe_result(miner_id, correct, score)

        latency_ms = (time.time() - start_time) * 1000

        submission = SubmissionRecord(
            miner_id=miner_id,
            task_id=task_id,
            verdict=verdict,
            confidence=confidence,
            score=score,
            latency_ms=latency_ms,
        )
        self.store.submissions.setdefault(task_id, []).append(submission)
        task.status = "completed"

        return submission

    def list_tasks(self, task_type: str | None = None, status: str | None = None) -> list[TaskRecord]:
        tasks = list(self.store.tasks.values())
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        if status:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.protocol import (
    ForgeryMethod,
    GroundTruth,
    MinerResponse,
    ProbeTask,
    ScoreResult,
    TaskRequest,
    Verdict,
)


class TestVerdict:
    def test_values(self):
        assert Verdict.AUTHENTIC == "authentic"
        assert Verdict.TAMPERED == "tampered"

    def test_is_str_enum(self):
        assert isinstance(Verdict.AUTHENTIC, str)
        assert isinstance(Verdict.TAMPERED, str)

    def test_from_value(self):
        assert Verdict("authentic") is Verdict.AUTHENTIC
        assert Verdict("tampered") is Verdict.TAMPERED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Verdict("invalid")


class TestForgeryMethod:
    def test_values(self):
        assert ForgeryMethod.COPY_MOVE == "copy_move"
        assert ForgeryMethod.SPLICING == "splicing"
        assert ForgeryMethod.INPAINTING == "inpainting"
        assert ForgeryMethod.COMPRESSION == "compression"
        assert ForgeryMethod.METADATA == "metadata"

    def test_is_str_enum(self):
        for member in ForgeryMethod:
            assert isinstance(member, str)

    def test_from_value(self):
        assert ForgeryMethod("copy_move") is ForgeryMethod.COPY_MOVE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ForgeryMethod("blur")


class TestTaskRequest:
    def test_creation_defaults(self):
        tr = TaskRequest(image_path=Path("/tmp/img.png"), task_id="t1")
        assert tr.image_path == Path("/tmp/img.png")
        assert tr.task_id == "t1"
        assert tr.timeout_ms == 30_000

    def test_custom_timeout(self):
        tr = TaskRequest(image_path=Path("/x.png"), task_id="t2", timeout_ms=5_000)
        assert tr.timeout_ms == 5_000

    def test_path_types(self):
        tr = TaskRequest(image_path=Path("relative/path.jpg"), task_id="t3")
        assert isinstance(tr.image_path, Path)


class TestGroundTruth:
    def test_defaults(self):
        gt = GroundTruth(verdict=Verdict.AUTHENTIC)
        assert gt.verdict == Verdict.AUTHENTIC
        assert gt.method is None
        assert gt.mask is None

    def test_with_all_fields(self):
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        gt = GroundTruth(
            verdict=Verdict.TAMPERED,
            method=ForgeryMethod.SPLICING,
            mask=mask,
        )
        assert gt.verdict == Verdict.TAMPERED
        assert gt.method == ForgeryMethod.SPLICING
        assert np.array_equal(gt.mask, mask)

    def test_authentic_no_method(self):
        gt = GroundTruth(verdict=Verdict.AUTHENTIC, method=None)
        assert gt.method is None


class TestMinerResponse:
    def test_minimal_creation(self):
        mr = MinerResponse(
            task_id="task_1",
            verdict=Verdict.TAMPERED,
            confidence=0.95,
        )
        assert mr.task_id == "task_1"
        assert mr.verdict == Verdict.TAMPERED
        assert mr.confidence == 0.95
        assert mr.method is None
        assert mr.mask is None
        assert mr.latency_ms == 0.0

    def test_full_creation(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mr = MinerResponse(
            task_id="task_2",
            verdict=Verdict.TAMPERED,
            confidence=0.88,
            method=ForgeryMethod.COPY_MOVE,
            mask=mask,
            latency_ms=123.4,
        )
        assert mr.method == ForgeryMethod.COPY_MOVE
        assert mr.latency_ms == 123.4
        assert mr.mask is not None

    def test_confidence_boundary_values(self):
        mr_zero = MinerResponse(task_id="t", verdict=Verdict.AUTHENTIC, confidence=0.0)
        mr_one = MinerResponse(task_id="t", verdict=Verdict.TAMPERED, confidence=1.0)
        assert mr_zero.confidence == 0.0
        assert mr_one.confidence == 1.0


class TestProbeTask:
    def test_creation(self):
        gt = GroundTruth(verdict=Verdict.TAMPERED, method=ForgeryMethod.COMPRESSION)
        pt = ProbeTask(
            original_image_path=Path("/clean.png"),
            tampered_image_path=Path("/tampered.png"),
            ground_truth=gt,
        )
        assert pt.original_image_path == Path("/clean.png")
        assert pt.tampered_image_path == Path("/tampered.png")
        assert pt.ground_truth.verdict == Verdict.TAMPERED

    def test_ground_truth_accessible(self):
        gt = GroundTruth(verdict=Verdict.TAMPERED)
        pt = ProbeTask(
            original_image_path=Path("/a.png"),
            tampered_image_path=Path("/b.png"),
            ground_truth=gt,
        )
        assert pt.ground_truth.method is None


class TestScoreResult:
    def test_defaults(self):
        sr = ScoreResult(miner_id="m1")
        assert sr.miner_id == "m1"
        assert sr.probe_score == 0.0
        assert sr.consensus_score == 0.0
        assert sr.latency_score == 0.0
        assert sr.total_score == 0.0
        assert sr.details == {}

    def test_custom_values(self):
        sr = ScoreResult(
            miner_id="m2",
            probe_score=0.8,
            consensus_score=0.7,
            latency_score=0.9,
            total_score=0.78,
            details={"num_probes": 5},
        )
        assert sr.probe_score == 0.8
        assert sr.details["num_probes"] == 5

    def test_details_default_factory_independence(self):
        sr1 = ScoreResult(miner_id="a")
        sr2 = ScoreResult(miner_id="b")
        sr1.details["x"] = 1
        assert "x" not in sr2.details

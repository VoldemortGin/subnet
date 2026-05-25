from __future__ import annotations

import numpy as np
import pytest

from src.protocol import ForgeryMethod, GroundTruth, MinerResponse, Verdict
from src.validator.scorer import ProbeScorer, ScoringWeights


class TestScoringWeights:
    def test_default_values(self):
        w = ScoringWeights()
        assert w.alpha == 0.4
        assert w.beta == 0.4
        assert w.gamma == 0.2
        assert w.omega_probe == 0.60
        assert w.omega_consensus == 0.35
        assert w.omega_latency == 0.05

    def test_custom_values(self):
        w = ScoringWeights(alpha=0.5, beta=0.3, gamma=0.2)
        assert w.alpha == 0.5
        assert w.beta == 0.3


class TestComputeIoU:
    @pytest.fixture
    def scorer(self) -> ProbeScorer:
        return ProbeScorer()

    def test_perfect_overlap(self, scorer: ProbeScorer):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        assert scorer.compute_iou(mask, mask) == 1.0

    def test_no_overlap(self, scorer: ProbeScorer):
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[:50, :] = 255
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[50:, :] = 255
        assert scorer.compute_iou(pred, gt) == 0.0

    def test_partial_overlap(self, scorer: ProbeScorer):
        # Create masks with known 50% IoU
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[:60, :] = 255  # 6000 pixels
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[:40, :] = 255  # 4000 pixels
        # Intersection = 4000, union = 6000, IoU = 4000/6000 = 2/3
        iou = scorer.compute_iou(pred, gt)
        assert abs(iou - 2 / 3) < 1e-6

    def test_pred_none_returns_zero(self, scorer: ProbeScorer):
        gt = np.ones((50, 50), dtype=np.uint8) * 255
        assert scorer.compute_iou(None, gt) == 0.0

    def test_gt_none_returns_zero(self, scorer: ProbeScorer):
        pred = np.ones((50, 50), dtype=np.uint8) * 255
        assert scorer.compute_iou(pred, None) == 0.0

    def test_both_none_returns_zero(self, scorer: ProbeScorer):
        assert scorer.compute_iou(None, None) == 0.0

    def test_both_empty_returns_zero(self, scorer: ProbeScorer):
        pred = np.zeros((100, 100), dtype=np.uint8)
        gt = np.zeros((100, 100), dtype=np.uint8)
        assert scorer.compute_iou(pred, gt) == 0.0


class TestScoreProbe:
    @pytest.fixture
    def scorer(self) -> ProbeScorer:
        return ProbeScorer()

    def test_wrong_verdict_returns_zero(self, scorer: ProbeScorer):
        response = MinerResponse(
            task_id="t1", verdict=Verdict.AUTHENTIC, confidence=0.9
        )
        gt = GroundTruth(verdict=Verdict.TAMPERED, method=ForgeryMethod.COPY_MOVE)
        assert scorer.score_probe(response, gt) == 0.0

    def test_perfect_tampered_score(self, scorer: ProbeScorer):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        response = MinerResponse(
            task_id="t1",
            verdict=Verdict.TAMPERED,
            confidence=1.0,
            method=ForgeryMethod.COPY_MOVE,
            mask=mask,
        )
        gt = GroundTruth(
            verdict=Verdict.TAMPERED,
            method=ForgeryMethod.COPY_MOVE,
            mask=mask,
        )
        score = scorer.score_probe(response, gt)
        # IoU=1.0, C_cal=1.0 (conf=1.0, ideal=1.0), M_bonus=1.0
        # 0.4*1 + 0.4*1 + 0.2*1 = 1.0
        assert abs(score - 1.0) < 1e-6

    def test_perfect_authentic_score(self, scorer: ProbeScorer):
        response = MinerResponse(
            task_id="t1",
            verdict=Verdict.AUTHENTIC,
            confidence=0.0,
        )
        gt = GroundTruth(verdict=Verdict.AUTHENTIC)
        score = scorer.score_probe(response, gt)
        # IoU=0 (both None), C_cal=1.0 (conf=0.0, ideal=0.0), M_bonus=0 (gt.method is None)
        # 0.4*0 + 0.4*1 + 0.2*0 = 0.4
        assert abs(score - 0.4) < 1e-6

    def test_calibration_penalizes_overconfidence(self, scorer: ProbeScorer):
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        # Overconfident authentic response
        response = MinerResponse(
            task_id="t1", verdict=Verdict.AUTHENTIC, confidence=0.8
        )
        gt = GroundTruth(verdict=Verdict.AUTHENTIC)
        score = scorer.score_probe(response, gt)
        # C_cal = 1 - |0.8 - 0.0| = 0.2
        # 0.4*0 + 0.4*0.2 + 0.2*0 = 0.08
        assert abs(score - 0.08) < 1e-6

    def test_method_bonus_only_when_both_match(self, scorer: ProbeScorer):
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        response_wrong_method = MinerResponse(
            task_id="t1",
            verdict=Verdict.TAMPERED,
            confidence=1.0,
            method=ForgeryMethod.SPLICING,
            mask=mask,
        )
        gt = GroundTruth(
            verdict=Verdict.TAMPERED,
            method=ForgeryMethod.COPY_MOVE,
            mask=mask,
        )
        score = scorer.score_probe(response_wrong_method, gt)
        # IoU=1.0, C_cal=1.0, M_bonus=0
        # 0.4*1 + 0.4*1 + 0.2*0 = 0.8
        assert abs(score - 0.8) < 1e-6

    def test_no_mask_iou_is_zero(self, scorer: ProbeScorer):
        response = MinerResponse(
            task_id="t1",
            verdict=Verdict.TAMPERED,
            confidence=1.0,
            method=ForgeryMethod.COPY_MOVE,
            mask=None,
        )
        gt = GroundTruth(
            verdict=Verdict.TAMPERED,
            method=ForgeryMethod.COPY_MOVE,
            mask=np.ones((50, 50), dtype=np.uint8) * 255,
        )
        score = scorer.score_probe(response, gt)
        # IoU=0, C_cal=1, M_bonus=1 -> 0.4*0 + 0.4*1 + 0.2*1 = 0.6
        assert abs(score - 0.6) < 1e-6


class TestScoreConsensus:
    @pytest.fixture
    def scorer(self) -> ProbeScorer:
        return ProbeScorer()

    def test_unanimous_tampered(self, scorer: ProbeScorer):
        responses = [
            MinerResponse(task_id="m1", verdict=Verdict.TAMPERED, confidence=0.9),
            MinerResponse(task_id="m2", verdict=Verdict.TAMPERED, confidence=0.8),
            MinerResponse(task_id="m3", verdict=Verdict.TAMPERED, confidence=0.7),
        ]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = scorer.score_consensus(responses, weights)
        assert result == {"m1": 1.0, "m2": 1.0, "m3": 1.0}

    def test_majority_vote(self, scorer: ProbeScorer):
        responses = [
            MinerResponse(task_id="m1", verdict=Verdict.TAMPERED, confidence=0.9),
            MinerResponse(task_id="m2", verdict=Verdict.TAMPERED, confidence=0.8),
            MinerResponse(task_id="m3", verdict=Verdict.AUTHENTIC, confidence=0.7),
        ]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = scorer.score_consensus(responses, weights)
        assert result["m1"] == 1.0
        assert result["m2"] == 1.0
        assert result["m3"] == 0.0

    def test_weighted_minority_wins(self, scorer: ProbeScorer):
        # Single high-weight miner can outweigh two low-weight miners
        responses = [
            MinerResponse(task_id="m1", verdict=Verdict.AUTHENTIC, confidence=0.9),
            MinerResponse(task_id="m2", verdict=Verdict.TAMPERED, confidence=0.8),
            MinerResponse(task_id="m3", verdict=Verdict.TAMPERED, confidence=0.7),
        ]
        weights = {"m1": 10.0, "m2": 1.0, "m3": 1.0}
        result = scorer.score_consensus(responses, weights)
        # authentic_weight=10, tampered_weight=2 -> majority is AUTHENTIC
        assert result["m1"] == 1.0
        assert result["m2"] == 0.0
        assert result["m3"] == 0.0

    def test_empty_responses(self, scorer: ProbeScorer):
        result = scorer.score_consensus([], {})
        assert result == {}

    def test_missing_weight_uses_default(self, scorer: ProbeScorer):
        responses = [
            MinerResponse(task_id="m1", verdict=Verdict.TAMPERED, confidence=0.9),
            MinerResponse(task_id="m2", verdict=Verdict.AUTHENTIC, confidence=0.8),
        ]
        # m2 not in weights dict -> defaults to 0.5
        weights = {"m1": 0.5}
        result = scorer.score_consensus(responses, weights)
        # tampered=0.5, authentic=0.5, tie goes to TAMPERED (>=)
        assert result["m1"] == 1.0
        assert result["m2"] == 0.0

    def test_tie_goes_to_tampered(self, scorer: ProbeScorer):
        responses = [
            MinerResponse(task_id="m1", verdict=Verdict.TAMPERED, confidence=0.9),
            MinerResponse(task_id="m2", verdict=Verdict.AUTHENTIC, confidence=0.8),
        ]
        weights = {"m1": 1.0, "m2": 1.0}
        result = scorer.score_consensus(responses, weights)
        # Equal weight -> tampered >= authentic -> majority = TAMPERED
        assert result["m1"] == 1.0
        assert result["m2"] == 0.0


class TestScoreEpoch:
    @pytest.fixture
    def scorer(self) -> ProbeScorer:
        return ProbeScorer()

    def test_perfect_scores(self, scorer: ProbeScorer):
        result = scorer.score_epoch(
            probe_scores=[1.0, 1.0, 1.0],
            consensus_scores=[1.0, 1.0],
            avg_latency_ms=0.0,
            max_latency_ms=1000.0,
            miner_id="m1",
        )
        assert result.miner_id == "m1"
        assert abs(result.probe_score - 1.0) < 1e-6
        assert abs(result.consensus_score - 1.0) < 1e-6
        assert abs(result.latency_score - 1.0) < 1e-6
        # 0.60*1 + 0.35*1 + 0.05*1 = 1.0
        assert abs(result.total_score - 1.0) < 1e-6

    def test_zero_scores(self, scorer: ProbeScorer):
        result = scorer.score_epoch(
            probe_scores=[0.0, 0.0],
            consensus_scores=[0.0],
            avg_latency_ms=1000.0,
            max_latency_ms=1000.0,
            miner_id="m2",
        )
        assert result.probe_score == 0.0
        assert result.consensus_score == 0.0
        assert result.latency_score == 0.0
        assert result.total_score == 0.0

    def test_latency_scoring(self, scorer: ProbeScorer):
        # Half of max latency -> latency_score = 0.5
        result = scorer.score_epoch(
            probe_scores=[0.5],
            consensus_scores=[0.5],
            avg_latency_ms=500.0,
            max_latency_ms=1000.0,
            miner_id="m3",
        )
        assert abs(result.latency_score - 0.5) < 1e-6

    def test_latency_exceeds_max_clamped(self, scorer: ProbeScorer):
        result = scorer.score_epoch(
            probe_scores=[1.0],
            consensus_scores=[1.0],
            avg_latency_ms=2000.0,
            max_latency_ms=1000.0,
            miner_id="m4",
        )
        # max(0, 1 - 2000/1000) = max(0, -1) = 0
        assert result.latency_score == 0.0

    def test_max_latency_zero(self, scorer: ProbeScorer):
        result = scorer.score_epoch(
            probe_scores=[1.0],
            consensus_scores=[1.0],
            avg_latency_ms=100.0,
            max_latency_ms=0.0,
            miner_id="m5",
        )
        assert result.latency_score == 0.0

    def test_empty_probe_scores(self, scorer: ProbeScorer):
        result = scorer.score_epoch(
            probe_scores=[],
            consensus_scores=[1.0],
            avg_latency_ms=0.0,
            max_latency_ms=1000.0,
            miner_id="m6",
        )
        assert result.probe_score == 0.0

    def test_empty_consensus_scores(self, scorer: ProbeScorer):
        result = scorer.score_epoch(
            probe_scores=[1.0],
            consensus_scores=[],
            avg_latency_ms=0.0,
            max_latency_ms=1000.0,
            miner_id="m7",
        )
        assert result.consensus_score == 0.0

    def test_details_populated(self, scorer: ProbeScorer):
        result = scorer.score_epoch(
            probe_scores=[0.8, 0.6],
            consensus_scores=[1.0, 0.0, 1.0],
            avg_latency_ms=250.0,
            max_latency_ms=500.0,
            miner_id="m8",
        )
        assert result.details["num_probes"] == 2
        assert result.details["num_consensus"] == 3
        assert result.details["avg_latency_ms"] == 250.0
        assert result.details["max_latency_ms"] == 500.0

    def test_weighted_total_calculation(self, scorer: ProbeScorer):
        result = scorer.score_epoch(
            probe_scores=[0.8],
            consensus_scores=[0.6],
            avg_latency_ms=200.0,
            max_latency_ms=1000.0,
            miner_id="m9",
        )
        # probe=0.8, consensus=0.6, latency=1-200/1000=0.8
        expected = 0.60 * 0.8 + 0.35 * 0.6 + 0.05 * 0.8
        assert abs(result.total_score - expected) < 1e-6

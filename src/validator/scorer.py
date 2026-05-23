from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.protocol import GroundTruth, MinerResponse, ScoreResult, Verdict


@dataclass
class ScoringWeights:
    alpha: float = 0.4   # IoU weight
    beta: float = 0.4    # calibration weight
    gamma: float = 0.2   # method bonus weight

    omega_probe: float = 0.60
    omega_consensus: float = 0.35
    omega_latency: float = 0.05


class ProbeScorer:
    def __init__(self, weights: ScoringWeights | None = None):
        self.weights = weights or ScoringWeights()

    def compute_iou(
        self, pred_mask: np.ndarray | None, gt_mask: np.ndarray | None
    ) -> float:
        if pred_mask is None or gt_mask is None:
            return 0.0
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    def score_probe(
        self, response: MinerResponse, ground_truth: GroundTruth
    ) -> float:
        if response.verdict != ground_truth.verdict:
            return 0.0

        iou = self.compute_iou(response.mask, ground_truth.mask)

        ideal_conf = 1.0 if ground_truth.verdict == Verdict.TAMPERED else 0.0
        c_cal = 1.0 - abs(response.confidence - ideal_conf)

        m_bonus = 1.0 if (
            ground_truth.method is not None
            and response.method == ground_truth.method
        ) else 0.0

        w = self.weights
        return w.alpha * iou + w.beta * c_cal + w.gamma * m_bonus

    def score_consensus(
        self,
        responses: list[MinerResponse],
        miner_weights: dict[str, float],
    ) -> dict[str, float]:
        if not responses:
            return {}

        tampered_weight = 0.0
        authentic_weight = 0.0
        for r in responses:
            w = miner_weights.get(r.task_id, 0.5)
            if r.verdict == Verdict.TAMPERED:
                tampered_weight += w
            else:
                authentic_weight += w

        majority = (
            Verdict.TAMPERED if tampered_weight >= authentic_weight
            else Verdict.AUTHENTIC
        )

        return {
            r.task_id: 1.0 if r.verdict == majority else 0.0
            for r in responses
        }

    def score_epoch(
        self,
        probe_scores: list[float],
        consensus_scores: list[float],
        avg_latency_ms: float,
        max_latency_ms: float,
        miner_id: str = "",
    ) -> ScoreResult:
        avg_probe = (
            sum(probe_scores) / len(probe_scores) if probe_scores else 0.0
        )
        avg_consensus = (
            sum(consensus_scores) / len(consensus_scores)
            if consensus_scores
            else 0.0
        )
        s_latency = max(0.0, 1.0 - avg_latency_ms / max_latency_ms) if max_latency_ms > 0 else 0.0

        w = self.weights
        total = (
            w.omega_probe * avg_probe
            + w.omega_consensus * avg_consensus
            + w.omega_latency * s_latency
        )

        return ScoreResult(
            miner_id=miner_id,
            probe_score=avg_probe,
            consensus_score=avg_consensus,
            latency_score=s_latency,
            total_score=total,
            details={
                "num_probes": len(probe_scores),
                "num_consensus": len(consensus_scores),
                "avg_latency_ms": avg_latency_ms,
                "max_latency_ms": max_latency_ms,
            },
        )

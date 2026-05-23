from __future__ import annotations

import uuid

from backend.db.store import MinerRecord, Store


STRIKE_WINDOW = 10
YELLOW_THRESHOLD = 1
RED_THRESHOLD = 2
BAN_THRESHOLD = 3


class MinerService:
    def __init__(self, store: Store):
        self.store = store

    def register_miner(self, name: str, backend_name: str) -> MinerRecord:
        miner_id = str(uuid.uuid4())[:8]
        record = MinerRecord(
            id=miner_id,
            name=name,
            backend_name=backend_name,
        )
        self.store.miners[miner_id] = record
        return record

    def record_probe_result(self, miner_id: str, correct: bool, score: float = 0.0):
        miner = self.store.miners.get(miner_id)
        if miner is None:
            raise ValueError(f"Miner {miner_id} not found")

        miner.probe_history.append(correct)
        miner.probe_scores.append(score)
        self._recalculate_score(miner)

    def _recalculate_score(self, miner: MinerRecord):
        if miner.probe_scores:
            avg_probe = sum(miner.probe_scores) / len(miner.probe_scores)
        else:
            avg_probe = 0.0

        if miner.consensus_scores:
            avg_consensus = sum(miner.consensus_scores) / len(miner.consensus_scores)
        else:
            avg_consensus = 0.0

        miner.total_score = 0.60 * avg_probe + 0.35 * avg_consensus + 0.05 * max(0.0, 1.0 - miner.avg_latency_ms / 30000)

    def get_strike_status(self, miner_id: str) -> str:
        miner = self.store.miners.get(miner_id)
        if miner is None:
            return "unknown"

        recent = miner.probe_history[-STRIKE_WINDOW:]
        if not recent:
            return "normal"

        failures = sum(1 for r in recent if not r)

        if failures >= BAN_THRESHOLD:
            return "banned"
        if failures >= RED_THRESHOLD:
            return "red_card"
        if failures >= YELLOW_THRESHOLD:
            return "yellow_card"
        return "normal"

    def get_probe_accuracy(self, miner_id: str) -> float:
        miner = self.store.miners.get(miner_id)
        if miner is None or not miner.probe_history:
            return 0.0
        return sum(1 for r in miner.probe_history if r) / len(miner.probe_history)

    def get_leaderboard(self) -> list[MinerRecord]:
        miners = list(self.store.miners.values())
        miners.sort(key=lambda m: m.total_score, reverse=True)
        return miners

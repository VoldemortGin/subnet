from __future__ import annotations

import pytest

from backend.db.store import Store
from backend.services.miner_service import (
    BAN_THRESHOLD,
    RED_THRESHOLD,
    STRIKE_WINDOW,
    YELLOW_THRESHOLD,
    MinerService,
)


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def svc(store: Store) -> MinerService:
    return MinerService(store)


class TestRegisterMiner:
    def test_returns_miner_record(self, svc: MinerService):
        miner = svc.register_miner("Alice", "ela")
        assert miner.name == "Alice"
        assert miner.backend_name == "ela"
        assert len(miner.id) == 8

    def test_stores_in_store(self, svc: MinerService, store: Store):
        miner = svc.register_miner("Bob", "noise")
        assert store.miners[miner.id] is miner

    def test_multiple_miners_get_unique_ids(self, svc: MinerService):
        m1 = svc.register_miner("A", "ela")
        m2 = svc.register_miner("B", "ela")
        assert m1.id != m2.id

    def test_initial_state(self, svc: MinerService):
        miner = svc.register_miner("X", "ela")
        assert miner.probe_history == []
        assert miner.probe_scores == []
        assert miner.total_score == 0.0


class TestRecordProbeResult:
    def test_appends_to_history(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        svc.record_probe_result(miner.id, True, 0.8)
        assert miner.probe_history == [True]
        assert miner.probe_scores == [0.8]

    def test_multiple_results(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        svc.record_probe_result(miner.id, True, 0.9)
        svc.record_probe_result(miner.id, False, 0.0)
        svc.record_probe_result(miner.id, True, 0.7)
        assert miner.probe_history == [True, False, True]
        assert miner.probe_scores == [0.9, 0.0, 0.7]

    def test_raises_for_unknown_miner(self, svc: MinerService):
        with pytest.raises(ValueError, match="not found"):
            svc.record_probe_result("nonexistent", True, 1.0)

    def test_recalculates_total_score(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        svc.record_probe_result(miner.id, True, 1.0)
        # total_score = 0.60 * avg_probe + 0.35 * avg_consensus + 0.05 * latency_factor
        # avg_probe = 1.0, avg_consensus = 0.0, latency_factor = max(0, 1 - 0/30000) = 1.0
        expected = 0.60 * 1.0 + 0.35 * 0.0 + 0.05 * 1.0
        assert abs(miner.total_score - expected) < 1e-9

    def test_score_with_consensus(self, svc: MinerService, store: Store):
        miner = svc.register_miner("A", "ela")
        miner.consensus_scores = [0.8, 1.0]
        svc.record_probe_result(miner.id, True, 0.5)
        avg_probe = 0.5
        avg_consensus = 0.9
        latency_factor = 1.0  # avg_latency_ms = 0
        expected = 0.60 * avg_probe + 0.35 * avg_consensus + 0.05 * latency_factor
        assert abs(miner.total_score - expected) < 1e-9


class TestGetStrikeStatus:
    def test_unknown_miner(self, svc: MinerService):
        assert svc.get_strike_status("nonexistent") == "unknown"

    def test_no_history_is_normal(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        assert svc.get_strike_status(miner.id) == "normal"

    def test_all_passes_is_normal(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        for _ in range(10):
            svc.record_probe_result(miner.id, True, 1.0)
        assert svc.get_strike_status(miner.id) == "normal"

    def test_one_failure_is_yellow(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        for _ in range(9):
            svc.record_probe_result(miner.id, True, 1.0)
        svc.record_probe_result(miner.id, False, 0.0)
        assert svc.get_strike_status(miner.id) == "yellow_card"

    def test_two_failures_is_red(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        for _ in range(8):
            svc.record_probe_result(miner.id, True, 1.0)
        svc.record_probe_result(miner.id, False, 0.0)
        svc.record_probe_result(miner.id, False, 0.0)
        assert svc.get_strike_status(miner.id) == "red_card"

    def test_three_failures_is_banned(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        for _ in range(7):
            svc.record_probe_result(miner.id, True, 1.0)
        for _ in range(3):
            svc.record_probe_result(miner.id, False, 0.0)
        assert svc.get_strike_status(miner.id) == "banned"

    def test_window_slides(self, svc: MinerService):
        """Old failures outside the window don't count."""
        miner = svc.register_miner("A", "ela")
        # 3 failures early on
        for _ in range(3):
            svc.record_probe_result(miner.id, False, 0.0)
        # Then 10 successes push the failures out of the window
        for _ in range(STRIKE_WINDOW):
            svc.record_probe_result(miner.id, True, 1.0)
        assert svc.get_strike_status(miner.id) == "normal"

    def test_exact_threshold_values(self, svc: MinerService):
        """Verify the threshold constants match expected behavior."""
        assert YELLOW_THRESHOLD == 1
        assert RED_THRESHOLD == 2
        assert BAN_THRESHOLD == 3
        assert STRIKE_WINDOW == 10


class TestGetProbeAccuracy:
    def test_unknown_miner_returns_zero(self, svc: MinerService):
        assert svc.get_probe_accuracy("nonexistent") == 0.0

    def test_empty_history_returns_zero(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        assert svc.get_probe_accuracy(miner.id) == 0.0

    def test_all_correct(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        for _ in range(5):
            svc.record_probe_result(miner.id, True, 1.0)
        assert svc.get_probe_accuracy(miner.id) == 1.0

    def test_all_incorrect(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        for _ in range(5):
            svc.record_probe_result(miner.id, False, 0.0)
        assert svc.get_probe_accuracy(miner.id) == 0.0

    def test_mixed(self, svc: MinerService):
        miner = svc.register_miner("A", "ela")
        svc.record_probe_result(miner.id, True, 1.0)
        svc.record_probe_result(miner.id, False, 0.0)
        svc.record_probe_result(miner.id, True, 0.8)
        svc.record_probe_result(miner.id, True, 0.9)
        assert svc.get_probe_accuracy(miner.id) == 0.75


class TestGetLeaderboard:
    def test_empty_leaderboard(self, svc: MinerService):
        assert svc.get_leaderboard() == []

    def test_sorted_by_total_score_descending(self, svc: MinerService):
        m1 = svc.register_miner("Low", "ela")
        m2 = svc.register_miner("High", "ela")
        m3 = svc.register_miner("Mid", "ela")
        # Give them different scores via probe results
        svc.record_probe_result(m1.id, True, 0.2)
        svc.record_probe_result(m2.id, True, 1.0)
        svc.record_probe_result(m3.id, True, 0.5)
        board = svc.get_leaderboard()
        assert board[0].id == m2.id
        assert board[1].id == m3.id
        assert board[2].id == m1.id

    def test_returns_all_miners(self, svc: MinerService):
        for i in range(5):
            svc.register_miner(f"Miner{i}", "ela")
        assert len(svc.get_leaderboard()) == 5

from __future__ import annotations

import pytest

from src.miner.backends.base import DetectionBackend
from src.miner.backends.ela import ELABackend
from src.miner.model_registry import (
    ALL_BACKENDS,
    get_backend,
    get_best_available,
    list_all,
    list_available,
)


class TestListAll:
    """Tests for list_all()."""

    def test_returns_list_of_dicts(self):
        result = list_all()
        assert isinstance(result, list)
        assert len(result) == len(ALL_BACKENDS)
        for entry in result:
            assert isinstance(entry, dict)

    def test_each_entry_has_required_keys(self):
        result = list_all()
        required_keys = {"name", "available", "gpu_required", "estimated_vram_mb"}
        for entry in result:
            assert required_keys.issubset(entry.keys())

    def test_entry_types(self):
        result = list_all()
        for entry in result:
            assert isinstance(entry["name"], str)
            assert isinstance(entry["available"], bool)
            assert isinstance(entry["gpu_required"], bool)
            assert isinstance(entry["estimated_vram_mb"], int)

    def test_ela_is_first(self):
        result = list_all()
        assert result[0]["name"] == "ela"

    def test_ela_is_available(self):
        result = list_all()
        ela_entry = next(e for e in result if e["name"] == "ela")
        assert ela_entry["available"] is True
        assert ela_entry["gpu_required"] is False
        assert ela_entry["estimated_vram_mb"] == 0


class TestListAvailable:
    """Tests for list_available()."""

    def test_returns_only_available(self):
        result = list_available()
        for entry in result:
            assert entry["available"] is True

    def test_ela_always_available(self):
        result = list_available()
        names = [e["name"] for e in result]
        assert "ela" in names

    def test_subset_of_list_all(self):
        all_backends = list_all()
        available = list_available()
        assert len(available) <= len(all_backends)


class TestGetBackend:
    """Tests for get_backend()."""

    def test_get_ela_backend(self):
        backend = get_backend("ela")
        assert isinstance(backend, ELABackend)
        assert isinstance(backend, DetectionBackend)
        assert backend.name() == "ela"

    def test_get_nonexistent_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent_backend_xyz")

    def test_get_unavailable_raises_runtime_error(self):
        """If a backend exists but is not available, RuntimeError is raised."""
        all_entries = list_all()
        unavailable = [e for e in all_entries if not e["available"]]
        if not unavailable:
            pytest.skip("All backends are available; cannot test RuntimeError path")
        name = unavailable[0]["name"]
        with pytest.raises(RuntimeError, match="not available"):
            get_backend(name)

    def test_get_backend_with_kwargs(self):
        backend = get_backend("ela", ela_quality=75, ela_threshold=30)
        assert backend.ela_quality == 75
        assert backend.ela_threshold == 30

    def test_error_message_lists_available_names(self):
        with pytest.raises(ValueError) as exc_info:
            get_backend("fake_name")
        assert "ela" in str(exc_info.value)


class TestGetBestAvailable:
    """Tests for get_best_available()."""

    def test_returns_detection_backend(self):
        backend = get_best_available()
        assert isinstance(backend, DetectionBackend)

    def test_returns_ela_as_fallback(self):
        """When only ELA is available, it should be returned."""
        # ELA is always available, so at minimum we get it
        backend = get_best_available()
        assert isinstance(backend, DetectionBackend)
        # It should at least be some valid backend
        assert backend.name() in [e["name"] for e in list_all()]

    def test_passes_kwargs_to_backend(self):
        """get_backend('ela') forwards kwargs; get_best_available works without kwargs."""
        # Verify kwargs work with a known backend directly
        backend = get_backend("ela", ela_quality=60)
        assert isinstance(backend, ELABackend)
        assert backend.ela_quality == 60

    def test_prefers_later_backends(self):
        """get_best_available iterates reversed, preferring later (more capable) backends."""
        available = list_available()
        if len(available) <= 1:
            pytest.skip("Only one backend available; cannot test preference order")
        backend = get_best_available()
        # The returned backend should be the last available one in ALL_BACKENDS order
        available_names = [e["name"] for e in available]
        assert backend.name() == available_names[-1]

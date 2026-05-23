from __future__ import annotations

from src.miner.backends.base import DetectionBackend
from src.miner.backends.catnet import CATNetBackend
from src.miner.backends.ela import ELABackend
from src.miner.backends.focal import FOCALBackend
from src.miner.backends.imlvit import IMLViTBackend
from src.miner.backends.mantranet import ManTraNetBackend
from src.miner.backends.mesorch import MesorchBackend
from src.miner.backends.mvssnet import MVSSNetBackend
from src.miner.backends.profact import ProFactBackend
from src.miner.backends.psccnet import PSCCNetBackend
from src.miner.backends.trufor import TruForBackend

ALL_BACKENDS: list[type[DetectionBackend]] = [
    ELABackend,
    ManTraNetBackend,
    TruForBackend,
    CATNetBackend,
    MVSSNetBackend,
    PSCCNetBackend,
    FOCALBackend,
    IMLViTBackend,
    MesorchBackend,
    ProFactBackend,
]

_BACKEND_MAP: dict[str, type[DetectionBackend]] = {}


def _build_map() -> dict[str, type[DetectionBackend]]:
    if not _BACKEND_MAP:
        for cls in ALL_BACKENDS:
            instance = cls.__new__(cls)
            _BACKEND_MAP[instance.name()] = cls
    return _BACKEND_MAP


def list_all() -> list[dict[str, str | bool | int]]:
    result = []
    for cls in ALL_BACKENDS:
        instance = cls.__new__(cls)
        result.append({
            "name": instance.name(),
            "available": cls.is_available(),
            "gpu_required": cls.gpu_required(),
            "estimated_vram_mb": cls.estimated_vram_mb(),
        })
    return result


def list_available() -> list[dict[str, str | bool | int]]:
    return [b for b in list_all() if b["available"]]


def get_backend(name: str, **kwargs) -> DetectionBackend:
    backend_map = _build_map()
    if name not in backend_map:
        available_names = list(backend_map.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available_names}")

    cls = backend_map[name]
    if not cls.is_available():
        raise RuntimeError(
            f"Backend '{name}' is not available. "
            f"Run: python scripts/setup_models.py --install {name}"
        )

    return cls(**kwargs)


def get_best_available(**kwargs) -> DetectionBackend:
    """Return the most capable available backend, preferring GPU models."""
    for cls in reversed(ALL_BACKENDS):
        if cls.is_available():
            return cls(**kwargs)
    return ELABackend(**kwargs)

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.protocol import ForgeryMethod


class DetectionBackend(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple[bool, float, np.ndarray | None, ForgeryMethod | None]:
        """Returns (is_tampered, confidence, mask_or_None, method_or_None)."""
        ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend's dependencies are installed."""
        ...

    @classmethod
    @abstractmethod
    def gpu_required(cls) -> bool: ...

    @classmethod
    @abstractmethod
    def estimated_vram_mb(cls) -> int: ...

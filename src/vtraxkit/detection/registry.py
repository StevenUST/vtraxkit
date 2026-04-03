"""Backend registry for detectors."""

from __future__ import annotations

from typing import Any

from .base import DetectorBackend

_DETECTORS: dict[str, type[DetectorBackend]] = {}


def register_detector(name: str):
    """Decorator to register a detector backend."""
    def decorator(cls: type[DetectorBackend]) -> type[DetectorBackend]:
        _DETECTORS[name] = cls
        return cls
    return decorator


def get_detector(name: str | DetectorBackend, **kwargs: Any) -> DetectorBackend:
    """Get a detector by name or pass through an existing instance.

    Supports ``"yolo:yolov8s.pt"`` syntax for model variant.
    """
    if isinstance(name, DetectorBackend):
        return name

    variant = None
    if ":" in name:
        name, variant = name.split(":", 1)
        kwargs.setdefault("model_name", variant)

    if name not in _DETECTORS:
        # Try auto-importing built-in backends
        if name == "yolo":
            from . import yolo  # noqa: F401
        if name not in _DETECTORS:
            available = list(_DETECTORS.keys())
            raise ValueError(f"Unknown detector: {name!r}. Available: {available}")

    return _DETECTORS[name](**kwargs)


def available_detectors() -> list[str]:
    return list(_DETECTORS.keys())

"""Backend registry for pose estimators."""

from __future__ import annotations

from typing import Any

from .base import PoseBackend

_POSE_BACKENDS: dict[str, type[PoseBackend]] = {}


def register_pose(name: str):
    """Decorator to register a pose backend."""
    def decorator(cls: type[PoseBackend]) -> type[PoseBackend]:
        _POSE_BACKENDS[name] = cls
        return cls
    return decorator


def get_pose_backend(name: str | PoseBackend, **kwargs: Any) -> PoseBackend:
    """Get a pose backend by name or pass through an existing instance."""
    if isinstance(name, PoseBackend):
        return name

    if name not in _POSE_BACKENDS:
        if name == "mediapipe":
            from . import mediapipe_backend  # noqa: F401
        if name not in _POSE_BACKENDS:
            available = list(_POSE_BACKENDS.keys())
            raise ValueError(f"Unknown pose backend: {name!r}. Available: {available}")

    return _POSE_BACKENDS[name](**kwargs)


def available_pose_backends() -> list[str]:
    return list(_POSE_BACKENDS.keys())

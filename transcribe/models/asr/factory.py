"""Factory for creating ASR backend instances."""

from __future__ import annotations

from transcribe.models.asr.base import ASRBase

_BACKENDS: dict[str, type[ASRBase]] = {}


def register_backend(name: str, cls: type[ASRBase]) -> None:
    """Register an ASR backend class under the given name."""
    _BACKENDS[name] = cls


def create_asr(
    backend: str,
    device: str = "cpu",
    hotword_path: str | None = None,
    **kwargs,
) -> ASRBase:
    """Create an ASR transcriber instance by backend name.

    Args:
        backend: Backend name (e.g. ``"Fun-ASR-Nano"``).
        device: Compute device (``"cpu"`` or ``"cuda"``).
        hotword_path: Path to hotword file, or None.
        **kwargs: Backend-specific keyword arguments.

    Returns:
        An ASR transcriber instance.

    Raises:
        ValueError: If the backend name is not registered.
    """
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown ASR backend: {backend!r}. "
            f"Available: {list(_BACKENDS.keys())}"
        )
    return cls(device=device, hotword_path=hotword_path, **kwargs)


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    return list(_BACKENDS.keys())

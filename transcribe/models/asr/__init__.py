"""ASR backend package.

Provides :class:`ASRBase`, :func:`create_asr`, and :func:`list_backends`.
Backend modules are imported here to trigger self-registration.
"""

from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import create_asr, list_backends
from transcribe.models.asr.utils import parse_timestamps, restore_hotwords

# Import backend modules to trigger register_backend() calls
from transcribe.models.asr import funasr_nano, funasr_paraformer, qwen3_asr  # noqa: F401

__all__ = [
    "ASRBase",
    "create_asr",
    "list_backends",
    "parse_timestamps",
    "restore_hotwords",
]

"""Abstract base class for ASR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from transcribe.data.types import AudioSegment, TranscriptSegment, WordTimestamp


class ASRBase(ABC):
    """Unified interface for all ASR backends.

    Subclasses must implement ``__init__``, ``transcribe``, and ``cleanup``.
    The ``__init__`` signature must accept ``device`` and ``hotword_path``
    as the first two positional arguments (after ``self``), with backend-specific
    parameters passed via ``**kwargs``.
    """

    @abstractmethod
    def __init__(self, device: str, hotword_path: str | None, **kwargs) -> None:
        ...

    @abstractmethod
    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps."""
        ...

    def transcribe_words(self, audio: AudioSegment) -> list[WordTimestamp]:
        """Return word-level timestamps.

        Default implementation derives from transcribe() output.
        Subclasses should override for finer granularity.
        """
        segments = self.transcribe(audio)
        return [
            WordTimestamp(word=seg.text, start_time=seg.start_time, end_time=seg.end_time)
            for seg in segments
        ]

    @abstractmethod
    def cleanup(self) -> None:
        """Release model from GPU/CPU memory."""
        ...

    @property
    def supports_hotwords(self) -> bool:
        """Whether this backend supports hotword boosting."""
        return True

"""Abstract base class for ASR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from transcribe.data.types import AudioSegment, TranscriptSegment, WordTimestamp


class ASRBase(ABC):
    """Unified interface for all ASR backends.

    Subclasses must implement ``__init__``, ``transcribe_words``, and
    ``cleanup``.  The ``__init__`` signature must accept ``device`` and
    ``hotword_path`` as the first two positional arguments (after ``self``),
    with backend-specific parameters passed via ``**kwargs``.
    """

    @abstractmethod
    def __init__(self, device: str, hotword_path: str | None, **kwargs) -> None:
        ...

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps.

        Default implementation derives from ``transcribe_words()`` output.
        Subclasses may override if they need segment-level logic.
        """
        words = self.transcribe_words(audio)
        if not words:
            return []

        segments: list[TranscriptSegment] = []
        buf = [words[0]]

        for w in words[1:]:
            # Merge words into segments at natural boundaries (gaps > 0.5s)
            if w.start_time - buf[-1].end_time > 0.5:
                segments.append(TranscriptSegment(
                    speaker_id="SPEAKER_00",
                    start_time=buf[0].start_time,
                    end_time=buf[-1].end_time,
                    text="".join(ww.word for ww in buf),
                ))
                buf = [w]
            else:
                buf.append(w)

        if buf:
            segments.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=buf[0].start_time,
                end_time=buf[-1].end_time,
                text="".join(ww.word for ww in buf),
            ))

        return segments

    @abstractmethod
    def transcribe_words(self, audio: AudioSegment) -> list[WordTimestamp]:
        """Return word-level timestamps. Subclasses must implement."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release model from GPU/CPU memory."""
        ...

    @property
    def provides_segments(self) -> bool:
        """Whether :meth:`transcribe` returns subtitle-ready segments.

        When ``True``, the pipeline calls :meth:`transcribe` directly and
        skips :class:`~transcribe.models.segmentation.SubtitleSegmenter`.
        Speaker attribution uses simple dominant-speaker voting only
        (no turn-boundary splitting).
        """
        return False

    @property
    def supports_hotwords(self) -> bool:
        """Whether this backend supports hotword boosting."""
        return True

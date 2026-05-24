"""Whisper ASR backend using faster-whisper (CTranslate2)."""

from __future__ import annotations

from pathlib import Path

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend


class WhisperTranscriber(ASRBase):
    """Multi-language speech recognition using faster-whisper.

    Uses CTranslate2 for efficient inference with native hotword support
    and built-in Silero VAD for automatic segmentation. Supports 99 languages
    with automatic language detection.

    Requires the ``whisper`` optional dependency group.
    """

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        *,
        model_size: str = "large-v3",
        language: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "Whisper 后端需要 faster-whisper 包。"
                "请运行: uv sync --extra whisper"
            )

        self._device = device
        self._language = language
        self._hotwords = self._load_hotwords(hotword_path)

        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        self._model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
        )

    @property
    def supports_hotwords(self) -> bool:
        return True

    def _load_hotwords(self, path: str | None) -> str | None:
        """Read hotword file and comma-join for faster-whisper ``hotwords`` param.

        Args:
            path: Path to hotword file (one word per line), or None.

        Returns:
            Comma-joined hotword string, or None if no hotwords.
        """
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        words = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return ",".join(words) if words else None

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps."""
        segments_iter, info = self._model.transcribe(
            audio.waveform,
            language=self._language,
            hotwords=self._hotwords,
            vad_filter=True,
        )

        segments: list[TranscriptSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    speaker_id="SPEAKER_00",
                    start_time=seg.start + audio.start_time,
                    end_time=seg.end + audio.start_time,
                    text=text,
                )
            )

        return segments

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


register_backend("Whisper", WhisperTranscriber)

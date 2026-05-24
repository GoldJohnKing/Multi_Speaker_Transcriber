"""Whisper ASR backend using faster-whisper (CTranslate2)."""

from __future__ import annotations

from pathlib import Path

from transcribe.data.types import AudioSegment, TranscriptSegment, WordTimestamp
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend
from transcribe.models.asr.utils import restore_hotwords


class WhisperTranscriber(ASRBase):
    """Speech recognition using faster-whisper with hotword support.

    Uses the CTranslate2-based faster-whisper library for efficient Whisper
    inference.  Supports word-level timestamps and hotwords via the native
    ``hotwords`` parameter (prompt prefix injection through ``<|startofprev|>``
    token, available since faster-whisper v1.0.2).

    The default model is ``large-v3`` for best Chinese transcription quality.
    """

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        *,
        model_name: str = "large-v3",
        compute_type: str | None = None,
        language: str = "zh",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "Whisper 后端需要 faster-whisper 包。"
                "请运行: uv sync"
            )

        self._device = device
        self._language = language
        self._hotword_str, self._hotword_list = self._load_hotwords(hotword_path)

        # Determine compute type based on device
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    @staticmethod
    def _load_hotwords(path: str | None) -> tuple[str | None, list[str]]:
        """Load hotwords from text file (one per line), space-joined.

        faster-whisper accepts hotwords as a single space-separated string.

        Args:
            path: Path to hotword file, or None.

        Returns:
            Tuple of (space-joined hotword string or None, list of individual words).
        """
        if path is None:
            return None, []
        p = Path(path)
        if not p.exists():
            return None, []
        words = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return (" ".join(words) if words else None), words

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps."""
        segments_iter, _info = self._model.transcribe(
            audio.waveform,
            language=self._language,
            hotwords=self._hotword_str,
            word_timestamps=False,
        )

        result: list[TranscriptSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            # Restore hotword terms broken by Whisper punctuation
            text = restore_hotwords(text, self._hotword_list)

            result.append(TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=seg.start + audio.start_time,
                end_time=seg.end + audio.start_time,
                text=text,
            ))

        return result

    def transcribe_words(self, audio: AudioSegment) -> list[WordTimestamp]:
        """Return word-level timestamps from faster-whisper output."""
        segments_iter, _info = self._model.transcribe(
            audio.waveform,
            language=self._language,
            hotwords=self._hotword_str,
            word_timestamps=True,
        )

        words: list[WordTimestamp] = []
        for seg in segments_iter:
            if seg.words:
                for w in seg.words:
                    word_text = restore_hotwords(w.word.strip(), self._hotword_list)
                    if word_text:
                        words.append(WordTimestamp(
                            word=word_text,
                            start_time=w.start + audio.start_time,
                            end_time=w.end + audio.start_time,
                        ))
            else:
                # Fallback: segment without word-level timestamps
                text = seg.text.strip()
                if text:
                    text = restore_hotwords(text, self._hotword_list)
                    words.append(WordTimestamp(
                        word=text,
                        start_time=seg.start + audio.start_time,
                        end_time=seg.end + audio.start_time,
                    ))

        return words

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu":
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


register_backend("Whisper", WhisperTranscriber)

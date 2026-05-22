"""Fun-ASR (Paraformer) ASR backend with hotword support."""

from __future__ import annotations

from pathlib import Path

import torch
from funasr import AutoModel

from transcribe.data.types import AudioSegment, TranscriptSegment, WordTimestamp
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend
from transcribe.models.asr.utils import parse_timestamps, restore_hotwords


class FunASRParaformerTranscriber(ASRBase):
    """Speech recognition using FunASR SeACo-Paraformer with hotword support.

    Uses the paraformer-zh model with a separate ct-punc punctuation model.
    Timestamps are in Paraformer format (milliseconds, nested or flat lists).
    """

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        model_name: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
    ) -> None:
        self._device = device
        self._hotwords, self._hotword_list = self._load_hotwords(hotword_path)
        self._model = AutoModel(
            model=model_name,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
        )

    def _load_hotwords(
        self, path: str | None
    ) -> tuple[str | None, list[str]]:
        """Load hotwords from text file (one per line), space-joined.

        Paraformer accepts hotwords as a single space-joined string via the
        ``hotword`` parameter.

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
        result = self._model.generate(
            input=audio.waveform,
            batch_size_s=300,
            hotword=self._hotwords,
        )

        segments: list[TranscriptSegment] = []
        if not result:
            return segments

        for res in result:
            text = res.get("text", "")
            if not text:
                continue

            timestamps = res.get("timestamp", [])

            # Restore hotword terms broken by ct-punc punctuation insertion
            text = restore_hotwords(text, self._hotword_list)

            # Parse timestamps — handles nested/flat ms formats
            parsed_start, parsed_end = parse_timestamps(timestamps)

            if parsed_start is not None and parsed_end is not None:
                start_time = parsed_start + audio.start_time
                end_time = parsed_end + audio.start_time
            else:
                start_time = audio.start_time
                end_time = audio.end_time

            segments.append(
                TranscriptSegment(
                    speaker_id="SPEAKER_00",
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                )
            )

        return segments

    def transcribe_words(self, audio: AudioSegment) -> list[WordTimestamp]:
        """Override: return per-word timestamps from Paraformer ms-format output."""
        result = self._model.generate(
            input=audio.waveform,
            batch_size_s=300,
            hotword=self._hotwords,
        )

        words: list[WordTimestamp] = []
        if not result:
            return words

        for res in result:
            text = res.get("text", "")
            timestamps = res.get("timestamp", [])

            if (
                timestamps
                and isinstance(timestamps, list)
                and len(timestamps) > 0
                and isinstance(timestamps[0], (list, tuple))
                and len(timestamps) == len(text)
            ):
                # Paraformer format: [[start_ms, end_ms], ...] — one entry per char
                for i, ts in enumerate(timestamps):
                    ch = text[i]
                    words.append(WordTimestamp(
                        word=ch,
                        start_time=ts[0] / 1000.0 + audio.start_time,
                        end_time=ts[1] / 1000.0 + audio.start_time,
                    ))
            elif timestamps:
                # Timestamp count doesn't match text length — fallback
                text = restore_hotwords(text, self._hotword_list)
                parsed_start, parsed_end = parse_timestamps(timestamps)
                if parsed_start is not None:
                    words.append(WordTimestamp(
                        word=text,
                        start_time=parsed_start + audio.start_time,
                        end_time=parsed_end + audio.start_time,
                    ))
                else:
                    words.append(WordTimestamp(
                        word=text,
                        start_time=audio.start_time,
                        end_time=audio.end_time,
                    ))
            elif text:
                text = restore_hotwords(text, self._hotword_list)
                words.append(WordTimestamp(
                    word=text,
                    start_time=audio.start_time,
                    end_time=audio.end_time,
                ))

        return words

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


register_backend("Fun-ASR-Paraformer", FunASRParaformerTranscriber)

"""Qwen3-ASR backend with ForcedAligner for character-level timestamps."""

from __future__ import annotations

from pathlib import Path

import torch

from transcribe.data.types import AudioSegment, TranscriptSegment, WordTimestamp
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend
from transcribe.models.asr.utils import segment_by_timestamps

# Punctuation characters present in ASR text but absent from ForcedAligner timestamps.
_PUNCT = frozenset("，。！？,;:、；：…—")


def _align_text_to_timestamps(
    text: str,
    time_stamps: list,
    offset: float = 0.0,
) -> list[tuple[str, float, float]]:
    """Align *text* (with punctuation) to *time_stamps* (without punctuation).

    The Qwen3-ASR ``text`` field includes punctuation, but the ForcedAligner
    ``time_stamps`` skip punctuation characters.  This function reconciles the
    two by interpolating short (20 ms) timestamps for punctuation characters.

    Multi-character tokens in *time_stamps* (e.g. ``"NPC"``) are flattened and
    their durations distributed evenly across the constituent characters.

    Args:
        text: Full transcript text including punctuation.
        time_stamps: ForcedAligner output — objects with ``.text``,
            ``.start_time``, ``.end_time`` attributes.
        offset: Seconds to add to every timestamp (typically ``audio.start_time``).

    Returns:
        ``[(char, start_sec, end_sec), ...]`` suitable for
        :func:`segment_by_timestamps`.
    """
    # Step 1: Flatten multi-char tokens → individual (char, start, end).
    ts_flat: list[tuple[str, float, float]] = []
    for ts in time_stamps:
        n = len(ts.text)
        dur = ts.end_time - ts.start_time
        for i, ch in enumerate(ts.text):
            s = ts.start_time + dur * i / n
            e = ts.start_time + dur * (i + 1) / n
            ts_flat.append((ch, offset + s, offset + e))

    # Step 2: Walk through text, matching non-punct chars against ts_flat.
    aligned: list[tuple[str, float, float]] = []
    ts_idx = 0
    for ch in text:
        if ch in _PUNCT:
            prev_end = aligned[-1][2] if aligned else offset
            aligned.append((ch, prev_end, prev_end + 0.02))
        elif ts_idx < len(ts_flat):
            aligned.append(ts_flat[ts_idx])
            ts_idx += 1
        else:
            # Unexpected trailing char (shouldn't happen) — best-effort
            prev_end = aligned[-1][2] if aligned else offset
            aligned.append((ch, prev_end, prev_end + 0.02))

    return aligned


class Qwen3ASRTranscriber(ASRBase):
    """Speech recognition using Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B.

    Encapsulates both the ASR model and the forced aligner in a single
    ``ASRBase`` subclass.  Audio is passed as ``(np.ndarray, sample_rate)``
    tuples.  Hotwords are mapped to the ``context`` parameter (LLM system
    prompt biasing) rather than traditional weighted decoder biasing.

    Requires the ``qwen-asr`` optional dependency group.
    """

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        *,
        asr_model: str = "Qwen/Qwen3-ASR-1.7B",
        aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        language: str | None = None,
    ) -> None:
        # Deferred import — module must be importable without qwen-asr
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "Qwen3-ASR 后端需要 qwen-asr 包。"
                "请运行: uv sync --extra qwen-asr"
            )

        self._device = device
        self._language = language
        self._context = self._load_context(hotword_path)

        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        device_map = "cuda:0" if device == "cuda" else "cpu"

        self._model = Qwen3ASRModel.from_pretrained(
            asr_model,
            dtype=dtype,
            device_map=device_map,
            forced_aligner=aligner_model,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
            max_inference_batch_size=32,
            max_new_tokens=256,
        )

    @property
    def supports_hotwords(self) -> bool:
        return True

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with character-level timestamps."""
        results = self._model.transcribe(
            audio=(audio.waveform, audio.sample_rate),
            context=self._context,
            language=self._language,
            return_time_stamps=True,
        )

        if not results or not results[0].text:
            return []

        r = results[0]

        # Fallback: no timestamps → single segment covering entire audio
        if not r.time_stamps:
            return [TranscriptSegment(
                speaker_id="SPEAKER_00",
                start_time=audio.start_time,
                end_time=audio.end_time,
                text=r.text,
            )]

        # Align text (with punctuation) to time_stamps (without punctuation)
        char_ts = _align_text_to_timestamps(r.text, r.time_stamps, audio.start_time)

        return segment_by_timestamps(char_ts)

    def transcribe_words(self, audio: AudioSegment) -> list[WordTimestamp]:
        """Override: return character-level timestamps from ForcedAligner."""
        results = self._model.transcribe(
            audio=(audio.waveform, audio.sample_rate),
            context=self._context,
            language=self._language,
            return_time_stamps=True,
        )

        if not results or not results[0].text:
            return []

        r = results[0]
        if not r.time_stamps:
            # Fallback: single word covering entire audio
            return [WordTimestamp(
                word=r.text,
                start_time=audio.start_time,
                end_time=audio.end_time,
            )]

        char_ts = _align_text_to_timestamps(r.text, r.time_stamps, audio.start_time)
        return [
            WordTimestamp(word=ch, start_time=s, end_time=e)
            for ch, s, e in char_ts
        ]

    def cleanup(self) -> None:
        """Release ASR + ForcedAligner from GPU/CPU memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_context(self, path: str | None) -> str:
        """Read hotword file, join with spaces for Qwen3-ASR ``context`` param."""
        if not path:
            return ""
        p = Path(path)
        if not p.exists():
            return ""
        with open(p, encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip()]
        return " ".join(terms)


register_backend("Qwen3-ASR", Qwen3ASRTranscriber)

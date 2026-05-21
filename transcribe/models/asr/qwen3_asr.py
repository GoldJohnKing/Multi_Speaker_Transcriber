"""Qwen3-ASR backend with ForcedAligner for character-level timestamps."""

from __future__ import annotations

import torch

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend
from transcribe.models.asr.utils import segment_by_timestamps


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

        # Build character-level timestamp list with offset
        char_ts = [
            (ts.text, audio.start_time + ts.start_time, audio.start_time + ts.end_time)
            for ts in r.time_stamps
        ]

        return segment_by_timestamps(char_ts)

    def cleanup(self) -> None:
        """Release ASR + ForcedAligner from GPU/CPU memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_context(self, path: str | None) -> str:
        """Read hotword file, join with spaces for Qwen3-ASR ``context`` param."""
        if not path:
            return ""
        with open(path, encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip()]
        return " ".join(terms)


register_backend("Qwen3-ASR", Qwen3ASRTranscriber)

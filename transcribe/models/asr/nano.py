"""Fun-ASR-Nano ASR backend with hotword support."""

from __future__ import annotations

from pathlib import Path

import torch
from funasr import AutoModel

from transcribe.data.types import AudioSegment, TranscriptSegment
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import register_backend
from transcribe.models.asr.utils import parse_timestamps, restore_hotwords


class FunASRNanoTranscriber(ASRBase):
    """Speech recognition using Fun-ASR-Nano with hotword support.

    Uses the FunAudioLLM/Fun-ASR-Nano-2512 model which generates punctuation
    via LLM and provides timestamps in dict format (seconds).
    """

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        model_name: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model: str = "fsmn-vad",
    ) -> None:
        self._device = device
        self._hotword_list = self._load_hotwords(hotword_path)

        model_kwargs: dict = {
            "model": model_name,
            "trust_remote_code": True,
            "vad_model": vad_model,
            "vad_kwargs": {"max_single_segment_time": 30000},
            "device": device,
        }

        # BF16 for GPU (Ampere+) — fp16 is broken on CUDA (outputs all "!")
        if device != "cpu":
            if torch.cuda.is_bf16_supported():
                model_kwargs["bf16"] = True

        self._model = AutoModel(**model_kwargs)

    def _load_hotwords(self, path: str | None) -> list[str]:
        """Load hotwords from text file (one per line).

        Fun-ASR-Nano accepts hotwords as a list of strings.

        Args:
            path: Path to hotword file, or None.

        Returns:
            List of hotword strings. Empty list if path is None or missing.
        """
        if path is None:
            return []
        p = Path(path)
        if not p.exists():
            return []
        return [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps."""
        # Fun-ASR-Nano requires torch tensor wrapped in a list
        waveform_tensor = torch.from_numpy(audio.waveform)

        result = self._model.generate(
            input=[waveform_tensor],
            cache={},
            batch_size=1,
            language="中文",
            itn=True,
            hotwords=self._hotword_list if self._hotword_list else None,
        )

        segments: list[TranscriptSegment] = []
        if not result:
            return segments

        for res in result:
            text = res.get("text", "")
            if not text:
                continue

            # Restore hotword terms broken by LLM punctuation (safety net)
            text = restore_hotwords(text, self._hotword_list)

            # Parse timestamps — handles Fun-ASR-Nano dict format
            timestamps = res.get("timestamps", [])
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

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


register_backend("Fun-ASR-Nano", FunASRNanoTranscriber)

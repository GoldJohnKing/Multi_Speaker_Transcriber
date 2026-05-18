"""Stage 3: Speaker diarization using Pyannote Audio 3.1."""

from __future__ import annotations

import torch

from transcribe.data.types import AudioSegment, DiarizationResult, SpeakerSegment


class Diarizer:
    """Speaker diarization using Pyannote Audio."""

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: str | None = None,
        num_speakers: int | None = None,
    ) -> None:
        self._device = device
        self._num_speakers = num_speakers
        self._pipeline = self._load_pipeline(model_name, hf_token)

    def _load_pipeline(self, model_name: str, hf_token: str | None):
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=hf_token,
        )
        if self._device != "cpu" and torch.cuda.is_available():
            pipeline = pipeline.to(torch.device(self._device))
        return pipeline

    def process(self, audio: AudioSegment) -> DiarizationResult:
        """Run speaker diarization on audio."""
        waveform_tensor = torch.tensor(audio.waveform, dtype=torch.float32)
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)

        input_dict = {
            "waveform": waveform_tensor,
            "sample_rate": audio.sample_rate,
        }

        kwargs = {}
        if self._num_speakers is not None:
            kwargs["num_speakers"] = self._num_speakers

        diarization = self._pipeline(input_dict, **kwargs)

        segments: list[SpeakerSegment] = []
        overlap_regions: list[tuple[float, float]] = []
        speaker_set: set[str] = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_set.add(speaker)
            is_overlap = False

            for existing in segments:
                if (
                    existing.speaker_id != speaker
                    and existing.start_time < turn.end + audio.start_time
                    and existing.end_time > turn.start + audio.start_time
                ):
                    is_overlap = True
                    overlap_start = max(existing.start_time, turn.start + audio.start_time)
                    overlap_end = min(existing.end_time, turn.end + audio.start_time)
                    overlap_regions.append((overlap_start, overlap_end))

            segments.append(
                SpeakerSegment(
                    speaker_id=speaker,
                    start_time=turn.start + audio.start_time,
                    end_time=turn.end + audio.start_time,
                    is_overlap=is_overlap,
                )
            )

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speaker_set),
            overlap_regions=overlap_regions,
        )

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._pipeline
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

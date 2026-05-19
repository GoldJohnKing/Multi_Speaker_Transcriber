"""Stage 4: Speech separation for overlapping regions using SepFormer."""

from __future__ import annotations

import numpy as np
import torch

from transcribe.data.types import AudioSegment, DiarizationResult


class Separator:
    """Speech separation for overlapping regions using SepFormer."""

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "speechbrain/sepformer-whamr16k",
        max_segment_seconds: float = 10.0,
    ) -> None:
        self._device = device
        self._max_segment_seconds = max_segment_seconds
        self._model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load SepFormer model from SpeechBrain."""
        from speechbrain.inference.separation import SepformerSeparation

        model = SepformerSeparation.from_hparams(
            source=model_name,
            savedir=f"pretrained_models/{model_name.split('/')[-1]}",
            run_opts={"device": self._device},
        )
        return model

    def separate_overlaps(
        self, audio: AudioSegment, diarization: DiarizationResult
    ) -> list[AudioSegment]:
        """Separate overlapping speech regions into individual speaker tracks.

        Args:
            audio: Full audio segment.
            diarization: Diarization result with overlap regions.

        Returns:
            List of separated AudioSegments, each from an overlap region.
            Non-overlap regions are NOT included (handled directly by ASR).
            For each overlap region, 2 separated tracks are returned
            (one per estimated source).
        """
        if not diarization.overlap_regions:
            return []

        separated_segments: list[AudioSegment] = []

        for overlap_start, overlap_end in diarization.overlap_regions:
            # Clip duration limit
            clip_duration = min(
                overlap_end - overlap_start, self._max_segment_seconds
            )
            actual_end = overlap_start + clip_duration

            # Extract overlap audio chunk
            start_sample = int(
                (overlap_start - audio.start_time) * audio.sample_rate
            )
            end_sample = int(
                (actual_end - audio.start_time) * audio.sample_rate
            )
            # Clamp to valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio.waveform), end_sample)
            if end_sample <= start_sample:
                continue

            chunk = audio.waveform[start_sample:end_sample]

            # Convert to tensor — SepFormer expects (batch, time)
            wav_tensor = torch.tensor(chunk, dtype=torch.float32)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)  # add batch dim

            # Run separation
            est_sources = self._model.separate_batch(wav_tensor)

            # est_sources shape: (batch, time, num_speakers)
            separated = est_sources[0]  # (time, num_speakers)

            for spk_idx in range(separated.shape[-1]):
                spk_waveform = separated[:, spk_idx].cpu().numpy()
                spk_waveform = spk_waveform.astype(np.float32)

                separated_segments.append(
                    AudioSegment(
                        waveform=spk_waveform,
                        sample_rate=audio.sample_rate,
                        start_time=overlap_start,
                        end_time=actual_end,
                    )
                )

        return separated_segments

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

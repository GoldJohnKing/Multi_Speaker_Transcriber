"""Stage 4: Speech separation for overlapping regions using ClearVoice MossFormer2_SS_16K."""

from __future__ import annotations

import numpy as np
import torch

from transcribe.data.types import AudioSegment, DiarizationResult


class Separator:
    """Speech separation for overlapping regions using ClearVoice SS.

    Operates natively at 16kHz. Uses MossFormer2_SS_16K which outputs
    [num_spks, 1, T] shaped arrays — one track per separated speaker.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "MossFormer2_SS_16K",
        max_segment_seconds: float = 10.0,
    ) -> None:
        self._device = device
        self._max_segment_seconds = max_segment_seconds
        self._model_name = model_name
        self._model = self._load_model()

    def _load_model(self):
        """Load ClearVoice speech separation model."""
        from clearvoice import ClearVoice

        from transcribe.models.rocm_compat import patch_clearvoice_for_rocm

        patch_clearvoice_for_rocm()
        cv = ClearVoice(task="speech_separation", model_names=[self._model_name])
        return cv

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

            # ClearVoice expects [1, T] input
            input_array = chunk.reshape(1, -1).astype(np.float32)
            output_array = self._model(input_array, False)  # [num_spks, 1, T]

            # output_array shape: [num_spks, 1, T]
            for spk_idx in range(output_array.shape[0]):
                spk_waveform = output_array[spk_idx, 0, :].astype(np.float32)

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

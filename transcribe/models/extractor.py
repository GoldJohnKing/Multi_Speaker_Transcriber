"""Stage 4A: Target Speaker Extraction using ClearVoice AV_MossFormer2_TSE_16K.

Extracts individual speaker audio from overlapping speech using video face
tracking as visual cues. Requires video input (not pure audio).
"""

from __future__ import annotations

import torch

from transcribe.data.types import AudioSegment, DiarizationResult


class TargetSpeakerExtractor:
    """Target speaker extraction using ClearVoice TSE.

    Uses AV_MossFormer2_TSE_16K to extract individual speakers from mixed
    audio by leveraging visual face tracking from the input video.
    """

    def __init__(
        self,
        device: str = "cpu",
        model: str = "AV_MossFormer2_TSE_16K",
        min_track_frames: int = 50,
    ) -> None:
        self._device = device
        self._model_name = model
        self._min_track_frames = min_track_frames
        self._model = self._load_model()

    def _load_model(self):
        """Load ClearVoice TSE model."""
        from clearvoice import ClearVoice

        from transcribe.models.rocm_compat import patch_clearvoice_for_rocm

        patch_clearvoice_for_rocm()
        cv = ClearVoice(task="target_speaker_extraction", model_names=[self._model_name])
        return cv

    def extract(
        self,
        video_path: str,
        audio: AudioSegment,
        diarization: DiarizationResult,
    ) -> dict[str, AudioSegment]:
        """Extract each speaker's audio from video using face tracking.

        Args:
            video_path: Path to the input video file.
            audio: Denoised full audio segment (16kHz).
            diarization: Speaker diarization result.

        Returns:
            {face_track_id: AudioSegment} mapping. Each face track's
            extracted speaker audio. face_track_id is a string like "face_0".
        """
        import numpy as np

        # ClearVoice TSE takes the video path and audio as input
        # The model performs face detection and visual feature extraction internally
        input_waveform = audio.waveform
        if isinstance(input_waveform, torch.Tensor):
            input_waveform = input_waveform.cpu().numpy()

        input_array = input_waveform.reshape(1, -1).astype(np.float32)

        # ClearVoice TSE processes video + audio together
        # Returns a dict-like or structured output with per-face-track audio
        try:
            result = self._model(input_array, False, video_path=video_path)
        except Exception as e:
            raise RuntimeError(
                f"TSE extraction failed. Ensure video file contains visible faces: {e}"
            ) from e

        # Process results — ClearVoice TSE returns per-face-track audio
        face_tracks: dict[str, AudioSegment] = {}

        if isinstance(result, dict):
            # Result is {face_id: waveform_array}
            for face_id, waveform in result.items():
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy()
                waveform = np.ascontiguousarray(waveform, dtype=np.float32)
                if waveform.ndim > 1:
                    waveform = waveform.squeeze()

                face_tracks[str(face_id)] = AudioSegment(
                    waveform=waveform,
                    sample_rate=audio.sample_rate,
                    start_time=audio.start_time,
                    end_time=audio.end_time,
                )
        elif isinstance(result, (np.ndarray, torch.Tensor)):
            # Fallback: result is [num_faces, 1, T] array
            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()
            for idx in range(result.shape[0]):
                waveform = result[idx, 0, :].astype(np.float32)
                face_tracks[f"face_{idx}"] = AudioSegment(
                    waveform=waveform,
                    sample_rate=audio.sample_rate,
                    start_time=audio.start_time,
                    end_time=audio.end_time,
                )
        else:
            raise RuntimeError(
                f"Unexpected TSE output type: {type(result)}. "
                "ClearVoice TSE API may have changed."
            )

        return face_tracks

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

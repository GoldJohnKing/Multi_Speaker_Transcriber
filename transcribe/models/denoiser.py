"""Stage 2: Noise suppression using ClearVoice MossFormerGAN_SE_16K."""

from __future__ import annotations

import numpy as np
import torch

from transcribe.data.types import AudioSegment

# Default SNR threshold (dB) above which audio is considered clean enough
DEFAULT_SNR_THRESHOLD = 25.0


def estimate_snr(audio: AudioSegment, frame_duration: float = 0.03) -> float:
    """Estimate signal-to-noise ratio of audio using energy percentiles.

    Splits audio into short frames, estimates noise floor from the quietest
    frames and signal level from the loudest frames.

    Args:
        audio: Input audio segment.
        frame_duration: Frame length in seconds (default 30ms).

    Returns:
        Estimated SNR in dB. Higher values mean cleaner audio.
        Returns a large value (40.0) if noise floor is negligible.
    """
    frame_len = int(audio.sample_rate * frame_duration)
    if frame_len == 0 or len(audio.waveform) < frame_len:
        return 40.0

    n_frames = len(audio.waveform) // frame_len
    frames = audio.waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames**2, axis=1)

    noise_floor = np.percentile(energies, 10)
    speech_energy = np.percentile(energies, 90)

    if noise_floor < 1e-10:
        return 40.0

    return float(10 * np.log10(speech_energy / noise_floor))


class Denoiser:
    """Noise suppression using ClearVoice MossFormerGAN_SE_16K.

    Operates natively at 16kHz — no resampling needed.
    """

    def __init__(self, device: str = "cpu", model: str = "MossFormerGAN_SE_16K") -> None:
        self._device = device
        self._model_name = model
        self._model = self._load_model()

    def _load_model(self):
        """Load ClearVoice speech enhancement model."""
        from clearvoice import ClearVoice

        from transcribe.models.rocm_compat import patch_clearvoice_for_rocm

        patch_clearvoice_for_rocm()
        cv = ClearVoice(task="speech_enhancement", model_names=[self._model_name])
        return cv

    def process(self, audio: AudioSegment) -> AudioSegment:
        """Apply noise suppression to *audio*.

        Input must be at 16kHz (native ClearVoice SE sample rate).
        No resampling is performed — the audio is passed directly.
        """
        waveform = audio.waveform

        # Ensure 1-D float32 numpy array
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform[0]
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)

        # ClearVoice expects [1, T] input
        input_array = waveform.reshape(1, -1).astype(np.float32)
        output_array = self._model(input_array, False)  # [1, T]
        result = output_array.squeeze(0).astype(np.float32)

        return AudioSegment(
            waveform=result,
            sample_rate=audio.sample_rate,
            start_time=audio.start_time,
            end_time=audio.end_time,
        )

    def cleanup(self) -> None:
        """Release model from GPU memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

"""Stage 2: Noise suppression using DeepFilterNet v2."""

from __future__ import annotations

import sys
import types
from collections import namedtuple

import numpy as np
import torch
import torchaudio.functional as TA_F

from transcribe.data.types import AudioSegment


# ---------------------------------------------------------------------------
# DeepFilterNet / torchaudio compatibility shim
# torchaudio >= 2.11 removed ``torchaudio.backend.common.AudioMetaData`` which
# ``df.io`` imports unconditionally.  We patch it before any ``df`` import.
# ---------------------------------------------------------------------------
if "torchaudio.backend.common" not in sys.modules:
    _backend = types.ModuleType("torchaudio.backend.common")
    _backend.AudioMetaData = namedtuple(
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
    )
    sys.modules["torchaudio.backend.common"] = _backend

# Internal model sample rate (DeepFilterNet always uses 48 kHz)
_DF_SR = 48_000


class Denoiser:
    """Noise suppression using DeepFilterNet."""

    def __init__(self, device: str = "cpu", post_filter: bool = False, model: str = "DeepFilterNet2") -> None:
        self._device = device
        self._post_filter = post_filter
        self._model_name = model
        self._model, self._df_state = self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load DeepFilterNet model (downloads ~50 MB on first call)."""
        from df import init_df

        model, df_state, suffix = init_df(
            default_model=self._model_name,
            post_filter=self._post_filter,
            log_level="WARNING",
            log_file=None,
        )
        # Move model to requested device (init_df auto-detects, we override)
        if self._device and self._device != "cpu":
            model = model.to(self._device)
        else:
            model = model.to("cpu")
        return model, df_state

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, audio: AudioSegment) -> AudioSegment:
        """Apply noise suppression to *audio*.

        The input may be at any sample rate; it will be resampled to the
        model's internal rate (48 kHz) and back again so the caller does not
        need to care about rate conversion.
        """
        from df import enhance

        waveform = audio.waveform

        # Ensure 1-D float32 numpy array
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform[0]
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)

        # Resample to model rate if needed
        needs_resample = audio.sample_rate != _DF_SR
        if needs_resample:
            wav_t = torch.from_numpy(waveform).unsqueeze(0)  # [1, T]
            wav_t = TA_F.resample(wav_t, audio.sample_rate, _DF_SR)
        else:
            wav_t = torch.from_numpy(waveform).unsqueeze(0)  # [1, T]

        # DeepFilterNet expects Tensor[C, T]
        with torch.no_grad():
            enhanced = enhance(self._model, self._df_state, wav_t)

        # enhanced is Tensor[C, T] → squeeze to 1-D
        enhanced = enhanced.squeeze(0).cpu()

        # Resample back to original rate
        if needs_resample:
            enhanced = TA_F.resample(
                enhanced.unsqueeze(0), _DF_SR, audio.sample_rate
            ).squeeze(0)

        result_np = np.ascontiguousarray(enhanced.numpy(), dtype=np.float32)

        return AudioSegment(
            waveform=result_np,
            sample_rate=audio.sample_rate,
            start_time=audio.start_time,
            end_time=audio.end_time,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release model from GPU memory."""
        del self._model
        del self._df_state
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

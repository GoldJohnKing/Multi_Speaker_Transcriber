"""Stage 3: Speaker diarization using Pyannote Audio."""

from __future__ import annotations

import os
import sys
import types

import torch

from transcribe.data.types import AudioSegment, DiarizationResult, SpeakerSegment


# ---------------------------------------------------------------------------
# Pyannote / torchaudio compatibility shim
#
# torchaudio >= 2.9 removed ``AudioMetaData`` and ``list_audio_backends``
# which pyannote.audio 3.x imports unconditionally.  We monkey-patch them
# back before any pyannote import.  Idempotent — safe to call repeatedly.
#
# Reference: ebook2audiobook project (Apache-2.0)
# https://github.com/DrewThomasson/ebook2audiobook/blob/main/lib/classes/background_detector.py  # noqa: E501
# ---------------------------------------------------------------------------
_pyannote_patched = False


def _patch_torchaudio_for_pyannote() -> None:
    global _pyannote_patched
    if _pyannote_patched:
        return

    import torchaudio

    # --- AudioMetaData (class, not namedtuple — matches torchaudio API) ---
    if not hasattr(torchaudio, "AudioMetaData"):

        class _AudioMetaData:
            def __init__(
                self,
                sample_rate: int = 0,
                num_frames: int = 0,
                num_channels: int = 0,
                bits_per_sample: int = 0,
                encoding: str = "UNKNOWN",
            ) -> None:
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        torchaudio.AudioMetaData = _AudioMetaData  # type: ignore[attr-defined]

    # --- list_audio_backends ---
    if not hasattr(torchaudio, "list_audio_backends"):

        def _list_audio_backends() -> list[str]:
            backends: list[str] = []
            try:
                from torchaudio.utils import ffmpeg_utils  # type: ignore[import-untyped]

                if ffmpeg_utils.get_versions():
                    backends.append("ffmpeg")
            except Exception:
                pass
            try:
                import soundfile  # noqa: F401

                backends.append("soundfile")
            except Exception:
                pass
            return backends

        torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

    # --- torchaudio.backend.common module (removed in >= 2.9) ---
    if "torchaudio.backend.common" not in sys.modules:
        _mod = types.ModuleType("torchaudio.backend.common")
        _mod.AudioMetaData = torchaudio.AudioMetaData  # type: ignore[attr-defined]
        sys.modules["torchaudio.backend.common"] = _mod

    _pyannote_patched = True


def _patch_torch_load_for_pyannote() -> None:
    """Patch lightning's checkpoint loader to use ``weights_only=False``.

    PyTorch 2.6 changed ``torch.load`` default to ``weights_only=True``.
    pyannote / lightning checkpoints contain custom globals that are not
    on the safe list.  The simplest reliable fix is to patch
    ``lightning_fabric.utilities.cloud_io._load`` to pass
    ``weights_only=False``, matching the pre-2.6 behaviour.
    """
    try:
        import lightning_fabric.utilities.cloud_io as _cio

        if getattr(_cio, "_pyannote_patched", False):
            return

        _orig_load = _cio._load

        def _patched_load(*args: object, **kwargs: object) -> object:
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        _cio._load = _patched_load
        _cio._pyannote_patched = True
    except ImportError:
        pass


def _patch_huggingface_hub_for_pyannote() -> None:
    """Monkey-patch huggingface_hub so that ``use_auth_token`` is remapped
    to ``token`` when passed to ``hf_hub_download``, and inject ``HF_TOKEN``
    when no token is provided at all.

    pyannote.audio 4.x may not propagate the ``token`` kwarg to every
    internal ``from_pretrained`` call (e.g. ``PLDA.from_pretrained(**plda)``
    in ``get_plda`` drops the token).  This patch ensures the token is
    always present for gated repo downloads.
    """
    import huggingface_hub

    if getattr(huggingface_hub, "_pyannote_patched", False):
        return

    _orig = huggingface_hub.hf_hub_download

    def _wrapped(*args: object, **kwargs: object) -> object:
        # Remap legacy use_auth_token → token
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)

        # Inject HF_TOKEN from environment when no token is provided
        if "token" not in kwargs or kwargs["token"] is None:
            env_token = os.environ.get("HF_TOKEN")
            if env_token:
                kwargs["token"] = env_token

        return _orig(*args, **kwargs)

    huggingface_hub.hf_hub_download = _wrapped
    huggingface_hub._pyannote_patched = True  # type: ignore[attr-defined]


class Diarizer:
    """Speaker diarization using Pyannote Audio."""

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "pyannote/speaker-diarization-community-1",
        hf_token: str | None = None,
        num_speakers: int | None = None,
    ) -> None:
        self._device = device
        self._num_speakers = num_speakers
        self._pipeline = self._load_pipeline(model_name, hf_token)

    def _load_pipeline(self, model_name: str, hf_token: str | None):
        _patch_torchaudio_for_pyannote()
        _patch_huggingface_hub_for_pyannote()
        _patch_torch_load_for_pyannote()

        token = hf_token or os.environ.get("HF_TOKEN")

        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            model_name,
            token=token,
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

        diarization_output = self._pipeline(input_dict, **kwargs)

        # pyannote 4.0: prefer exclusive output (each time → exactly 1 speaker)
        if hasattr(diarization_output, "exclusive_speaker_diarization"):
            annotation = diarization_output.exclusive_speaker_diarization
        elif hasattr(diarization_output, "speaker_diarization"):
            annotation = diarization_output.speaker_diarization
        else:
            annotation = diarization_output  # pyannote 3.x

        # Extract overlap regions from the regular (non-exclusive) output
        overlap_regions = self._extract_overlap_regions(diarization_output, audio.start_time)

        # Build speaker segments from the exclusive annotation
        segments: list[SpeakerSegment] = []
        speaker_set: set[str] = set()

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_set.add(speaker)
            segments.append(
                SpeakerSegment(
                    speaker_id=speaker,
                    start_time=turn.start + audio.start_time,
                    end_time=turn.end + audio.start_time,
                )
            )

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speaker_set),
            overlap_regions=overlap_regions,
        )

    def _extract_overlap_regions(self, diarization_output, audio_start_time: float = 0.0) -> list[tuple[float, float]]:
        """Detect time regions where multiple speakers overlap."""
        # Get the regular (non-exclusive) annotation for overlap detection
        if hasattr(diarization_output, "speaker_diarization"):
            annotation = diarization_output.speaker_diarization
        else:
            annotation = diarization_output

        overlap_regions: list[tuple[float, float]] = []
        turns = list(annotation.itertracks(yield_label=True))
        for i, (turn_a, _, spk_a) in enumerate(turns):
            for turn_b, _, spk_b in turns[i + 1:]:
                if spk_a == spk_b:
                    continue
                ov_start = max(turn_a.start, turn_b.start)
                ov_end = min(turn_a.end, turn_b.end)
                if ov_start < ov_end:
                    overlap_regions.append((ov_start + audio_start_time, ov_end + audio_start_time))
        return overlap_regions

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._pipeline
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

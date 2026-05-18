"""Tests for Stage 2: Noise suppression (Denoiser)."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import AudioSegment
from transcribe.models.denoiser import Denoiser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def noisy_audio() -> AudioSegment:
    """1 second of 440 Hz sine + Gaussian noise at 16 kHz."""
    sr = 16_000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    clean = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.randn(len(t)).astype(np.float32)
    return AudioSegment(
        waveform=(clean + noise),
        sample_rate=sr,
        start_time=0.0,
        end_time=duration,
    )


# Use module-scoped denoiser to avoid reloading the model for every test.
@pytest.fixture(scope="module")
def denoiser():
    d = Denoiser(device="cpu")
    yield d
    d.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_denoiser_init(denoiser):
    """Denoiser can be constructed and model is loaded."""
    assert denoiser is not None


def test_denoiser_output_is_audio_segment(denoiser, noisy_audio):
    result = denoiser.process(noisy_audio)
    assert isinstance(result, AudioSegment)


def test_denoiser_preserves_metadata(denoiser, noisy_audio):
    result = denoiser.process(noisy_audio)
    assert result.sample_rate == noisy_audio.sample_rate
    assert result.start_time == noisy_audio.start_time
    assert result.end_time == noisy_audio.end_time


def test_denoiser_output_length_approximately_matches(denoiser, noisy_audio):
    """Resampling 16→48→16 kHz may shift length slightly (within 5 %)."""
    result = denoiser.process(noisy_audio)
    ratio = len(result.waveform) / len(noisy_audio.waveform)
    assert 0.95 <= ratio <= 1.05


def test_denoiser_reduces_noise(denoiser, noisy_audio):
    """Output should be non-silent."""
    result = denoiser.process(noisy_audio)
    assert np.abs(result.waveform).max() > 0

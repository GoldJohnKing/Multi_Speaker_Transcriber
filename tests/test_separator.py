"""Tests for Stage 4: Speech separation using SepFormer."""

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, DiarizationResult, SpeakerSegment
from transcribe.models.separator import Separator


@pytest.fixture
def overlap_audio():
    """2 seconds of synthetic mixed audio at 16kHz."""
    sr = 16000
    duration = 2.0
    waveform = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    return AudioSegment(
        waveform=waveform,
        sample_rate=sr,
        start_time=0.0,
        end_time=duration,
    )


@pytest.fixture
def diarization_with_overlap():
    """Diarization result with overlap regions."""
    return DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 0.0, 1.0),
            SpeakerSegment("SPEAKER_01", 0.5, 1.5, is_overlap=True),
            SpeakerSegment("SPEAKER_00", 1.5, 2.0),
        ],
        num_speakers=2,
        overlap_regions=[(0.5, 1.5)],
    )


def test_separator_init():
    sep = Separator(device="cpu")
    assert sep is not None


def test_separator_returns_list(overlap_audio, diarization_with_overlap):
    sep = Separator(device="cpu")
    separated = sep.separate_overlaps(overlap_audio, diarization_with_overlap)
    assert isinstance(separated, list)


def test_separator_output_are_audio_segments(overlap_audio, diarization_with_overlap):
    sep = Separator(device="cpu")
    separated = sep.separate_overlaps(overlap_audio, diarization_with_overlap)
    for seg in separated:
        assert isinstance(seg, AudioSegment)


def test_separator_no_overlap_returns_empty(overlap_audio):
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 0.0, 2.0)],
        num_speakers=1,
        overlap_regions=[],
    )
    sep = Separator(device="cpu")
    separated = sep.separate_overlaps(overlap_audio, diarization)
    assert separated == []


def test_separator_cleanup():
    sep = Separator(device="cpu")
    sep.cleanup()  # should not crash

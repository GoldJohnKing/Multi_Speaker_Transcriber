"""Tests for speaker diarization module."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from transcribe.data.types import AudioSegment, DiarizationResult
from transcribe.models.diarizer import Diarizer


def _make_audio(duration: float = 3.0, sr: int = 16000) -> AudioSegment:
    waveform = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    return AudioSegment(waveform=waveform, sample_rate=sr, start_time=0.0, end_time=duration)


class MockTurn:
    """Fake pyannote turn with start/end attributes."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


@pytest.fixture
def mock_pipeline():
    """Create a mock Pyannote pipeline that returns fake diarization output."""
    mock = MagicMock()

    # Simulate: SPEAKER_00 from 0.0-1.5, SPEAKER_01 from 1.0-2.5, SPEAKER_00 from 2.5-3.0
    tracks = [
        (MockTurn(0.0, 1.5), None, "SPEAKER_00"),
        (MockTurn(1.0, 2.5), None, "SPEAKER_01"),
        (MockTurn(2.5, 3.0), None, "SPEAKER_00"),
    ]
    # pyannote 4.0: pipeline returns DiarizeOutput with .exclusive_speaker_diarization
    # and .speaker_diarization
    mock.return_value.exclusive_speaker_diarization.itertracks.return_value = tracks
    mock.return_value.speaker_diarization.itertracks.return_value = tracks
    return mock


def test_diarizer_process_returns_result(mock_pipeline):
    """Diarizer.process should return DiarizationResult."""
    with patch("transcribe.models.diarizer.Diarizer._load_pipeline", return_value=mock_pipeline):
        diarizer = Diarizer(device="cpu")
        audio = _make_audio()
        result = diarizer.process(audio)

    assert isinstance(result, DiarizationResult)
    assert result.num_speakers == 2
    assert len(result.segments) == 3


def test_diarizer_segments_have_speaker_ids(mock_pipeline):
    """Each segment should have a valid speaker_id."""
    with patch("transcribe.models.diarizer.Diarizer._load_pipeline", return_value=mock_pipeline):
        diarizer = Diarizer(device="cpu")
        audio = _make_audio()
        result = diarizer.process(audio)

    for seg in result.segments:
        assert seg.speaker_id.startswith("SPEAKER_")
        assert seg.end_time > seg.start_time


def test_diarizer_cleanup():
    """Cleanup should not crash."""
    mock = MagicMock()
    with patch("transcribe.models.diarizer.Diarizer._load_pipeline", return_value=mock):
        diarizer = Diarizer(device="cpu")
        diarizer.cleanup()


def test_diarizer_num_speakers_passed_to_pipeline(mock_pipeline):
    """num_speakers should be forwarded to the pipeline."""
    with patch("transcribe.models.diarizer.Diarizer._load_pipeline", return_value=mock_pipeline):
        diarizer = Diarizer(device="cpu", num_speakers=2)
        audio = _make_audio()
        diarizer.process(audio)

    mock_pipeline.assert_called_once()
    call_kwargs = mock_pipeline.call_args
    assert call_kwargs[1].get("num_speakers") == 2


def test_diarizer_offset_audio(mock_pipeline):
    """Segment times should be offset by audio.start_time."""
    mock = MagicMock()
    tracks = [
        (MockTurn(0.0, 1.0), None, "SPEAKER_00"),
    ]
    mock.return_value.exclusive_speaker_diarization.itertracks.return_value = tracks
    mock.return_value.speaker_diarization.itertracks.return_value = tracks

    with patch("transcribe.models.diarizer.Diarizer._load_pipeline", return_value=mock):
        diarizer = Diarizer(device="cpu")
        audio = _make_audio()
        audio.start_time = 10.0  # offset
        result = diarizer.process(audio)

    assert result.segments[0].start_time == pytest.approx(10.0, abs=0.01)
    assert result.segments[0].end_time == pytest.approx(11.0, abs=0.01)


"""Tests for transcribe.models.audio_extractor."""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import pytest

# Ensure tests find the FFmpeg binary we installed to /tmp.
if os.path.isfile("/tmp/ffmpeg"):
    os.environ.setdefault("FFMPEG_PATH", "/tmp/ffmpeg")

from transcribe.data.types import AudioSegment
from transcribe.models.audio_extractor import AudioExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_test_wav(path: str, duration: float = 1.0, freq: int = 440) -> None:
    """Create a 16 kHz mono sine WAV at *path* using FFmpeg."""
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    subprocess.run(
        [
            ffmpeg,
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration}",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_wav(tmp_path):
    """Provide a short sine-wave WAV file for testing."""
    wav_path = str(tmp_path / "test.wav")
    _generate_test_wav(wav_path)
    return wav_path


@pytest.fixture()
def extractor():
    return AudioExtractor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAudioExtractor:
    def test_extract_returns_audio_segment(self, extractor, test_wav):
        result = extractor.extract(test_wav)
        assert isinstance(result, AudioSegment)

    def test_correct_sample_rate(self, extractor, test_wav):
        result = extractor.extract(test_wav)
        assert result.sample_rate == 16000

    def test_waveform_is_float32(self, extractor, test_wav):
        result = extractor.extract(test_wav)
        assert result.waveform.dtype == np.float32

    def test_start_time_is_zero(self, extractor, test_wav):
        result = extractor.extract(test_wav)
        assert result.start_time == 0.0

    def test_end_time_approx_duration(self, extractor, test_wav):
        result = extractor.extract(test_wav)
        # 1-second sine; allow small tolerance for encoding latency
        assert abs(result.end_time - 1.0) < 0.05

    def test_duration_property(self, extractor, test_wav):
        result = extractor.extract(test_wav)
        assert abs(result.duration - 1.0) < 0.05

    def test_file_not_found_raises(self, extractor, tmp_path):
        missing = str(tmp_path / "nonexistent.wav")
        with pytest.raises(FileNotFoundError):
            extractor.extract(missing)

    def test_corrupted_file_raises_runtime_error(self, extractor, tmp_path):
        bad_path = str(tmp_path / "corrupt.wav")
        with open(bad_path, "wb") as f:
            f.write(b"NOT_A_REAL_WAV_FILE_CONTENTS_GARBAGE")
        with pytest.raises(RuntimeError):
            extractor.extract(bad_path)

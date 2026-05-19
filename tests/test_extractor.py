"""Tests for Stage 4A: Target Speaker Extraction (ClearVoice TSE)."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, DiarizationResult, SpeakerSegment


@pytest.fixture
def sample_audio_16k():
    """1 second of synthetic audio at 16kHz."""
    sr = 16_000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    waveform = 0.3 * np.sin(2 * np.pi * 440 * t)
    return AudioSegment(waveform=waveform, sample_rate=sr, start_time=0.0, end_time=duration)


@pytest.fixture
def sample_diarization():
    return DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 0.0, 0.5),
            SpeakerSegment("SPEAKER_01", 0.3, 1.0, is_overlap=True),
        ],
        num_speakers=2,
        overlap_regions=[(0.3, 0.5)],
    )


def test_extractor_init():
    """Extractor can be constructed without a video file."""
    from transcribe.models.extractor import TargetSpeakerExtractor

    extractor = TargetSpeakerExtractor(device="cpu")
    assert extractor is not None
    extractor.cleanup()


def test_extractor_cleanup():
    from transcribe.models.extractor import TargetSpeakerExtractor

    extractor = TargetSpeakerExtractor(device="cpu")
    extractor.cleanup()  # should not crash


@pytest.mark.slow
def test_extractor_requires_video(sample_audio_16k, sample_diarization, tmp_path):
    """TSE should raise an error when given an audio file instead of video."""
    # Create a dummy WAV file
    wav_path = tmp_path / "test.wav"
    import soundfile as sf

    sf.write(str(wav_path), sample_audio_16k.waveform, 16000)

    from transcribe.pipeline import run_pipeline
    from transcribe.data.types import PipelineConfig

    # This should fail because .wav is not a video file
    with pytest.raises(RuntimeError, match="需要视频文件输入"):
        run_pipeline(
            input_path=str(wav_path),
            output_path=str(tmp_path / "output.srt"),
            config=PipelineConfig(device="cpu", tse=True, diarize=True),
        )

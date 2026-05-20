"""Full pipeline integration tests (all stages).

Requires all models to be cached locally.
Marked as slow / optional for CI.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def sample_wav(tmp_path: Path):
    """Generate a 2-second sine wave WAV file at 16kHz."""
    wav_path = tmp_path / "test.wav"
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=2",
            "-ar", "16000", "-ac", "1",
            "-y", str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


@pytest.mark.slow
def test_full_pipeline_with_all_stages(sample_wav, tmp_path):
    """Test full pipeline without denoising."""
    from transcribe.data.types import PipelineConfig
    from transcribe.pipeline import run_pipeline

    output = str(tmp_path / "output.srt")
    config = PipelineConfig(device="cpu")

    result = run_pipeline(
        input_path=str(sample_wav),
        output_path=output,
        config=config,
    )
    assert Path(result).exists()


@pytest.mark.slow
def test_full_pipeline_no_diarize(sample_wav, tmp_path):
    """Test pipeline with no diarization (--no-diarize)."""
    from transcribe.data.types import PipelineConfig
    from transcribe.pipeline import run_pipeline

    output = str(tmp_path / "output.srt")
    config = PipelineConfig(device="cpu", diarize=False)

    result = run_pipeline(
        input_path=str(sample_wav),
        output_path=output,
        config=config,
    )
    assert Path(result).exists()

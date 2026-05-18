"""Integration tests for the basic ASR pipeline (Phase 1).

These tests require FunASR models to be cached locally.
Mark as slow / optional for CI.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def sample_wav(tmp_path: Path):
    """Create a short WAV file via FFmpeg."""
    wav_path = tmp_path / "test.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


@pytest.mark.slow
def test_pipeline_produces_srt(sample_wav, tmp_path):
    """Full pipeline should produce an SRT file."""
    from transcribe.config import PipelineConfig
    from transcribe.pipeline import run_pipeline

    output = str(tmp_path / "output.srt")
    config = PipelineConfig(device="cpu", denoise=False, hotwords=None)

    result = run_pipeline(
        input_path=str(sample_wav),
        output_path=output,
        config=config,
    )
    assert Path(result).exists()
    content = Path(result).read_text(encoding="utf-8")
    # Sine wave has no speech, so output should be empty or very minimal
    assert isinstance(content, str)


@pytest.mark.slow
def test_cli_help():
    """CLI --help should not crash."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "transcribe", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "input" in result.stdout


def test_cli_parse_basic_args():
    """CLI should parse basic arguments."""
    from transcribe.cli import parse_args

    args = parse_args(["input.mp4", "-o", "output.srt", "--denoise", "-v"])
    assert args.input == "input.mp4"
    assert args.output == "output.srt"
    assert args.denoise is True
    assert args.verbose is True

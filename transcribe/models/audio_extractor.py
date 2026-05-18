"""Stage 1: Audio extraction — convert any video/audio to 16 kHz mono WAV."""

from __future__ import annotations

import io
import os
import subprocess

import numpy as np
import soundfile as sf

from transcribe.data.types import AudioSegment

# Default FFmpeg path; can be overridden for testing or unusual setups.
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")

TARGET_SAMPLE_RATE = 16_000


class AudioExtractor:
    """Extract audio from video/audio files via FFmpeg and return an AudioSegment."""

    def extract(self, input_path: str) -> AudioSegment:
        """Convert *input_path* to a 16 kHz mono float32 AudioSegment.

        Parameters
        ----------
        input_path:
            Path to a video or audio file understood by FFmpeg.

        Returns
        -------
        AudioSegment
            Mono float32 waveform at 16 kHz with timing metadata.

        Raises
        ------
        FileNotFoundError
            If *input_path* does not exist on disk.
        RuntimeError
            If the FFmpeg subprocess exits with a non-zero code.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        cmd = [
            FFMPEG_PATH,
            "-i", input_path,
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", "1",            # mono
            "-f", "wav",
            "pipe:1",              # write WAV to stdout
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"FFmpeg failed (exit code {result.returncode}): {stderr}"
            )

        waveform, sr = sf.read(io.BytesIO(result.stdout), dtype="float32")

        # soundfile returns shape (num_frames,) for mono — ensure 1-D.
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1).astype(np.float32)

        duration = len(waveform) / sr
        return AudioSegment(
            waveform=waveform,
            sample_rate=sr,
            start_time=0.0,
            end_time=duration,
        )

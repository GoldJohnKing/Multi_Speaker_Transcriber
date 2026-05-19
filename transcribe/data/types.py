"""Core data types for the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioSegment:
    """A segment of audio data."""

    waveform: np.ndarray  # float32, shape (samples,) for mono
    sample_rate: int
    start_time: float  # seconds, relative to original audio
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SpeakerSegment:
    """A speaker time segment."""

    speaker_id: str  # "SPEAKER_00", "SPEAKER_01", ...
    start_time: float
    end_time: float
    is_overlap: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DiarizationResult:
    """Speaker diarization result."""

    segments: list[SpeakerSegment]
    num_speakers: int
    overlap_regions: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class TranscriptSegment:
    """A transcribed segment."""

    speaker_id: str
    start_time: float
    end_time: float
    text: str  # with punctuation


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    device: str = "auto"  # "cpu" | "cuda" | "auto"
    denoise: bool = False  # enable noise suppression (ClearVoice SE)
    diarize: bool = True  # enable speaker diarization (Pyannote)
    separate: bool = False  # enable overlap speech separation (ClearVoice SS)
    tse: bool = False  # enable target speaker extraction (ClearVoice TSE)
    hotwords: str | None = None  # hotword file path
    language: str = "zh"
    cache_dir: str = ".cache"
    num_speakers: int | None = None  # known speaker count, auto-detect if None
    speaker_references: str | None = None  # directory of speaker reference audio samples

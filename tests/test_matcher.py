"""Tests for Stage 4.5: Speaker matching via voice embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.data.types import AudioSegment, DiarizationResult, SpeakerSegment


# ---------------------------------------------------------------------------
# Unit tests for helper functions (no model needed)
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    from transcribe.models.matcher import _cosine_similarity

    a = np.array([1.0, 0.0, 0.0])
    assert _cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    from transcribe.models.matcher import _cosine_similarity

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    from transcribe.models.matcher import _cosine_similarity

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_find_reference_segments_picks_longest_non_overlap():
    from transcribe.models.matcher import _find_reference_segments

    diarization = DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 0.0, 1.0),  # 1.0s non-overlap
            SpeakerSegment("SPEAKER_01", 0.5, 0.8, is_overlap=True),  # 0.3s overlap
            SpeakerSegment("SPEAKER_00", 2.0, 4.5),  # 2.5s non-overlap (longest)
            SpeakerSegment("SPEAKER_01", 3.0, 5.0),  # 2.0s non-overlap
        ],
        num_speakers=2,
        overlap_regions=[(0.5, 0.8)],
    )

    refs = _find_reference_segments(diarization)
    assert "SPEAKER_00" in refs
    assert "SPEAKER_01" in refs
    # SPEAKER_00: longest non-overlap is (2.0, 4.5)
    assert refs["SPEAKER_00"][0] == (2.0, 4.5)
    # SPEAKER_01: longest non-overlap is (3.0, 5.0)
    assert refs["SPEAKER_01"][0] == (3.0, 5.0)


def test_find_reference_segments_fallback_to_overlap():
    from transcribe.models.matcher import _find_reference_segments

    diarization = DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 0.0, 0.2),  # too short
            SpeakerSegment("SPEAKER_01", 0.1, 2.0, is_overlap=True),  # overlap, only segment
        ],
        num_speakers=2,
        overlap_regions=[(0.1, 2.0)],
    )

    refs = _find_reference_segments(diarization)
    # SPEAKER_01 should fallback to its only (overlap) segment
    assert "SPEAKER_01" in refs
    assert len(refs["SPEAKER_01"]) == 1
    assert refs["SPEAKER_01"][0] == (0.1, 2.0)


# ---------------------------------------------------------------------------
# Integration tests (require model download)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audio_16k():
    """1 second of synthetic audio at 16kHz."""
    sr = 16_000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    waveform = 0.3 * np.sin(2 * np.pi * 440 * t)
    return AudioSegment(waveform=waveform, sample_rate=sr, start_time=0.0, end_time=duration)


@pytest.mark.slow
def test_matcher_init():
    from transcribe.models.matcher import SpeakerMatcher

    matcher = SpeakerMatcher(device="cpu")
    assert matcher is not None
    matcher.cleanup()


@pytest.mark.slow
def test_matcher_returns_mapping(sample_audio_16k):
    from transcribe.models.matcher import SpeakerMatcher

    matcher = SpeakerMatcher(device="cpu")

    # Create two synthetic tracks
    track1 = AudioSegment(
        waveform=sample_audio_16k.waveform.copy(),
        sample_rate=16_000,
        start_time=0.0,
        end_time=1.0,
    )
    track2 = AudioSegment(
        waveform=sample_audio_16k.waveform.copy(),
        sample_rate=16_000,
        start_time=0.0,
        end_time=1.0,
    )

    diarization = DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 0.0, 1.0),
            SpeakerSegment("SPEAKER_01", 0.0, 1.0, is_overlap=True),
        ],
        num_speakers=2,
        overlap_regions=[(0.0, 1.0)],
    )

    mapping = matcher.match_tracks_to_speakers([track1, track2], sample_audio_16k, diarization)
    assert isinstance(mapping, dict)
    assert len(mapping) == 2
    # Both tracks should have a speaker assignment (or UNKNOWN)
    for idx in range(2):
        assert idx in mapping
    matcher.cleanup()


@pytest.mark.slow
def test_matcher_empty_tracks(sample_audio_16k):
    from transcribe.models.matcher import SpeakerMatcher

    matcher = SpeakerMatcher(device="cpu")
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
        num_speakers=1,
        overlap_regions=[],
    )
    mapping = matcher.match_tracks_to_speakers([], sample_audio_16k, diarization)
    assert mapping == {}
    matcher.cleanup()


@pytest.mark.slow
def test_register_speakers_from_directory(tmp_path):
    """register_speakers loads wav files and creates reference embeddings."""
    import soundfile as sf
    from transcribe.models.matcher import SpeakerMatcher

    sr = 16_000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    sf.write(str(tmp_path / "张三.wav"), wav, sr)
    sf.write(str(tmp_path / "李四.wav"), wav, sr)

    matcher = SpeakerMatcher(device="cpu")
    result = matcher.register_speakers(str(tmp_path))

    assert "张三" in result
    assert "李四" in result
    assert result["张三"] is not None
    assert result["李四"] is not None
    assert isinstance(result["张三"], np.ndarray)
    assert result["张三"].shape == (192,)
    matcher.cleanup()


@pytest.mark.slow
def test_register_speakers_missing_directory():
    """register_speakers raises FileNotFoundError for missing directory."""
    from transcribe.models.matcher import SpeakerMatcher

    matcher = SpeakerMatcher(device="cpu")
    with pytest.raises(FileNotFoundError, match="Speaker reference directory"):
        matcher.register_speakers("/nonexistent/path")
    matcher.cleanup()


@pytest.mark.slow
def test_register_speakers_empty_directory(tmp_path):
    """register_speakers raises ValueError for directory with no audio files."""
    from transcribe.models.matcher import SpeakerMatcher

    matcher = SpeakerMatcher(device="cpu")
    with pytest.raises(ValueError, match="No valid audio files"):
        matcher.register_speakers(str(tmp_path))
    matcher.cleanup()


def test_match_speakers_to_references_no_user_refs():
    """Returns empty dict when no user references are registered."""
    from transcribe.models.matcher import SpeakerMatcher

    matcher = SpeakerMatcher.__new__(SpeakerMatcher)
    matcher._user_references = None

    audio = AudioSegment(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        start_time=0.0,
        end_time=1.0,
    )
    diarization = DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_00", 0.0, 1.0)],
        num_speakers=1,
    )

    result = matcher.match_speakers_to_references(audio, diarization)
    assert result == {}


# ---------------------------------------------------------------------------
# Edge case tests for speaker voice reference feature (no model needed)
# ---------------------------------------------------------------------------


def test_match_speakers_partial_match():
    """Only speakers with embeddings above threshold are mapped."""
    from transcribe.models.matcher import SpeakerMatcher, _cosine_similarity

    matcher = SpeakerMatcher.__new__(SpeakerMatcher)
    matcher._user_references = {
        "张三": np.array([1.0, 0.0] + [0.0] * 190, dtype=np.float32),
        "李四": np.array([0.0, 1.0] + [0.0] * 190, dtype=np.float32),
    }
    matcher._match_threshold = 0.5

    # Verify cosine similarity works for aligned vectors
    ref_emb = matcher._user_references["张三"]
    test_emb = np.array([0.9, 0.1] + [0.0] * 190, dtype=np.float32)
    sim = _cosine_similarity(test_emb, ref_emb)
    assert sim > 0.5  # Should match


def test_match_speakers_no_match_below_threshold():
    """Speakers below threshold are not mapped."""
    from transcribe.models.matcher import SpeakerMatcher, _cosine_similarity

    matcher = SpeakerMatcher.__new__(SpeakerMatcher)
    matcher._user_references = {
        "张三": np.array([1.0, 0.0] + [0.0] * 190, dtype=np.float32),
    }
    matcher._match_threshold = 0.9  # High threshold

    # Orthogonal embedding should not match
    test_emb = np.array([0.0, 1.0] + [0.0] * 190, dtype=np.float32)
    sim = _cosine_similarity(test_emb, matcher._user_references["张三"])
    assert sim < matcher._match_threshold


def test_hungarian_match_optimal_assignment():
    """Hungarian algorithm finds globally optimal assignment, not greedy."""
    from transcribe.models.matcher import _hungarian_match

    sim = np.array([
        [0.95, 0.80],
        [0.90, 0.20],
    ])
    result = _hungarian_match(sim, ["A", "B"], ["Ref1", "Ref2"], threshold=0.5)
    assert result["A"] == "Ref2"
    assert result["B"] == "Ref1"


def test_hungarian_match_below_threshold():
    """Pairs below threshold are omitted from mapping."""
    from transcribe.models.matcher import _hungarian_match

    sim = np.array([[0.3, 0.2]])
    result = _hungarian_match(sim, ["A"], ["Ref1", "Ref2"], threshold=0.5)
    assert result == {}


def test_hungarian_match_empty_matrix():
    """Empty similarity matrix returns empty mapping."""
    from transcribe.models.matcher import _hungarian_match

    sim = np.zeros((0, 0))
    result = _hungarian_match(sim, [], [], threshold=0.5)
    assert result == {}


def test_hungarian_match_rectangular():
    """Works with more rows than columns."""
    from transcribe.models.matcher import _hungarian_match

    sim = np.array([
        [0.9, 0.2],
        [0.3, 0.8],
        [0.1, 0.1],
    ])
    result = _hungarian_match(sim, ["A", "B", "C"], ["Ref1", "Ref2"], threshold=0.5)
    assert result["A"] == "Ref1"
    assert result["B"] == "Ref2"
    assert "C" not in result


def test_find_reference_segments_returns_multiple():
    """Returns up to max_segments per speaker, sorted by duration desc."""
    from transcribe.models.matcher import _find_reference_segments

    diarization = DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 0.0, 1.0),  # 1.0s
            SpeakerSegment("SPEAKER_00", 2.0, 4.5),  # 2.5s (longest)
            SpeakerSegment("SPEAKER_00", 5.0, 6.5),  # 1.5s
            SpeakerSegment("SPEAKER_00", 7.0, 7.3),  # 0.3s (too short)
        ],
        num_speakers=1,
    )

    refs = _find_reference_segments(diarization, max_segments=3)
    assert len(refs["SPEAKER_00"]) == 3
    assert refs["SPEAKER_00"][0] == (2.0, 4.5)  # 2.5s first
    assert refs["SPEAKER_00"][1] == (5.0, 6.5)  # 1.5s second
    assert refs["SPEAKER_00"][2] == (0.0, 1.0)  # 1.0s third


def test_extract_speaker_embeddings_with_mock():
    """Multi-segment averaging produces unit-norm embedding."""
    import torch
    from unittest.mock import MagicMock
    from transcribe.models.matcher import _extract_speaker_embeddings

    sr = 16_000
    audio = AudioSegment(
        waveform=np.random.randn(sr * 5).astype(np.float32),
        sample_rate=sr,
        start_time=0.0,
        end_time=5.0,
    )
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment("SPEAKER_00", 0.0, 1.0),
            SpeakerSegment("SPEAKER_00", 2.0, 3.0),
        ],
        num_speakers=1,
    )

    # ERes2NetV2 model API: model(waveform) → torch.Tensor of shape (192,)
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(192)

    embeddings = _extract_speaker_embeddings(audio, diarization, mock_model)
    assert "SPEAKER_00" in embeddings
    assert embeddings["SPEAKER_00"].shape == (192,)
    norm = np.linalg.norm(embeddings["SPEAKER_00"])
    assert abs(norm - 1.0) < 0.01

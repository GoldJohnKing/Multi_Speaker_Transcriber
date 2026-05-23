"""Tests for overlap separation module."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from transcribe.data.types import AudioSegment, WordTimestamp
from transcribe.models.separator import (
    OverlapClip,
    extract_overlap_clips,
    _trim_words_to_overlaps,
    _map_local_to_global_speakers,
)


def _make_audio(duration: float = 30.0, sr: int = 16000) -> AudioSegment:
    waveform = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    return AudioSegment(waveform=waveform, sample_rate=sr, start_time=0.0, end_time=duration)


class TestExtractOverlapClips:
    def test_single_overlap_region(self):
        audio = _make_audio(30.0)
        clips = extract_overlap_clips(audio, overlap_regions=[(10.0, 12.0)], padding=3.0)
        assert len(clips) == 1
        assert clips[0].start_time == pytest.approx(7.0)
        assert clips[0].end_time == pytest.approx(15.0)
        assert clips[0].source_overlaps == [(10.0, 12.0)]

    def test_two_adjacent_overlaps_merge(self):
        audio = _make_audio(30.0)
        clips = extract_overlap_clips(audio, overlap_regions=[(5.0, 8.0), (10.0, 13.0)], padding=3.0)
        assert len(clips) == 1
        assert clips[0].start_time == pytest.approx(2.0)
        assert clips[0].end_time == pytest.approx(16.0)
        assert clips[0].source_overlaps == [(5.0, 8.0), (10.0, 13.0)]

    def test_two_distant_overlaps_stay_separate(self):
        audio = _make_audio(60.0)
        clips = extract_overlap_clips(audio, overlap_regions=[(5.0, 8.0), (40.0, 43.0)], padding=3.0)
        assert len(clips) == 2
        assert clips[0].start_time == pytest.approx(2.0)
        assert clips[0].end_time == pytest.approx(11.0)
        assert clips[1].start_time == pytest.approx(37.0)
        assert clips[1].end_time == pytest.approx(46.0)

    def test_padding_clamped_to_audio_start(self):
        audio = _make_audio(10.0)
        clips = extract_overlap_clips(audio, overlap_regions=[(2.0, 4.0)], padding=3.0)
        assert clips[0].start_time == pytest.approx(0.0)
        assert clips[0].end_time == pytest.approx(7.0)

    def test_padding_clamped_to_audio_end(self):
        audio = _make_audio(10.0)
        clips = extract_overlap_clips(audio, overlap_regions=[(8.0, 9.5)], padding=3.0)
        assert clips[0].start_time == pytest.approx(5.0)
        assert clips[0].end_time == pytest.approx(10.0)

    def test_no_overlaps_returns_empty(self):
        audio = _make_audio()
        clips = extract_overlap_clips(audio, overlap_regions=[], padding=3.0)
        assert clips == []

    def test_source_overlaps_preserved(self):
        audio = _make_audio(30.0)
        clips = extract_overlap_clips(audio, overlap_regions=[(5.0, 8.0), (9.0, 11.0)], padding=3.0)
        assert len(clips) == 1
        assert clips[0].source_overlaps == [(5.0, 8.0), (9.0, 11.0)]

    def test_audio_start_offset_respected(self):
        waveform = np.random.randn(16000 * 30).astype(np.float32) * 0.1
        audio = AudioSegment(waveform=waveform, sample_rate=16000, start_time=60.0, end_time=90.0)
        clips = extract_overlap_clips(audio, overlap_regions=[(65.0, 67.0)], padding=3.0)
        assert clips[0].start_time == pytest.approx(62.0)
        assert clips[0].end_time == pytest.approx(70.0)

    def test_clip_waveform_is_slice_of_original(self):
        audio = _make_audio(30.0)
        clips = extract_overlap_clips(audio, overlap_regions=[(10.0, 12.0)], padding=3.0)
        clip = clips[0]
        start_sample = int((clip.start_time - audio.start_time) * audio.sample_rate)
        end_sample = int((clip.end_time - audio.start_time) * audio.sample_rate)
        expected = audio.waveform[start_sample:end_sample]
        np.testing.assert_array_equal(clip.waveform, expected)


class TestTrimToOverlapRegions:
    def test_trim_keeps_words_in_overlap(self):
        words = [
            WordTimestamp(word="你", start_time=2.0, end_time=2.5),
            WordTimestamp(word="好", start_time=2.5, end_time=3.0),
            WordTimestamp(word="吗", start_time=5.0, end_time=5.5),
            WordTimestamp(word="我", start_time=8.0, end_time=8.5),
            WordTimestamp(word="是", start_time=8.5, end_time=9.0),
        ]
        overlaps = [(5.0, 10.0)]
        result = _trim_words_to_overlaps(words, overlaps)
        assert len(result) == 3
        assert result[0].word == "吗"
        assert result[1].word == "我"
        assert result[2].word == "是"

    def test_trim_multiple_overlap_regions(self):
        words = [
            WordTimestamp(word="A", start_time=1.0, end_time=2.0),
            WordTimestamp(word="B", start_time=5.0, end_time=6.0),
            WordTimestamp(word="C", start_time=8.0, end_time=9.0),
            WordTimestamp(word="D", start_time=12.0, end_time=13.0),
        ]
        overlaps = [(4.0, 7.0), (11.0, 14.0)]
        result = _trim_words_to_overlaps(words, overlaps)
        assert len(result) == 2
        assert result[0].word == "B"
        assert result[1].word == "D"

    def test_word_straddling_boundary_kept(self):
        words = [
            WordTimestamp(word="X", start_time=3.0, end_time=5.5),
        ]
        overlaps = [(5.0, 10.0)]
        result = _trim_words_to_overlaps(words, overlaps)
        assert len(result) == 1
        assert result[0].word == "X"


class TestMapLocalToGlobalSpeakers:
    def test_identity_mapping(self):
        local_labels = ["SPEAKER_00", "SPEAKER_01"]
        global_speakers = ["SPEAKER_00", "SPEAKER_01"]
        mapping = _map_local_to_global_speakers(
            local_labels=local_labels,
            local_times=[(0.0, 10.0), (0.0, 10.0)],
            global_speakers=global_speakers,
            global_segs=[
                MagicMock(speaker_id="SPEAKER_00", start_time=0.0, end_time=5.0),
                MagicMock(speaker_id="SPEAKER_01", start_time=5.0, end_time=10.0),
            ],
        )
        assert mapping["SPEAKER_00"] == "SPEAKER_00"

    def test_different_label_ordering(self):
        local_labels = ["A", "B"]
        global_speakers = ["SPEAKER_00", "SPEAKER_01"]
        mapping = _map_local_to_global_speakers(
            local_labels=local_labels,
            local_times=[(0.0, 5.0), (5.0, 10.0)],
            global_speakers=global_speakers,
            global_segs=[
                MagicMock(speaker_id="SPEAKER_00", start_time=0.0, end_time=6.0),
                MagicMock(speaker_id="SPEAKER_01", start_time=4.0, end_time=10.0),
            ],
        )
        assert mapping["A"] == "SPEAKER_00"
        assert mapping["B"] == "SPEAKER_01"

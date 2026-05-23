"""Tests for _replace_overlap_segments in pipeline.py."""

from dataclasses import replace

from transcribe.data.types import TranscriptSegment
from transcribe.pipeline import _replace_overlap_segments


def _seg(speaker: str, start: float, end: float, text: str = "", is_overlap: bool = False) -> TranscriptSegment:
    return TranscriptSegment(
        speaker_id=speaker, start_time=start, end_time=end, text=text, is_overlap=is_overlap,
    )


class TestReplaceOverlapSegments:
    def test_no_overlap_regions(self):
        """When no overlap regions, all main segments are kept."""
        main = [_seg("A", 0.0, 5.0), _seg("B", 5.0, 10.0)]
        result = _replace_overlap_segments(main, [], [])
        assert result == main

    def test_no_separated_segments(self):
        """Overlap segments removed, nothing inserted back."""
        main = [_seg("A", 0.0, 5.0), _seg("B", 5.0, 10.0)]
        regions = [(3.0, 7.0)]
        result = _replace_overlap_segments(main, [], regions)
        assert len(result) == 0  # both segments intersect [3,7]

    def test_replaced_with_separated(self):
        """Main segments in overlap are removed, separated ones inserted."""
        main = [_seg("A", 0.0, 5.0), _seg("B", 5.0, 10.0)]
        overlap = [_seg("A", 3.0, 5.0, is_overlap=True), _seg("B", 5.0, 7.0, is_overlap=True)]
        regions = [(3.0, 7.0)]
        result = _replace_overlap_segments(main, overlap, regions)
        assert len(result) == 2
        assert result[0].is_overlap is True
        assert result[1].is_overlap is True

    def test_segment_straddling_boundary_is_removed(self):
        """A long segment that straddles an overlap boundary (midpoint outside) 
        is still removed because it has temporal intersection."""
        # Segment [0, 10] with overlap region [8, 12]
        # Midpoint = 5.0 — old code would keep this (5.0 not in [8,12])
        # New code: 0 < 12 and 10 > 8 → intersection → removed
        main = [_seg("A", 0.0, 10.0, text="long segment")]
        overlap = [_seg("A", 8.0, 10.0, text="sep", is_overlap=True)]
        regions = [(8.0, 12.0)]
        result = _replace_overlap_segments(main, overlap, regions)
        assert len(result) == 1
        assert result[0].is_overlap is True
        assert result[0].text == "sep"

    def test_segment_entirely_outside_overlap_kept(self):
        """Segments with no intersection to any overlap region are kept."""
        main = [_seg("A", 0.0, 3.0), _seg("B", 10.0, 15.0)]
        regions = [(5.0, 8.0)]
        result = _replace_overlap_segments(main, [], regions)
        assert len(result) == 2

    def test_multiple_overlap_regions(self):
        """Multiple non-adjacent overlap regions handled correctly."""
        main = [
            _seg("A", 0.0, 5.0),
            _seg("B", 5.0, 10.0),
            _seg("A", 10.0, 15.0),
            _seg("B", 15.0, 20.0),
        ]
        regions = [(3.0, 7.0), (13.0, 17.0)]
        result = _replace_overlap_segments(main, [], regions)
        # Segments [0,5] intersects [3,7] → removed
        # [5,10] intersects [3,7] → removed
        # [10,15] intersects [13,17] → removed
        # [15,20] intersects [13,17] → removed
        assert len(result) == 0

    def test_result_sorted_by_start_time(self):
        """Merged result is sorted by start time."""
        main = [_seg("A", 0.0, 3.0), _seg("B", 8.0, 12.0)]
        overlap = [_seg("B", 6.0, 7.0, is_overlap=True), _seg("A", 4.0, 5.5, is_overlap=True)]
        regions = [(4.0, 7.0)]
        result = _replace_overlap_segments(main, overlap, regions)
        # [0,3] kept (no intersection with [4,7])
        # [8,12] kept (no intersection with [4,7])
        # Plus the overlap segments
        assert len(result) == 4
        times = [(s.start_time, s.end_time) for s in result]
        assert times == sorted(times)

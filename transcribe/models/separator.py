"""Overlap speech separation — PixIT-based separation for overlap regions."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from transcribe.data.types import AudioSegment, DiarizationResult, TranscriptSegment, WordTimestamp
from transcribe.models.segmentation import SubtitleSegmenter

_logger = logging.getLogger(__name__)


@dataclass
class OverlapClip:
    """A padded, merged audio clip containing one or more overlap regions."""

    waveform: np.ndarray
    sample_rate: int
    start_time: float  # absolute time in original audio
    end_time: float
    source_overlaps: list[tuple[float, float]]  # original (unpadded) overlap regions


def extract_overlap_clips(
    audio: AudioSegment,
    overlap_regions: list[tuple[float, float]],
    padding: float = 3.0,
) -> list[OverlapClip]:
    """Extract padded, merged audio clips for overlap regions.

    For each overlap region, creates a clip extended by *padding* seconds on
    each side. Adjacent or overlapping padded clips are merged into a single
    clip. Each clip tracks which original overlap regions it covers.

    Padding is clamped to audio boundary. Returns empty list if no overlaps.
    """
    if not overlap_regions:
        return []

    # Filter out zero-duration overlaps
    filtered = [(s, e) for s, e in overlap_regions if e > s]
    if not filtered:
        return []

    audio_start = audio.start_time
    audio_end = audio.end_time

    # Step 1: Pad each overlap region
    padded: list[tuple[float, float, tuple[float, float]]] = []
    for ov_start, ov_end in filtered:
        padded_start = max(ov_start - padding, audio_start)
        padded_end = min(ov_end + padding, audio_end)
        padded.append((padded_start, padded_end, (ov_start, ov_end)))

    # Sort by start time
    padded.sort(key=lambda x: x[0])

    # Step 2: Merge overlapping/adjacent padded regions
    merged: list[tuple[float, float, list[tuple[float, float]]]] = []
    cur_start, cur_end, cur_sources = padded[0][0], padded[0][1], [padded[0][2]]

    for p_start, p_end, source in padded[1:]:
        if p_start <= cur_end:  # overlapping or adjacent → merge
            cur_end = max(cur_end, p_end)
            cur_sources.append(source)
        else:
            merged.append((cur_start, cur_end, cur_sources))
            cur_start, cur_end, cur_sources = p_start, p_end, [source]
    merged.append((cur_start, cur_end, cur_sources))

    # Step 3: Extract waveform for each merged clip
    clips: list[OverlapClip] = []
    for clip_start, clip_end, sources in merged:
        start_sample = int((clip_start - audio_start) * audio.sample_rate)
        end_sample = int((clip_end - audio_start) * audio.sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio.waveform), end_sample)

        clips.append(OverlapClip(
            waveform=audio.waveform[start_sample:end_sample],
            sample_rate=audio.sample_rate,
            start_time=clip_start,
            end_time=clip_end,
            source_overlaps=sources,
        ))

    return clips


def _trim_words_to_overlaps(
    words: list[WordTimestamp],
    overlap_regions: list[tuple[float, float]],
) -> list[WordTimestamp]:
    """Keep only words that intersect with any overlap region.

    A word is kept if any part of its time range falls within an overlap
    region. This is intentionally lenient — it's better to keep a boundary
    word than to lose it.
    """
    if not words or not overlap_regions:
        return []

    result: list[WordTimestamp] = []
    for w in words:
        for ov_start, ov_end in overlap_regions:
            if w.start_time < ov_end and w.end_time > ov_start:
                result.append(w)
                break
    return result


def _map_local_to_global_speakers(
    local_labels: list[str],
    local_times: list[tuple[float, float]],
    global_speakers: list[str],
    global_segs: list,
) -> dict[str, str]:
    """Map PixIT's local speaker labels to global diarization speaker IDs.

    Uses bipartite matching (Hungarian algorithm) to find the optimal 1:1
    assignment that maximizes total temporal overlap between local and global
    speakers. Falls back to greedy matching if scipy is unavailable.

    Args:
        local_labels: Speaker labels from PixIT output (e.g., ["A", "B"]).
        local_times: (start, end) for each local speaker's active time.
        global_speakers: Global speaker IDs from diarization.
        global_segs: Global SpeakerSegment-like objects with speaker_id, start_time, end_time.

    Returns:
        Mapping from local label → global speaker ID.

    Note:
        When ``n_local > n_global``, only ``n_global`` pairs are returned.
        Unmapped local labels are handled by the caller via ``dict.get(label, label)``
        which falls back to the local label string.
    """
    if not local_labels or not global_speakers:
        return {}

    n_local = len(local_labels)
    n_global = len(global_speakers)

    # Build overlap matrix: cost[i][j] = negative overlap (Hungarian minimizes)
    overlap_matrix = [[0.0] * n_global for _ in range(n_local)]

    for i, (local_start, local_end) in enumerate(local_times):
        for j, g_speaker in enumerate(global_speakers):
            total_ov = 0.0
            for seg in global_segs:
                if seg.speaker_id == g_speaker:
                    ov = max(0.0, min(local_end, seg.end_time) - max(local_start, seg.start_time))
                    total_ov += ov
            overlap_matrix[i][j] = total_ov

    # Try Hungarian algorithm for optimal 1:1 matching
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        cost = np.array(overlap_matrix)
        # Negate because linear_sum_assignment minimizes cost
        row_ind, col_ind = linear_sum_assignment(-cost)

        mapping: dict[str, str] = {}
        for r, c in zip(row_ind, col_ind):
            if overlap_matrix[r][c] > 0:
                mapping[local_labels[r]] = global_speakers[c]
            else:
                # No overlap at all — keep original label
                mapping[local_labels[r]] = local_labels[r]
        return mapping

    except ImportError:
        # Fallback: greedy with dedup
        _logger.warning("scipy not available, using greedy speaker mapping")
        used_global: set[str] = set()
        mapping: dict[str, str] = {}

        for i, label in enumerate(local_labels):
            best_speaker = label
            best_overlap = 0.0
            for j, g_speaker in enumerate(global_speakers):
                if g_speaker in used_global:
                    continue
                if overlap_matrix[i][j] > best_overlap:
                    best_overlap = overlap_matrix[i][j]
                    best_speaker = g_speaker
            mapping[label] = best_speaker
            used_global.add(best_speaker)

        return mapping


class OverlapSeparator:
    """Separate overlapping speakers using Pyannote PixIT and re-ASR each stream.

    Usage:
        separator = OverlapSeparator(device="cuda")
        overlap_segments = separator.separate(audio, diarization, asr_backend)
        separator.cleanup()
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "pyannote/speech-separation-ami-1.0",
        padding: float = 3.0,
        num_speakers: int | None = None,
    ) -> None:
        self._device = device
        self._padding = padding
        self._num_speakers = num_speakers
        self._segmenter = SubtitleSegmenter()
        self._pipeline = self._load_pipeline(model_name)

    def _load_pipeline(self, model_name: str):
        """Load Pyannote SpeechSeparation pipeline."""
        import os

        # Lazy import: torch only needed when separator is actually used
        import torch  # noqa: F811

        # Apply same compatibility patches as diarizer
        from transcribe.models.diarizer import (
            _patch_torchaudio_for_pyannote,
            _patch_huggingface_hub_for_pyannote,
            _patch_torch_load_for_pyannote,
        )
        _patch_torchaudio_for_pyannote()
        _patch_huggingface_hub_for_pyannote()
        _patch_torch_load_for_pyannote()

        # Patch speechbrain compatibility: speechbrain 1.x passes `token`
        # through **kwargs to Pretrained.__init__() which doesn't accept it.
        self._patch_speechbrain_token()

        token = os.environ.get("HF_TOKEN")

        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            model_name,
            token=token,
            cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        )
        if self._device != "cpu" and torch.cuda.is_available():
            pipeline = pipeline.to(torch.device(self._device))
        return pipeline

    @staticmethod
    def _patch_speechbrain_token() -> None:
        """Monkey-patch speechbrain to handle extra kwargs from pyannote.

        SpeechBrain 1.x ``pretrained_from_hparams()`` passes all kwargs
        through to ``Pretrained.__init__()`` which only accepts ``modules``
        and ``hparams``.  Pyannote passes ``token``, ``huggingface_cache_dir``,
        and ``revision`` which cause ``TypeError``.  This patch pops those
        kwargs before class instantiation.
        """
        try:
            from speechbrain.inference import interfaces as _sb_iface
        except ImportError:
            return

        if getattr(_sb_iface, "_token_patched", False):
            return

        _orig_pretrained = _sb_iface.pretrained_from_hparams

        # kwargs not accepted by Pretrained.__init__()
        _UNSUPPORTED_KWARGS = frozenset({
            "token", "huggingface_cache_dir", "revision",
        })

        def _patched_pretrained(*args, **kwargs):
            for key in _UNSUPPORTED_KWARGS:
                kwargs.pop(key, None)
            # speechbrain expects device as string, not torch.device
            run_opts = kwargs.get("run_opts")
            if isinstance(run_opts, dict) and "device" in run_opts:
                import torch
                dev = run_opts["device"]
                if isinstance(dev, torch.device):
                    run_opts["device"] = str(dev)
            return _orig_pretrained(*args, **kwargs)

        _sb_iface.pretrained_from_hparams = _patched_pretrained
        _sb_iface._token_patched = True

    def separate(
        self,
        audio: AudioSegment,
        diarization: DiarizationResult,
        asr_backend,
    ) -> list[TranscriptSegment]:
        """Separate overlapping speech and transcribe each speaker.

        Args:
            audio: Full audio segment.
            diarization: Diarization result with overlap_regions.
            asr_backend: ASR backend with transcribe_words() method.

        Returns:
            TranscriptSegments for overlap regions only, with correct
            speaker attribution from separated streams.
        """
        overlap_regions = diarization.overlap_regions
        if not overlap_regions:
            return []

        # Step 1: Extract padded, merged clips
        clips = extract_overlap_clips(audio, overlap_regions, self._padding)

        if not clips:
            return []

        all_segments: list[TranscriptSegment] = []

        for clip in clips:
            try:
                clip_segments = self._process_clip(
                    clip, diarization, asr_backend,
                )
                all_segments.extend(clip_segments)
            except Exception:
                _logger.exception(
                    "Failed to separate clip %.1f-%.1fs, skipping",
                    clip.start_time, clip.end_time,
                )

        return all_segments

    def _process_clip(
        self,
        clip: OverlapClip,
        diarization: DiarizationResult,
        asr_backend,
    ) -> list[TranscriptSegment]:
        """Process a single overlap clip: separate → ASR → trim → segment."""
        import torch  # noqa: F811

        # Step 2: Run PixIT separation
        waveform_tensor = torch.tensor(clip.waveform, dtype=torch.float32)
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)

        input_dict = {
            "waveform": waveform_tensor,
            "sample_rate": clip.sample_rate,
        }
        kwargs = {}
        if self._num_speakers is not None:
            kwargs["num_speakers"] = self._num_speakers

        dia_annotation, sources = self._pipeline(input_dict, **kwargs)

        # Step 3: Get local speaker labels and active times
        local_labels = list(dia_annotation.labels())
        if not local_labels:
            return []

        # Compute per-speaker active time ranges from diarization
        local_times: list[tuple[float, float]] = []
        for label in local_labels:
            label_start = float("inf")
            label_end = float("-inf")
            for turn, _, spk in dia_annotation.itertracks(yield_label=True):
                if spk == label:
                    label_start = min(label_start, turn.start + clip.start_time)
                    label_end = max(label_end, turn.end + clip.start_time)
            if label_start == float("inf"):
                label_start = clip.start_time
                label_end = clip.start_time
            local_times.append((label_start, label_end))

        # Step 4: Map local labels to global speaker IDs
        global_speakers = list({s.speaker_id for s in diarization.segments})
        speaker_map = _map_local_to_global_speakers(
            local_labels=local_labels,
            local_times=local_times,
            global_speakers=global_speakers,
            global_segs=diarization.segments,
        )

        # Step 5: For each separated speaker, run ASR
        result_segments: list[TranscriptSegment] = []

        # Validate sources shape: (num_samples, num_speakers)
        if sources.data.ndim != 2:
            raise ValueError(
                f"Expected 2D sources array (samples, speakers), got shape {sources.data.shape}"
            )
        if sources.data.shape[1] != len(local_labels):
            raise ValueError(
                f"Sources has {sources.data.shape[1]} speakers but got {len(local_labels)} labels"
            )

        for s_idx, label in enumerate(local_labels):
            global_speaker = speaker_map.get(label, label)

            # Extract this speaker's separated audio
            # PixIT returns sources.data as (num_samples, num_speakers) waveform array
            separated_waveform = sources.data[:, s_idx].copy()
            if separated_waveform.ndim != 1:
                raise ValueError(
                    f"Expected 1D waveform for speaker {s_idx}, got shape {separated_waveform.shape}"
                )
            separated_audio = AudioSegment(
                waveform=separated_waveform,
                sample_rate=clip.sample_rate,
                start_time=clip.start_time,
                end_time=clip.end_time,
            )

            # ASR on separated stream
            words = asr_backend.transcribe_words(separated_audio)
            if not words:
                continue

            # Trim to original overlap regions only
            words = _trim_words_to_overlaps(words, clip.source_overlaps)
            if not words:
                continue

            # Segment into subtitle lines
            segments = self._segmenter.segment(words)

            # Tag with global speaker ID and overlap flag
            for seg in segments:
                result_segments.append(replace(
                    seg,
                    speaker_id=global_speaker,
                    is_overlap=True,
                ))

        return result_segments

    def cleanup(self) -> None:
        """Release PixIT model from GPU/CPU memory."""
        self._pipeline = None
        if self._device != "cpu":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

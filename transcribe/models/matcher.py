"""Stage 4.5: Speaker matching via voice embedding cosine similarity.

Resolves the core problem: separated audio tracks are anonymous — we don't
know which track corresponds to which speaker. This module uses the
ERes2NetV2 speaker embedding model (from 3D-Speaker / ModelScope) to extract
192-dim voice embeddings and match tracks to speakers via cosine similarity.

The ERes2NetV2 model (`iic/speech_eres2netv2_sv_zh-cn_16k-common`) is
optimized for Chinese speakers, achieving 6.14% EER on CN-Celeb — a
significant improvement over the previous SpeechBrain ECAPA-TDNN baseline.

Matching uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
for globally optimal 1:1 assignment instead of greedy matching. Embeddings
for reference speakers are computed as duration-weighted averages across
multiple segments for improved robustness.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from transcribe.data.types import AudioSegment, DiarizationResult

# Minimum reference segment duration in seconds for reliable embedding extraction
_MIN_REF_SECONDS = 0.5

# Default cosine similarity threshold for matching
DEFAULT_MATCH_THRESHOLD = 0.5

# Supported audio extensions for reference samples
_SUPPORTED_AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"})


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _hungarian_match(
    sim_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    threshold: float,
) -> dict[str, str]:
    """Optimal 1:1 assignment via Hungarian algorithm.

    Args:
        sim_matrix: (n_rows × n_cols) cosine similarity matrix.
        row_labels: Label for each row.
        col_labels: Label for each column.
        threshold: Minimum similarity to accept a match.

    Returns:
        {row_label: col_label} for accepted pairs. Pairs below threshold
        are omitted from the mapping.
    """
    if sim_matrix.size == 0:
        return {}

    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[str, str] = {}
    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i, j] >= threshold:
            mapping[row_labels[i]] = col_labels[j]

    return mapping


def _extract_embedding(
    waveform: np.ndarray,
    sample_rate: int,
    model,
) -> np.ndarray | None:
    """Extract a single embedding from a waveform segment.

    Args:
        waveform: 1-D float32 numpy array.
        sample_rate: Audio sample rate.
        model: ERes2NetV2 model from modelscope.

    Returns:
        192-dim embedding vector, or None if segment is too short.
    """
    if len(waveform) < int(sample_rate * _MIN_REF_SECONDS):
        return None

    with torch.no_grad():
        embedding = model(waveform)
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    return embedding.flatten()


def _load_reference_audio(
    audio_path: str, extractor, sample_rate: int = 16_000
) -> np.ndarray:
    """Load an audio file and resample to target sample rate via FFmpeg.

    Args:
        audio_path: Path to audio file.
        extractor: AudioExtractor instance to reuse.
        sample_rate: Target sample rate.

    Returns:
        1-D float32 waveform at the target sample rate.
    """
    segment = extractor.extract(audio_path, sample_rate=sample_rate)
    return segment.waveform


def _find_reference_segments(
    diarization: DiarizationResult,
    max_segments: int = 3,
) -> dict[str, list[tuple[float, float]]]:
    """Find the top-N longest non-overlap segments for each speaker.

    Args:
        diarization: Diarization result with speaker segments.
        max_segments: Maximum number of segments per speaker (default 3).

    Returns:
        {speaker_id: [(start_time, end_time), ...]} with segments sorted
        by duration descending. Returns up to `max_segments` segments per
        speaker. Falls back to overlap segments if no clean segment exists.
    """
    # Collect non-overlap segments per speaker, sorted by duration desc
    candidates: dict[str, list[tuple[float, float, float]]] = {}

    for seg in diarization.segments:
        if seg.is_overlap:
            continue
        duration = seg.end_time - seg.start_time
        if duration < _MIN_REF_SECONDS:
            continue

        if seg.speaker_id not in candidates:
            candidates[seg.speaker_id] = []
        candidates[seg.speaker_id].append((seg.start_time, seg.end_time, duration))

    # Sort each speaker's segments by duration descending, take top-N
    result: dict[str, list[tuple[float, float]]] = {}
    for speaker_id, segs in candidates.items():
        segs.sort(key=lambda x: x[2], reverse=True)
        result[speaker_id] = [(s, e) for s, e, _ in segs[:max_segments]]

    # Fallback: speakers with no non-overlap segments get longest segment
    # (may be an overlap segment; still enforce minimum duration)
    all_speakers = {seg.speaker_id for seg in diarization.segments}
    for speaker_id in all_speakers:
        if speaker_id in result:
            continue
        best: tuple[float, float, float] | None = None
        for seg in diarization.segments:
            if seg.speaker_id != speaker_id:
                continue
            duration = seg.end_time - seg.start_time
            if duration < _MIN_REF_SECONDS:
                continue
            if best is None or duration > best[2]:
                best = (seg.start_time, seg.end_time, duration)
        if best is not None:
            result[speaker_id] = [(best[0], best[1])]

    return result


def _extract_speaker_embeddings(
    audio: AudioSegment,
    diarization: DiarizationResult,
    model,
    max_segments: int = 3,
) -> dict[str, np.ndarray]:
    """Extract duration-weighted average embeddings for each speaker.

    For each speaker, selects up to `max_segments` longest non-overlap
    segments, extracts embeddings from each, and computes a duration-
    weighted average embedding.

    Args:
        audio: Original audio segment.
        diarization: Diarization result.
        model: Speaker embedding model.
        max_segments: Max reference segments per speaker.

    Returns:
        {speaker_id: embedding vector} for speakers with valid segments.
    """
    ref_segments = _find_reference_segments(diarization, max_segments=max_segments)
    speaker_embeddings: dict[str, np.ndarray] = {}

    for speaker_id, segments in ref_segments.items():
        embeddings: list[np.ndarray] = []
        weights: list[float] = []

        for start_t, end_t in segments:
            start_sample = int((start_t - audio.start_time) * audio.sample_rate)
            end_sample = int((end_t - audio.start_time) * audio.sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio.waveform), end_sample)
            if end_sample <= start_sample:
                continue

            chunk = audio.waveform[start_sample:end_sample]
            emb = _extract_embedding(chunk, audio.sample_rate, model)
            if emb is not None:
                embeddings.append(emb)
                weights.append(end_t - start_t)

        if not embeddings:
            continue

        # Duration-weighted average, then re-normalize to unit vector
        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr /= weights_arr.sum()
        avg = np.average(embeddings, axis=0, weights=weights_arr)
        norm = np.linalg.norm(avg)
        if norm > 1e-8:
            avg = avg / norm
        speaker_embeddings[speaker_id] = avg

    return speaker_embeddings


class SpeakerMatcher:
    """Match separated audio tracks to speaker labels via voice embeddings.

    Uses the ERes2NetV2 speaker embedding model (3D-Speaker / ModelScope)
    optimized for Chinese speakers to extract 192-dim voice embeddings and
    cosine similarity to match tracks to speakers.
    """

    # Default model: ERes2NetV2 trained on 200K Chinese speakers (CN-Celeb EER 6.14%)
    _DEFAULT_MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"

    def __init__(
        self,
        device: str = "cpu",
        embedding_model: str = _DEFAULT_MODEL,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self._device = device
        self._match_threshold = match_threshold
        self._model = self._load_model(embedding_model)
        self._user_references: dict[str, np.ndarray] | None = None

    def _load_model(self, model_name: str):
        """Load ERes2NetV2 speaker embedding model via ModelScope.

        The model handles FBank feature extraction internally, accepting
        raw numpy float32 waveforms at 16kHz.
        """
        from modelscope.models import Model

        model = Model.from_pretrained(model_name, device=self._device)
        model.eval()
        return model

    def register_speakers(
        self, reference_dir: str
    ) -> dict[str, np.ndarray]:
        """Load speaker reference audio files and compute embeddings.

        Scans the directory for audio files. The filename (without extension)
        is used as the speaker name.

        Args:
            reference_dir: Path to directory containing speaker audio samples.

        Returns:
            {speaker_name: 192-dim embedding vector} for all successfully
            loaded speakers.

        Raises:
            FileNotFoundError: If reference_dir does not exist.
            ValueError: If no valid audio files are found.
        """
        ref_path = Path(reference_dir)
        if not ref_path.is_dir():
            raise FileNotFoundError(
                f"Speaker reference directory not found: {reference_dir}"
            )

        from transcribe.models.audio_extractor import AudioExtractor

        extractor = AudioExtractor()

        embeddings: dict[str, np.ndarray] = {}
        for filepath in sorted(ref_path.iterdir()):
            if filepath.suffix.lower() not in _SUPPORTED_AUDIO_EXTS:
                continue

            speaker_name = filepath.stem
            waveform = _load_reference_audio(str(filepath), extractor)
            embedding = _extract_embedding(waveform, 16_000, self._model)
            if embedding is not None:
                embeddings[speaker_name] = embedding

        if not embeddings:
            raise ValueError(
                f"No valid audio files found in speaker reference directory: {reference_dir}"
            )

        self._user_references = embeddings
        return embeddings

    def match_speakers_to_references(
        self,
        audio: AudioSegment,
        diarization: DiarizationResult,
    ) -> dict[str, str]:
        """Map diarized speaker IDs to user-provided speaker names.

        For each SPEAKER_XX from diarization, extracts duration-weighted
        embeddings from reference segments and matches them against
        user-provided reference embeddings via cosine similarity.

        Args:
            audio: Original (denoised) full audio.
            diarization: Diarization result with speaker segments.

        Returns:
            {SPEAKER_XX: user_name} mapping. Speakers without a match above
            threshold are omitted from the mapping.
        """
        if not self._user_references:
            return {}

        # Extract embeddings for each diarized speaker (multi-segment averaging)
        speaker_embeddings = _extract_speaker_embeddings(
            audio, diarization, self._model
        )

        # Build similarity matrix and match via Hungarian algorithm
        speaker_ids = list(speaker_embeddings.keys())
        user_names = list(self._user_references.keys())

        sim_matrix = np.zeros((len(speaker_ids), len(user_names)))
        for i, spk_id in enumerate(speaker_ids):
            for j, usr_name in enumerate(user_names):
                sim_matrix[i, j] = _cosine_similarity(
                    speaker_embeddings[spk_id], self._user_references[usr_name]
                )

        name_map = _hungarian_match(
            sim_matrix, speaker_ids, user_names, self._match_threshold
        )

        return name_map

    def match_tracks_to_speakers(
        self,
        separated_tracks: list[AudioSegment],
        audio: AudioSegment,
        diarization: DiarizationResult,
    ) -> dict[int, str]:
        """Map separated track indices to speaker IDs.

        Args:
            separated_tracks: List of separated audio tracks.
            audio: Original (denoised) full audio.
            diarization: Diarization result with speaker segments.

        Returns:
            {track_index: speaker_id} mapping. Unmatched tracks
            are assigned "UNKNOWN".
        """
        if not separated_tracks:
            return {}

        # Extract reference embeddings (multi-segment averaging)
        ref_embeddings = _extract_speaker_embeddings(
            audio, diarization, self._model
        )

        # Extract track embeddings
        track_embeddings: list[np.ndarray | None] = []
        for track in separated_tracks:
            embedding = _extract_embedding(
                track.waveform, track.sample_rate, self._model
            )
            track_embeddings.append(embedding)

        # Build similarity matrix and match via Hungarian algorithm
        valid_indices = [i for i, e in enumerate(track_embeddings) if e is not None]
        speaker_ids = list(ref_embeddings.keys())

        sim_matrix = np.zeros((len(valid_indices), len(speaker_ids)))
        for row, track_idx in enumerate(valid_indices):
            for col, spk_id in enumerate(speaker_ids):
                sim_matrix[row, col] = _cosine_similarity(
                    track_embeddings[track_idx], ref_embeddings[spk_id]
                )

        matched = _hungarian_match(
            sim_matrix,
            [str(i) for i in valid_indices],
            speaker_ids,
            self._match_threshold,
        )

        mapping: dict[int, str] = {}
        for idx_str, spk_id in matched.items():
            mapping[int(idx_str)] = spk_id

        # Assign UNKNOWN to unmatched tracks
        for idx in range(len(separated_tracks)):
            if idx not in mapping:
                mapping[idx] = "UNKNOWN"

        return mapping

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        self._model = None
        self._user_references = None
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

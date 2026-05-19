"""Stage 4.5: Speaker matching via voice embedding cosine similarity.

Resolves the core problem: separated audio tracks are anonymous — we don't
know which track corresponds to which speaker. This module uses SpeechBrain's
ECAPA-TDNN speaker embedding model to extract 192-dim voice embeddings and
match tracks to speaker labels via cosine similarity.

Uses SpeechBrain directly (bypasses Pyannote's PretrainedSpeakerEmbedding
wrapper to avoid use_auth_token compatibility issues with SpeechBrain 1.x).
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

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


def _extract_embedding(
    waveform: np.ndarray,
    sample_rate: int,
    model,
) -> np.ndarray | None:
    """Extract a single embedding from a waveform segment.

    Args:
        waveform: 1-D float32 numpy array.
        sample_rate: Audio sample rate.
        model: SpeechBrain EncoderClassifier model.

    Returns:
        192-dim embedding vector, or None if segment is too short.
    """
    if len(waveform) < int(sample_rate * _MIN_REF_SECONDS):
        return None

    # SpeechBrain expects (batch, time) tensor
    wav_tensor = torch.tensor(waveform, dtype=torch.float32)
    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)  # [1, T]

    with torch.no_grad():
        embedding = model.encode_batch(wav_tensor)
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    # Embedding shape: [1, 1, 192] → [192]
    return embedding.squeeze()


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
) -> dict[str, tuple[float, float]]:
    """Find the longest non-overlap segment for each speaker as reference.

    Returns:
        {speaker_id: (start_time, end_time)} of the best reference segment.
    """
    references: dict[str, tuple[float, float, float]] = {}  # speaker → (start, end, duration)

    for seg in diarization.segments:
        # Skip overlap segments — we want clean single-speaker audio
        if seg.is_overlap:
            continue

        duration = seg.end_time - seg.start_time
        if duration < _MIN_REF_SECONDS:
            continue

        if seg.speaker_id not in references or duration > references[seg.speaker_id][2]:
            references[seg.speaker_id] = (seg.start_time, seg.end_time, duration)

    # If a speaker has no non-overlap segments, fall back to the longest
    # segment regardless of overlap status
    for seg in diarization.segments:
        if seg.speaker_id in references:
            continue
        duration = seg.end_time - seg.start_time
        if seg.speaker_id not in references or duration > references[seg.speaker_id][2]:
            references[seg.speaker_id] = (seg.start_time, seg.end_time, duration)

    return {spk: (s, e) for spk, (s, e, _) in references.items()}


class SpeakerMatcher:
    """Match separated audio tracks to speaker labels via voice embeddings.

    Uses SpeechBrain's ECAPA-TDNN speaker embedding model directly to extract
    192-dim voice embeddings and cosine similarity to match tracks to speakers.
    """

    def __init__(
        self,
        device: str = "cpu",
        embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self._device = device
        self._match_threshold = match_threshold
        self._model = self._load_model(embedding_model)
        self._user_references: dict[str, np.ndarray] | None = None

    def _load_model(self, model_name: str):
        """Load SpeechBrain ECAPA speaker embedding model directly.

        Bypasses Pyannote's PretrainedSpeakerEmbedding wrapper which has
        use_auth_token incompatibility with SpeechBrain 1.x.
        """
        from speechbrain.inference.speaker import EncoderClassifier

        model = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=f"pretrained_models/{model_name.split('/')[-1]}",
            run_opts={"device": self._device},
        )
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

        For each SPEAKER_XX from diarization, extracts an embedding from the
        best reference segment and matches it against user-provided reference
        embeddings via cosine similarity.

        Args:
            audio: Original (denoised) full audio.
            diarization: Diarization result with speaker segments.

        Returns:
            {SPEAKER_XX: user_name} mapping. Speakers without a match above
            threshold are omitted from the mapping.
        """
        if not self._user_references:
            return {}

        # Find reference segments for each diarized speaker
        ref_segments = _find_reference_segments(diarization)

        # Extract embeddings for each diarized speaker
        speaker_embeddings: dict[str, np.ndarray] = {}
        for speaker_id, (start_t, end_t) in ref_segments.items():
            start_sample = int((start_t - audio.start_time) * audio.sample_rate)
            end_sample = int((end_t - audio.start_time) * audio.sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio.waveform), end_sample)
            if end_sample <= start_sample:
                continue

            chunk = audio.waveform[start_sample:end_sample]
            embedding = _extract_embedding(chunk, audio.sample_rate, self._model)
            if embedding is not None:
                speaker_embeddings[speaker_id] = embedding

        # Match each diarized speaker to user references
        name_map: dict[str, str] = {}
        used_names: set[str] = set()

        # Build all scored pairs and greedily assign
        scored: list[tuple[str, str, float]] = []
        for speaker_id, spk_emb in speaker_embeddings.items():
            for user_name, ref_emb in self._user_references.items():
                sim = _cosine_similarity(spk_emb, ref_emb)
                scored.append((speaker_id, user_name, sim))

        scored.sort(key=lambda x: x[2], reverse=True)
        for speaker_id, user_name, sim in scored:
            if speaker_id in name_map or user_name in used_names:
                continue
            if sim < self._match_threshold:
                continue
            name_map[speaker_id] = user_name
            used_names.add(user_name)

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

        # Step 1: Find reference segments for each speaker
        ref_segments = _find_reference_segments(diarization)

        # Step 2: Extract reference embeddings
        ref_embeddings: dict[str, np.ndarray] = {}
        for speaker_id, (start_t, end_t) in ref_segments.items():
            start_sample = int((start_t - audio.start_time) * audio.sample_rate)
            end_sample = int((end_t - audio.start_time) * audio.sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio.waveform), end_sample)
            if end_sample <= start_sample:
                continue

            chunk = audio.waveform[start_sample:end_sample]
            embedding = _extract_embedding(chunk, audio.sample_rate, self._model)
            if embedding is not None:
                ref_embeddings[speaker_id] = embedding

        # Step 3: Extract track embeddings
        track_embeddings: list[np.ndarray | None] = []
        for track in separated_tracks:
            embedding = _extract_embedding(
                track.waveform, track.sample_rate, self._model
            )
            track_embeddings.append(embedding)

        # Step 4: Match tracks to speakers via cosine similarity
        mapping: dict[int, str] = {}
        used_speakers: set[str] = set()

        # Build similarity matrix
        scored: list[tuple[int, str, float]] = []
        for track_idx, track_emb in enumerate(track_embeddings):
            if track_emb is None:
                continue
            for speaker_id, ref_emb in ref_embeddings.items():
                sim = _cosine_similarity(track_emb, ref_emb)
                scored.append((track_idx, speaker_id, sim))

        # Sort by similarity descending and greedily assign
        scored.sort(key=lambda x: x[2], reverse=True)
        for track_idx, speaker_id, sim in scored:
            if track_idx in mapping or speaker_id in used_speakers:
                continue
            if sim < self._match_threshold:
                continue
            mapping[track_idx] = speaker_id
            used_speakers.add(speaker_id)

        # Assign UNKNOWN to unmatched tracks
        for idx in range(len(separated_tracks)):
            if idx not in mapping:
                mapping[idx] = "UNKNOWN"

        return mapping

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

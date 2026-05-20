# Phase 1: Diarizer & Matcher Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade diarizer to Pyannote 4.0 Community-1, replace greedy matching with Hungarian algorithm, and implement multi-segment duration-weighted embedding averaging.

**Architecture:** Three independent changes: (1) diarizer default model upgrade, (2) matcher algorithm upgrade (Hungarian), (3) matcher embedding upgrade (multi-segment averaging). Changes touch diarizer.py, matcher.py, pyproject.toml, config.yaml, and their tests.

**Tech Stack:** pyannote.audio 4.0, scipy.optimize.linear_sum_assignment, numpy

---

### Task 1: Update pyproject.toml dependencies

**Files:**
- Modify: `pyproject.toml:20-23`

- [ ] **Step 1: Update pyannote version constraint and add scipy**

Change `pyannote.audio>=3.1,<4` to `pyannote.audio>=3.1` (allow 4.x). Add `scipy` to diarize dependencies.

```toml
diarize = [
    "pyannote.audio>=3.1",
    "scipy",
    "speechbrain>=1.0",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: allow pyannote 4.x and add scipy dependency"
```

---

### Task 2: Upgrade Diarizer to Pyannote 4.0 Community-1

**Files:**
- Modify: `transcribe/models/diarizer.py:1,154`
- Modify: `config.yaml:7`

- [ ] **Step 1: Update default model name in Diarizer**

In `transcribe/models/diarizer.py`, change:

1. Module docstring (line 1): `"""Stage 3: Speaker diarization using Pyannote Audio."""` (remove "3.1")
2. Default model_name in `__init__` (line 154): from `"pyannote/speaker-diarization-3.1"` to `"pyannote/speaker-diarization-community-1"`

- [ ] **Step 2: Update config.yaml default model**

Change `config.yaml` line 7 from `model: pyannote/speaker-diarization-3.1` to `model: pyannote/speaker-diarization-community-1`.

- [ ] **Step 3: Verify existing tests pass**

Run: `uv run pytest tests/test_diarizer.py -v`
Expected: All 7 tests PASS (they mock `_load_pipeline` so model name change doesn't affect them).

- [ ] **Step 4: Commit**

```bash
git add transcribe/models/diarizer.py config.yaml
git commit -m "feat: upgrade diarizer default to pyannote/speaker-diarization-community-1"
```

---

### Task 3: Replace greedy matching with Hungarian algorithm

**Files:**
- Modify: `transcribe/models/matcher.py`
- Modify: `tests/test_matcher.py`

- [ ] **Step 1: Add scipy import and `_hungarian_match` helper to matcher.py**

Add at top of `matcher.py` (after existing imports):

```python
from scipy.optimize import linear_sum_assignment
```

Add new helper function after `_cosine_similarity`:

```python
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
```

- [ ] **Step 2: Replace greedy matching in `match_speakers_to_references`**

Replace lines 238–257 (the greedy matching block in `match_speakers_to_references`) with:

```python
        # Build similarity matrix and match via Hungarian algorithm
        speaker_ids = list(speaker_embeddings.keys())
        user_names = list(self._user_references.keys())

        sim_matrix = np.zeros((len(speaker_ids), len(user_names)))
        for i, spk_emb in enumerate(speaker_ids):
            for j, usr_name in enumerate(user_names):
                sim_matrix[i, j] = _cosine_similarity(
                    speaker_embeddings[spk_emb], self._user_references[usr_name]
                )

        name_map = _hungarian_match(
            sim_matrix, speaker_ids, user_names, self._match_threshold
        )
```

- [ ] **Step 3: Replace greedy matching in `match_tracks_to_speakers`**

Replace lines 307–333 (the greedy matching block + UNKNOWN assignment in `match_tracks_to_speakers`) with:

```python
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
```

- [ ] **Step 4: Add unit tests for Hungarian matching**

Add to `tests/test_matcher.py`:

```python
def test_hungarian_match_optimal_assignment():
    """Hungarian algorithm finds globally optimal assignment, not greedy."""
    from transcribe.models.matcher import _hungarian_match

    # Speaker A is closest to Ref1, Speaker B is closest to Ref1 too
    # but greedy would assign A→Ref1, leaving B unmatched.
    # Optimal: A→Ref2, B→Ref1 (total sim = 1.75 > greedy's 0.95)
    sim = np.array([
        [0.95, 0.80],  # Speaker A: very close to Ref1, close to Ref2
        [0.90, 0.20],  # Speaker B: close to Ref1, far from Ref2
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
    """Works with more rows than columns (more speakers than references)."""
    from transcribe.models.matcher import _hungarian_match

    sim = np.array([
        [0.9, 0.2],
        [0.3, 0.8],
        [0.1, 0.1],
    ])
    result = _hungarian_match(sim, ["A", "B", "C"], ["Ref1", "Ref2"], threshold=0.5)
    assert result["A"] == "Ref1"
    assert result["B"] == "Ref2"
    assert "C" not in result  # below threshold
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_matcher.py -v -k "not slow"`
Expected: All unit tests PASS (including new Hungarian tests).

- [ ] **Step 6: Commit**

```bash
git add transcribe/models/matcher.py tests/test_matcher.py
git commit -m "feat: replace greedy matching with Hungarian algorithm for optimal 1:1 assignment"
```

---

### Task 4: Implement multi-segment duration-weighted embedding averaging

**Files:**
- Modify: `transcribe/models/matcher.py`
- Modify: `tests/test_matcher.py`

- [ ] **Step 1: Add `_find_reference_segments_multi` function**

Add new function after `_find_reference_segments` in `matcher.py`. Keep the old `_find_reference_segments` for backward compatibility with the fallback path, or refactor it to call the new one internally.

Actually, refactor `_find_reference_segments` to return multiple segments:

Replace `_find_reference_segments` (lines 83–114) with:

```python
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
    candidates: dict[str, list[tuple[float, float, float]]] = {}  # speaker → [(start, end, dur)]

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
    all_speakers = {seg.speaker_id for seg in diarization.segments}
    for speaker_id in all_speakers:
        if speaker_id in result:
            continue
        best: tuple[float, float, float] | None = None
        for seg in diarization.segments:
            if seg.speaker_id != speaker_id:
                continue
            duration = seg.end_time - seg.start_time
            if best is None or duration > best[2]:
                best = (seg.start_time, seg.end_time, duration)
        if best is not None:
            result[speaker_id] = [(best[0], best[1])]

    return result
```

- [ ] **Step 2: Add `_extract_speaker_embeddings` helper**

Add new function after `_find_reference_segments`:

```python
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
        {speaker_id: 192-dim embedding vector} for speakers with valid segments.
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
```

- [ ] **Step 3: Simplify `match_speakers_to_references` to use the new helper**

Replace the embedding extraction block (lines 220–236 in the original) with:

```python
        # Extract embeddings for each diarized speaker (multi-segment averaging)
        speaker_embeddings = _extract_speaker_embeddings(
            audio, diarization, self._model
        )
```

- [ ] **Step 4: Simplify `match_tracks_to_speakers` to use the new helper**

Replace the ref embedding extraction block (lines 281–296 in the original) with:

```python
        # Extract reference embeddings (multi-segment averaging)
        ref_embeddings = _extract_speaker_embeddings(
            audio, diarization, self._model
        )
```

- [ ] **Step 5: Update existing test for `_find_reference_segments`**

Update `test_find_reference_segments_picks_longest_non_overlap` in `tests/test_matcher.py`:

```python
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
    # Now returns list of segments; first should be longest
    assert refs["SPEAKER_00"][0] == (2.0, 4.5)  # longest first
    assert refs["SPEAKER_01"][0] == (3.0, 5.0)
```

- [ ] **Step 6: Add test for multi-segment return**

```python
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
```

- [ ] **Step 7: Add test for `_extract_speaker_embeddings`**

```python
def test_extract_speaker_embeddings_with_mock():
    """Multi-segment averaging produces unit-norm embedding."""
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

    mock_model = MagicMock()
    mock_model.encode_batch.return_value = torch.randn(1, 1, 192)

    embeddings = _extract_speaker_embeddings(audio, diarization, mock_model)
    assert "SPEAKER_00" in embeddings
    assert embeddings["SPEAKER_00"].shape == (192,)
    # Should be approximately unit norm
    norm = np.linalg.norm(embeddings["SPEAKER_00"])
    assert abs(norm - 1.0) < 0.01
```

Note: This test needs `import torch` at the top of the test file.

- [ ] **Step 8: Run all matcher tests**

Run: `uv run pytest tests/test_matcher.py -v -k "not slow"`
Expected: All unit tests PASS.

- [ ] **Step 9: Commit**

```bash
git add transcribe/models/matcher.py tests/test_matcher.py
git commit -m "feat: multi-segment duration-weighted embedding averaging for speaker matching"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v -k "not slow"`
Expected: All tests PASS.

- [ ] **Step 2: Run slow integration tests (optional, requires models)**

Run: `uv run pytest -v -m slow`
Note: This requires model downloads. Skip if offline.

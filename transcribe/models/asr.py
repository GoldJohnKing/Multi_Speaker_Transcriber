"""Speech recognition using Fun-ASR-Nano with hotword support."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from funasr import AutoModel

from transcribe.data.types import AudioSegment, TranscriptSegment

# Chinese punctuation characters that may be inserted between hotword chars
_PUNC_PATTERN = r"[，。？！、；：""''（）【】《》…—· ]*"


def restore_hotwords(text: str, hotword_list: list[str]) -> str:
    """Remove punctuation inserted inside hotword terms.

    The ASR model's punctuation mechanism may insert commas, periods, etc. in
    the middle of a correctly-recognised hotword (e.g. ``朽，叶`` for the
    hotword ``朽叶``).  This function detects such breakages and restores the
    original hotword form.

    Args:
        text: ASR output text (already punctuated).
        hotword_list: List of hotword terms to protect.

    Returns:
        Text with hotword-internal punctuation removed.
    """
    if not text or not hotword_list:
        return text

    # Only multi-char hotwords can have internal punctuation
    multi_char = [w for w in hotword_list if len(w) >= 2]
    if not multi_char:
        return text

    # Build combined alternation pattern
    parts: list[str] = []
    for hw in multi_char:
        chars = list(hw)
        pattern = "".join(
            re.escape(c) + _PUNC_PATTERN for c in chars[:-1]
        ) + re.escape(chars[-1])
        parts.append(pattern)
    combined = re.compile("|".join(parts))

    hotword_set = set(hotword_list)

    def _replace(match: re.Match[str]) -> str:
        matched = match.group(0)
        # Strip all punctuation/spaces to get bare characters
        bare = re.sub(_PUNC_PATTERN, "", matched)
        if bare in hotword_set:
            return bare
        return matched

    return combined.sub(_replace, text)


def parse_timestamps(
    timestamps: list,
) -> tuple[float | None, float | None]:
    """Extract segment start and end times from ASR timestamp output.

    Handles two timestamp formats:

    1. Fun-ASR-Nano dict format (current):
       [{"start_time": 0.36, "end_time": 0.42, ...}, ...] — times in **seconds**

    2. Paraformer list format (legacy):
       [[start_ms, end_ms], ...] — times in **milliseconds**
       [start_ms, end_ms, start_ms, end_ms, ...] — flat, in **milliseconds**

    Args:
        timestamps: Raw timestamp data from ASR model output.

    Returns:
        Tuple of (start_time_seconds, end_time_seconds).
        (None, None) if timestamps is empty.
    """
    if not timestamps:
        return None, None

    first = timestamps[0]

    if isinstance(first, dict):
        # Fun-ASR-Nano format: already in seconds
        start = first.get("start_time")
        end = timestamps[-1].get("end_time") if timestamps else None
        return start, end

    if isinstance(first, (list, tuple)):
        # Nested Paraformer format: [[start_ms, end_ms], ...]
        return first[0] / 1000.0, timestamps[-1][1] / 1000.0

    if isinstance(first, (int, float)):
        # Flat Paraformer format: [start_ms, end_ms, start_ms, end_ms, ...]
        return timestamps[0] / 1000.0, timestamps[-1] / 1000.0

    return None, None


class ASRTranscriber:
    """Speech recognition using Fun-ASR-Nano with hotword support."""

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        model_name: str = "iic/Fun-ASR-Nano-2512",
        vad_model: str = "fsmn-vad",
    ) -> None:
        self._device = device
        self._hotword_list = self._load_hotwords(hotword_path)

        model_kwargs: dict = {
            "model": model_name,
            "trust_remote_code": True,
            "vad_model": vad_model,
            "vad_kwargs": {"max_single_segment_time": 30000},
            "device": device,
        }

        # BF16 for GPU (Ampere+) — fp16 is broken on CUDA (outputs all "!")
        if device != "cpu":
            if torch.cuda.is_bf16_supported():
                model_kwargs["bf16"] = True

        self._model = AutoModel(**model_kwargs)

    def _load_hotwords(self, path: str | None) -> list[str]:
        """Load hotwords from text file (one per line).

        Fun-ASR-Nano accepts hotwords as a list of strings.

        Args:
            path: Path to hotword file, or None.

        Returns:
            List of hotword strings. Empty list if path is None or missing.
        """
        if path is None:
            return []
        p = Path(path)
        if not p.exists():
            return []
        return [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps."""
        # Fun-ASR-Nano requires torch tensor wrapped in a list
        waveform_tensor = torch.from_numpy(audio.waveform)

        result = self._model.generate(
            input=[waveform_tensor],
            cache={},
            batch_size=1,
            language="中文",
            itn=True,
            hotwords=self._hotword_list if self._hotword_list else None,
        )

        segments: list[TranscriptSegment] = []
        if not result:
            return segments

        for res in result:
            text = res.get("text", "")
            if not text:
                continue

            # Restore hotword terms broken by LLM punctuation (safety net)
            text = restore_hotwords(text, self._hotword_list)

            # Parse timestamps — handles Fun-ASR-Nano dict format
            timestamps = res.get("timestamps", [])
            parsed_start, parsed_end = parse_timestamps(timestamps)

            if parsed_start is not None and parsed_end is not None:
                start_time = parsed_start + audio.start_time
                end_time = parsed_end + audio.start_time
            else:
                start_time = audio.start_time
                end_time = audio.end_time

            segments.append(
                TranscriptSegment(
                    speaker_id="SPEAKER_00",
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                )
            )

        return segments

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

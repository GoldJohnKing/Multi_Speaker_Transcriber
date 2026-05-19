"""Speech recognition using FunASR SeACo-Paraformer with hotword support."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from funasr import AutoModel

from transcribe.data.types import AudioSegment, TranscriptSegment

# Chinese punctuation characters that ct-punc may insert between hotword chars
_PUNC_PATTERN = r"[，。？！、；：""''（）【】《》…—· ]*"


def restore_hotwords(text: str, hotword_list: list[str]) -> str:
    """Remove punctuation that ct-punc inserted inside hotword terms.

    FunASR's ct-punc model processes Chinese text character-by-character and has
    no awareness of hotword boundaries.  It may insert commas, periods, etc. in
    the middle of a correctly-recognised hotword (e.g. ``朽，叶`` for the
    hotword ``朽叶``).  This function detects such breakages and restores the
    original hotword form.

    Args:
        text: ASR output text (already punctuated by ct-punc).
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


class ASRTranscriber:
    """Speech recognition using FunASR SeACo-Paraformer with hotword support."""

    def __init__(
        self,
        device: str = "cpu",
        hotword_path: str | None = None,
        model_name: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
    ) -> None:
        self._device = device
        self._hotwords, self._hotword_list = self._load_hotwords(hotword_path)
        self._model = AutoModel(
            model=model_name,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
        )

    def _load_hotwords(self, path: str | None) -> tuple[str | None, list[str]]:
        """Load hotwords from text file (one per line), space-joined.

        Returns:
            Tuple of (space-joined hotword string or None, list of individual words).
        """
        if path is None:
            return None, []
        p = Path(path)
        if not p.exists():
            return None, []
        words = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return (" ".join(words) if words else None), words

    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        """Transcribe audio to text segments with timestamps."""
        result = self._model.generate(
            input=audio.waveform,
            batch_size_s=300,
            hotword=self._hotwords,
        )

        segments: list[TranscriptSegment] = []
        if not result:
            return segments

        for res in result:
            text = res.get("text", "")
            timestamps = res.get("timestamp", [])
            if not text or not timestamps:
                continue

            # Restore hotword terms broken by ct-punc punctuation insertion
            text = restore_hotwords(text, self._hotword_list)

            # FunASR timestamp format: [[start_ms, end_ms], [start_ms, end_ms], ...]
            # Each pair corresponds to one recognized character/word.
            if timestamps and isinstance(timestamps[0], (list, tuple)):
                # Nested format: [[start, end], ...]
                start_time = timestamps[0][0] / 1000.0 + audio.start_time
                end_time = timestamps[-1][1] / 1000.0 + audio.start_time
            elif len(timestamps) >= 2:
                # Flat format: [start, end, start, end, ...]
                start_time = timestamps[0] / 1000.0 + audio.start_time
                end_time = timestamps[-1] / 1000.0 + audio.start_time
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

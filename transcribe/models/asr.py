"""Speech recognition using FunASR SeACo-Paraformer with hotword support."""

from __future__ import annotations

from pathlib import Path

from funasr import AutoModel

from transcribe.data.types import AudioSegment, TranscriptSegment


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
        self._hotwords = self._load_hotwords(hotword_path)
        self._model = AutoModel(
            model=model_name,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
        )

    def _load_hotwords(self, path: str | None) -> str | None:
        """Load hotwords from text file (one per line), space-joined."""
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        words = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return " ".join(words) if words else None

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

            if len(timestamps) >= 2:
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

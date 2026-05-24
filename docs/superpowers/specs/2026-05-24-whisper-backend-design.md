# Whisper ASR Backend Design

## Overview

Add a fourth ASR backend using `faster-whisper` (CTranslate2-based Whisper implementation) to the multi-speaker transcription pipeline.

## Decisions

| Item | Decision |
|------|----------|
| Backend name | `"Whisper"` |
| Library | `faster-whisper` (CTranslate2 backend) |
| Default model | `large-v3` (configurable) |
| Default language | `zh` |
| Hotword mechanism | Native `hotwords` parameter (prompt prefix via `<\|startofprev\|>`) |
| Timestamps | `word_timestamps=True` for word-level granularity |
| VAD | Built-in Silero VAD (automatic segmentation) |
| `supports_hotwords` | `True` |

## File Changes

1. **New** `transcribe/models/asr/whisper.py` — `WhisperTranscriber(ASRBase)` implementation
2. **Edit** `transcribe/models/asr/__init__.py` — add `whisper` import
3. **Edit** `transcribe/cli.py` — add `"Whisper"` to `--backend` choices
4. **Edit** `pyproject.toml` — add `faster-whisper` dependency

## Implementation Details

### Class: `WhisperTranscriber`

```python
class WhisperTranscriber(ASRBase):
    def __init__(self, device="cpu", hotword_path=None, *,
                 model_name="large-v3", compute_type=None, language="zh")
    def transcribe(self, audio) -> list[TranscriptSegment]
    def transcribe_words(self, audio) -> list[WordTimestamp]
    def cleanup(self) -> None
```

### Hotword Handling

- Read hotword file (one word per line), space-join into single string
- Pass to faster-whisper's `hotwords` parameter (v1.0.2+)
- Apply `restore_hotwords()` as safety net for punctuation breakage
- Budget: ~223 tokens shared with `condition_on_previous_text` context

### Word Timestamps

faster-whisper returns segments with `.words` when `word_timestamps=True`:
- Each word has `.word`, `.start`, `.end` attributes
- Map directly to `WordTimestamp(word, start_time + audio.start_time, end_time + audio.start_time)`

### Device Mapping

- `"cpu"` → `device="cpu"`, `compute_type="int8"`
- `"cuda"` → `device="cuda"`, `compute_type="float16"` (or `"auto"` based on GPU)
- `compute_type` parameter exposed as configurable override

## Comparison with Existing Backends

| Feature | FunASR Nano/Paraformer | Qwen3-ASR | Whisper |
|---------|----------------------|-----------|---------|
| Hotword mechanism | Decoder bias | LLM prompt | Prompt prefix |
| Timestamps | VAD / ForcedAligner | ForcedAligner | word_timestamps |
| Chinese quality | Excellent | Excellent | Good |
| Multilingual | No | No | Yes |
| Dependency | funasr | qwen-asr | faster-whisper |

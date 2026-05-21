# Dual ASR Backend Support — Design Spec

**Date**: 2026-05-21
**Branch**: Fun-ASR-Nano
**Scope**: Support both Fun-ASR (Paraformer) and Fun-ASR-Nano as selectable ASR backends

## Background

The project migrated from Paraformer (`paraformer-zh`) to Fun-ASR-Nano (`FunAudioLLM/Fun-ASR-Nano-2512`) on the `Fun-ASR-Nano` branch. This spec designs a dual-backend architecture that allows users to choose between the two via CLI or config, while keeping the system extensible for future backends (Qwen3-ASR, GLM-ASR-Nano, Mimo-V2.5-ASR).

## Approach

**Abstract base class + factory function + self-registration.** Each backend is an independent class in its own file, registering itself at import time. The pipeline uses a factory to instantiate the chosen backend. No registry configuration files — just Python imports.

## Directory Structure

```
transcribe/models/asr/
├── __init__.py           # Re-export ASRBase, create_asr, list_backends
├── base.py               # ASRBase abstract base class
├── paraformer.py         # FunASRParaformerTranscriber (from master)
├── nano.py               # FunASRNanoTranscriber (current branch)
├── factory.py            # register_backend, create_asr, list_backends
└── utils.py              # restore_hotwords, parse_timestamps (shared)
```

The current single file `transcribe/models/asr.py` will be **removed** and replaced by this package directory.

## ASRBase Abstract Base Class

```python
# base.py
from abc import ABC, abstractmethod
from transcribe.data.types import AudioSegment, TranscriptSegment

class ASRBase(ABC):
    """Unified interface for all ASR backends."""

    @abstractmethod
    def __init__(self, device: str, hotword_path: str | None, **kwargs) -> None:
        ...

    @abstractmethod
    def transcribe(self, audio: AudioSegment) -> list[TranscriptSegment]:
        ...

    @abstractmethod
    def cleanup(self) -> None:
        ...

    @property
    def supports_hotwords(self) -> bool:
        return True
```

- `__init__` signature: `device` + `hotword_path` + `**kwargs` for backend-specific parameters
- `transcribe` returns `list[TranscriptSegment]` — the only output the pipeline needs
- `cleanup` releases GPU memory
- `supports_hotwords` allows the pipeline to conditionally pass hotword parameters

## Factory & Registration

```python
# factory.py
from transcribe.models.asr.base import ASRBase

_BACKENDS: dict[str, type[ASRBase]] = {}

def register_backend(name: str, cls: type[ASRBase]) -> None:
    _BACKENDS[name] = cls

def create_asr(backend: str, device: str = "cpu",
               hotword_path: str | None = None, **kwargs) -> ASRBase:
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown ASR backend: {backend!r}. "
            f"Available: {list(_BACKENDS.keys())}"
        )
    return cls(device=device, hotword_path=hotword_path, **kwargs)

def list_backends() -> list[str]:
    return list(_BACKENDS.keys())
```

Each backend self-registers at import:

```python
# paraformer.py (bottom of file)
register_backend("Fun-ASR-Paraformer", FunASRParaformerTranscriber)

# nano.py (bottom of file)
register_backend("Fun-ASR-Nano", FunASRNanoTranscriber)
```

`__init__.py` triggers registration:

```python
from transcribe.models.asr.base import ASRBase
from transcribe.models.asr.factory import create_asr, list_backends
from transcribe.models.asr import paraformer, nano  # noqa: F401
```

## Shared Utilities (`utils.py`)

Two functions extracted from the current `asr.py`:

### `restore_hotwords(text, hotword_list) -> str`

Identical logic for both backends. Removes punctuation inserted inside hotword terms. Used by Paraformer (ct-punc breaks hotwords) and Nano (LLM may break hotwords as a safety net).

### `parse_timestamps(timestamps) -> tuple[float | None, float | None]`

Handles both timestamp formats:
- Fun-ASR-Nano: `[{"start_time": 0.36, "end_time": 0.42, ...}]` — seconds, dict format
- Paraformer: `[[start_ms, end_ms], ...]` or `[start_ms, end_ms, ...]` — milliseconds, list format

Returns `(start_seconds, end_seconds)` regardless of input format.

## Backend Implementation Details

### FunASRParaformerTranscriber

Registered as `"Fun-ASR-Paraformer"`. Logic from `master` branch.

| Aspect | Detail |
|--------|--------|
| Model | `paraformer-zh` via `funasr.AutoModel` |
| VAD | Built-in `fsmn-vad` |
| Punctuation | Separate `ct-punc` model |
| Hotword format | Space-joined string (`hotword=` parameter) |
| Input format | numpy array directly |
| Timestamp field | `res["timestamp"]` — milliseconds |
| Timestamp parsing | `parse_timestamps()` handles nested/flat ms formats |

Constructor kwargs: `model_name`, `vad_model`, `punc_model`

### FunASRNanoTranscriber

Registered as `"Fun-ASR-Nano"`. Logic from current `Fun-ASR-Nano` branch.

| Aspect | Detail |
|--------|--------|
| Model | `FunAudioLLM/Fun-ASR-Nano-2512` via `funasr.AutoModel` |
| VAD | Built-in `fsmn-vad` (via `vad_kwargs`) |
| Punctuation | LLM-native (no separate model) |
| Hotword format | List of strings (`hotwords=` parameter) |
| Input format | `torch.from_numpy()` wrapped in `[tensor]` |
| Timestamp field | `res["timestamps"]` — seconds, dict format |
| BF16 | Auto-enabled for GPU (Ampere+) |
| `trust_remote_code` | Required |

Constructor kwargs: `model_name`, `vad_model`

## CLI & Configuration Changes

### CLI (`cli.py`)

```python
parser.add_argument(
    "--backend",
    choices=["Fun-ASR-Paraformer", "Fun-ASR-Nano"],
    default="Fun-ASR-Nano",
    help="ASR backend (default: Fun-ASR-Nano)",
)
```

### PipelineConfig (`data/types.py`)

```python
@dataclass
class PipelineConfig:
    device: str = "auto"
    diarize: bool = True
    backend: str = "Fun-ASR-Nano"   # NEW
    hotwords: str | None = None
    language: str = "zh"
    cache_dir: str = ".cache"
    num_speakers: int | None = None
    speaker_references: str | None = None
```

### config.yaml

```yaml
device: auto
diarize: true
backend: Fun-ASR-Nano    # NEW —可选 Fun-ASR-Paraformer / Fun-ASR-Nano
language: zh
speaker_references: null
```

### config.py

Add `"backend"` to `_PIPELINE_CONFIG_FIELDS`.

### pipeline.py

```python
from transcribe.models.asr import create_asr

# Before:
# transcriber = ASRTranscriber(device=device, hotword_path=config.hotwords)

# After:
transcriber = create_asr(config.backend, device=device, hotword_path=config.hotwords)
```

## Dependency Strategy

Use the git-versioned `funasr` from the current branch as the unified dependency. It is compatible with both Paraformer and Nano APIs:

```toml
[project.optional-dependencies]
asr = [
    "funasr @ git+https://github.com/modelscope/FunASR.git@2ca745e5d11ad9650b94691d0d346d1435dc9b63",
    "modelscope",
    "tiktoken",
    "torch>=2.1",
    "torchaudio>=2.1",
    "transformers",
]
```

No additional dependencies needed for Paraformer support — `ct-punc` and `paraformer-zh` are loaded from ModelScope at runtime.

## Test Changes

- `tests/test_asr.py` updated to test both backends
- Each backend gets its own test class: `TestFunASRParaformerTranscriber`, `TestFunASRNanoTranscriber`
- Shared utility tests: `TestRestoreHotwords`, `TestParseTimestamps`
- Factory tests: `TestCreateASR`
- Slow tests (requiring model download) marked with `@pytest.mark.slow`

## Future Backend Addition (Guideline)

To add a new backend (e.g., Qwen3-ASR):

1. Create `transcribe/models/asr/qwen3.py` with a class extending `ASRBase`
2. Call `register_backend("Qwen3-ASR", Qwen3ASRTranscriber)` at module level
3. Import it in `__init__.py`
4. Add `"Qwen3-ASR"` to CLI choices
5. Encapsulate any extra models (e.g., ForcedAligner) inside the backend class

No changes to `factory.py`, `base.py`, or `pipeline.py` needed.

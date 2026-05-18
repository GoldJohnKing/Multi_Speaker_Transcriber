"""Pipeline orchestrator — serial dispatch of transcription stages."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as TA_F
from rich.console import Console

from transcribe.config import load_config, resolve_device
from transcribe.data.types import AudioSegment, PipelineConfig, TranscriptSegment
from transcribe.models.audio_extractor import AudioExtractor
from transcribe.models.asr import ASRTranscriber
from transcribe.models.denoiser import Denoiser
from transcribe.models.srt_writer import SrtWriter

console = Console()

# ASR models expect 16 kHz input
_ASR_SAMPLE_RATE = 16_000


def _default_output_path(input_path: str) -> str:
    return str(Path(input_path).with_suffix(".srt"))


def _resample(audio: AudioSegment, target_sr: int) -> AudioSegment:
    """Resample audio to target sample rate."""
    if audio.sample_rate == target_sr:
        return audio
    wav_t = torch.from_numpy(audio.waveform).unsqueeze(0)  # [1, T]
    wav_t = TA_F.resample(wav_t, audio.sample_rate, target_sr)
    waveform = np.ascontiguousarray(wav_t.squeeze(0).numpy(), dtype=np.float32)
    return AudioSegment(
        waveform=waveform,
        sample_rate=target_sr,
        start_time=audio.start_time,
        end_time=audio.end_time,
    )


def run_pipeline(
    input_path: str,
    output_path: str | None = None,
    config: PipelineConfig | None = None,
    verbose: bool = False,
) -> str:
    """Run the transcription pipeline.

    Args:
        input_path: Path to input video/audio file.
        output_path: Path to output SRT file.
        config: Pipeline configuration.
        verbose: Print detailed progress.

    Returns:
        Path to the output SRT file.
    """
    if config is None:
        config = PipelineConfig()

    device = resolve_device(config.device)
    output = output_path or _default_output_path(input_path)
    total_start = time.time()

    # Determine total stages for progress display
    total_stages = 3 + (1 if config.denoise else 0)

    if verbose:
        console.print(f"[bold]设备:[/bold] {device}")
        console.print(f"[bold]输入:[/bold] {input_path}")
        console.print()

    # Stage 1: Audio extraction
    # Use 48kHz when denoising (DeepFilterNet's native rate), 16kHz otherwise
    extract_sr = 48_000 if config.denoise else _ASR_SAMPLE_RATE
    step = 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 提取音频 ({extract_sr}Hz) ...", end=" ")
    extractor = AudioExtractor()
    audio = extractor.extract(input_path, sample_rate=extract_sr)
    if verbose:
        console.print(f"完成 ({time.time() - step_start:.1f}s)")

    # Stage 2: Noise suppression (optional)
    if config.denoise:
        step += 1
        step_start = time.time()
        if verbose:
            console.print(f"[{step}/{total_stages}] 噪声抑制 ...", end=" ")
        denoiser = Denoiser(device=device)
        audio = denoiser.process(audio)
        denoiser.cleanup()
        if verbose:
            console.print(f"完成 ({time.time() - step_start:.1f}s)")

    # Resample to ASR sample rate if needed
    if audio.sample_rate != _ASR_SAMPLE_RATE:
        audio = _resample(audio, _ASR_SAMPLE_RATE)

    # Stage 3: ASR
    step += 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 语音转文字 ...", end=" ")
    transcriber = ASRTranscriber(device=device, hotword_path=config.hotwords)
    segments = transcriber.transcribe(audio)
    if verbose:
        console.print(
            f"识别 {len(segments)} 个片段 ... 完成 ({time.time() - step_start:.1f}s)"
        )

    # Stage 4: SRT generation
    step += 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 生成 SRT ...", end=" ")
    writer = SrtWriter(speaker_label=True)
    writer.write(segments, output)
    if verbose:
        console.print(f"输出 {len(segments)} 条字幕 ... 完成 ({time.time() - step_start:.1f}s)")

    if verbose:
        elapsed = time.time() - total_start
        console.print(f"{'─' * 40}")
        mins, secs = divmod(int(elapsed), 60)
        console.print(
            f"[bold]总耗时:[/bold] {mins}m {secs}s | [bold]输出:[/bold] {output} ({len(segments)} 条字幕)"
        )

    return output

"""Pipeline orchestrator — ASR-first with post-hoc speaker attribution."""
from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from transcribe.config import load_config, resolve_device
from transcribe.data.types import (
    DiarizationResult,
    PipelineConfig,
)
from transcribe.models.audio_extractor import AudioExtractor
from transcribe.models.asr import create_asr
from transcribe.models.attribution import AttributionEngine
from transcribe.models.attribution.strategy import TimestampStrategy
from transcribe.models.diarizer import Diarizer
from transcribe.models.matcher import SpeakerMatcher
from transcribe.models.segmentation import SubtitleSegmenter
from transcribe.models.srt_writer import SrtWriter

console = Console()

_ASR_SAMPLE_RATE = 16_000


def _default_output_path(input_path: str) -> str:
    return str(Path(input_path).with_suffix(".srt"))


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

    total_stages = 4 + (1 if config.diarize else 0)
    step = 0

    if verbose:
        console.print(f"[bold]设备:[/bold] {device}")
        console.print(f"[bold]输入:[/bold] {input_path}")
        if config.speaker_references:
            console.print(f"[bold]声样目录:[/bold] {config.speaker_references}")
        console.print()

    # ── Stage 1: Audio extraction ───────────────────────────────────
    step += 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 提取音频 ({_ASR_SAMPLE_RATE}Hz) ...", end=" ")
    extractor = AudioExtractor()
    audio = extractor.extract(input_path, sample_rate=_ASR_SAMPLE_RATE)
    if verbose:
        console.print(f"完成 ({time.time() - step_start:.1f}s)")

    # ── Stage 2: ASR on full audio ──────────────────────────────────
    step += 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 语音转文字（全音频）...", end=" ")
    transcriber = create_asr(config.backend, device=device, hotword_path=config.hotwords)

    if transcriber.provides_segments:
        # Backend produces subtitle-ready segments directly
        all_segments = transcriber.transcribe(audio)
        transcriber.cleanup()
        if verbose:
            console.print(f"识别 {len(all_segments)} 个段落 ... 完成 ({time.time() - step_start:.1f}s)")
    else:
        # Backend returns word-level timestamps; pipeline handles segmentation
        words = transcriber.transcribe_words(audio)
        transcriber.cleanup()

        # Validate word-level timestamp quality for diarization pipeline
        if config.diarize and words:
            zero_dur_count = sum(1 for w in words if abs(w.end_time - w.start_time) < 1e-6)
            zero_ratio = zero_dur_count / len(words) if words else 0
            if zero_ratio > 0.8:
                if verbose:
                    console.print(
                        f"\n[bold yellow]警告: {zero_dur_count}/{len(words)} 个词的时长为零，"
                        "词级时间戳可能无效。重叠区域的说话人归因将严重受限。[/bold yellow]"
                    )
            elif verbose:
                console.print(f"识别 {len(words)} 个词 ... 完成 ({time.time() - step_start:.1f}s)")
        elif config.diarize and not words:
            if verbose:
                console.print(
                    "\n[bold yellow]警告: ASR 未产出任何词级时间戳，"
                    "重叠处理和说话人归因将不可用。[/bold yellow]"
                )
        elif verbose:
            console.print(f"识别 {len(words)} 个词 ... 完成 ({time.time() - step_start:.1f}s)")

    # ── Stage 3: Subtitle segmentation ──────────────────────────────
    step += 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 字幕分割 ...", end=" ")
    if not transcriber.provides_segments:
        segmenter = SubtitleSegmenter()
        all_segments = segmenter.segment(words)
    # else: all_segments already populated in Stage 2
    if verbose:
        console.print(f"{len(all_segments)} 条字幕 ... 完成 ({time.time() - step_start:.1f}s)")

    # ── Stage 4: Speaker diarization + attribution ──────────────────
    diarization: DiarizationResult | None = None
    if config.diarize:
        step += 1
        step_start = time.time()
        if verbose:
            console.print(f"[{step}/{total_stages}] 说话人识别 + 归因 ...", end=" ")

        diarizer = Diarizer(device=device, num_speakers=config.num_speakers)
        diarization = diarizer.process(audio)
        diarizer.cleanup()

        if transcriber.provides_segments:
            # Simple dominant-speaker voting only — no turn splitting,
            # no overlap handling. Preserves ASR's native segmentation.
            strategy = TimestampStrategy()
            all_segments = strategy.attribute(all_segments, diarization)
        else:
            engine = AttributionEngine()
            all_segments = engine.run(all_segments, diarization)

        if verbose:
            console.print(
                f"检测到 {diarization.num_speakers} 位说话人, "
                f"{len(diarization.overlap_regions)} 个重叠区域 ... "
                f"完成 ({time.time() - step_start:.1f}s)"
            )

    # ── Stage 5: Speaker reference matching ─────────────────────────
    speaker_name_map: dict[str, str] = {}
    if config.speaker_references and not config.diarize:
        if verbose:
            console.print(
                "[bold yellow]警告: --speaker-ref 需要 --diarize，"
                "声纹匹配将被跳过[/bold yellow]"
            )
    if config.speaker_references and diarization:
        ref_matcher = SpeakerMatcher(device=device)
        try:
            ref_matcher.register_speakers(config.speaker_references)
            speaker_name_map = ref_matcher.match_speakers_to_references(
                audio, diarization
            )
            if verbose and speaker_name_map:
                console.print(
                    "  说话人匹配: "
                    + ", ".join(f"{sid} → {name}" for sid, name in speaker_name_map.items())
                )
        except (FileNotFoundError, ValueError) as e:
            if verbose:
                console.print(f"[bold yellow]警告: 说话人声样加载失败: {e}[/bold yellow]")
        finally:
            ref_matcher.cleanup()

    # ── Stage 6: SRT generation ─────────────────────────────────────
    step += 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 生成 SRT ...", end=" ")
    writer = SrtWriter(speaker_label=config.diarize)
    writer.write(all_segments, output, speaker_name_map=speaker_name_map)
    if verbose:
        console.print(f"输出 {len(all_segments)} 条字幕 ... 完成 ({time.time() - step_start:.1f}s)")

    if verbose:
        elapsed = time.time() - total_start
        console.print(f"{'─' * 40}")
        mins, secs = divmod(int(elapsed), 60)
        console.print(
            f"[bold]总耗时:[/bold] {mins}m {secs}s | [bold]输出:[/bold] {output} ({len(all_segments)} 条字幕)"
        )

    return output

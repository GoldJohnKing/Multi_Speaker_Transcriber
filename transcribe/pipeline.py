"""Pipeline orchestrator — ASR-first with post-hoc speaker attribution."""
from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from transcribe.config import load_config, resolve_device
from transcribe.data.types import (
    AudioSegment,
    DiarizationResult,
    PipelineConfig,
    TranscriptSegment,
    WordTimestamp,
)
from transcribe.models.audio_extractor import AudioExtractor
from transcribe.models.asr import create_asr
from transcribe.models.attribution import AttributionEngine
from transcribe.models.diarizer import Diarizer
from transcribe.models.matcher import SpeakerMatcher
from transcribe.models.srt_writer import SrtWriter

console = Console()

_ASR_SAMPLE_RATE = 16_000


def _default_output_path(input_path: str) -> str:
    return str(Path(input_path).with_suffix(".srt"))


def _words_to_segments(
    words: list[WordTimestamp],
    *,
    max_gap: float = 1.0,
    max_duration: float = 7.0,
) -> list[TranscriptSegment]:
    """Convert word timestamps to subtitle segments (no-diarize mode)."""
    if not words:
        return []

    segments: list[TranscriptSegment] = []
    buf = [words[0]]

    def _flush(b: list[WordTimestamp]) -> TranscriptSegment:
        return TranscriptSegment(
            speaker_id="SPEAKER_00",
            start_time=b[0].start_time,
            end_time=b[-1].end_time,
            text="".join(w.word for w in b),
        )

    for w in words[1:]:
        buf_dur = buf[-1].end_time - buf[0].start_time
        if w.start_time - buf[-1].end_time > max_gap or buf_dur >= max_duration:
            segments.append(_flush(buf))
            buf = [w]
        else:
            buf.append(w)

    if buf:
        segments.append(_flush(buf))

    return segments


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

    total_stages = 3 + (1 if config.diarize else 0)
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
    words = transcriber.transcribe_words(audio)
    transcriber.cleanup()
    if verbose:
        console.print(f"识别 {len(words)} 个词 ... 完成 ({time.time() - step_start:.1f}s)")

    # ── Stage 3: Speaker diarization ────────────────────────────────
    diarization: DiarizationResult | None = None
    overlap_regions: list[tuple[float, float]] = []
    if config.diarize:
        step += 1
        step_start = time.time()
        if verbose:
            console.print(f"[{step}/{total_stages}] 说话人识别 ...", end=" ")
        diarizer = Diarizer(device=device, num_speakers=config.num_speakers)
        diarization = diarizer.process(audio)
        overlap_regions = diarization.overlap_regions
        diarizer.cleanup()
        if verbose:
            console.print(
                f"检测到 {diarization.num_speakers} 位说话人, "
                f"{len(overlap_regions)} 个重叠区域 ... "
                f"完成 ({time.time() - step_start:.1f}s)"
            )

    # ── Stage 4: Speaker attribution ────────────────────────────────
    if diarization is not None:
        engine = AttributionEngine()
        all_segments = engine.run(words, diarization, overlap_regions)
    else:
        all_segments = _words_to_segments(words)

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

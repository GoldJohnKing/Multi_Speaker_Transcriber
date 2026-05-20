"""Pipeline orchestrator — serial dispatch of transcription stages."""

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
)
from transcribe.models.audio_extractor import AudioExtractor
from transcribe.models.asr import ASRTranscriber
from transcribe.models.diarizer import Diarizer
from transcribe.models.srt_writer import SrtWriter

console = Console()

# All stages operate at 16kHz
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

    # Determine total stages for progress display
    # Stages: extract, (diarize), asr, srt
    total_stages = (
        3  # extract + asr + srt
        + (1 if config.diarize else 0)
    )

    if verbose:
        console.print(f"[bold]设备:[/bold] {device}")
        console.print(f"[bold]输入:[/bold] {input_path}")
        if config.speaker_references:
            console.print(f"[bold]声样目录:[/bold] {config.speaker_references}")
        console.print()

    # ── Stage 1: Audio extraction ───────────────────────────────────────
    extract_sr = _ASR_SAMPLE_RATE
    step = 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 提取音频 ({extract_sr}Hz) ...", end=" ")
    extractor = AudioExtractor()
    audio = extractor.extract(input_path, sample_rate=extract_sr)
    if verbose:
        console.print(f"完成 ({time.time() - step_start:.1f}s)")

    # ── Stage 2: Speaker diarization ───────────────────────────────────
    diarization: DiarizationResult | None = None
    if config.diarize:
        step += 1
        step_start = time.time()
        if verbose:
            console.print(f"[{step}/{total_stages}] 说话人识别 ...", end=" ")
        diarizer = Diarizer(device=device, num_speakers=config.num_speakers)
        diarization = diarizer.process(audio)
        diarizer.cleanup()
        if verbose:
            console.print(
                f"检测到 {diarization.num_speakers} 位说话人, "
                f"{len(diarization.overlap_regions)} 个重叠区域 ... "
                f"完成 ({time.time() - step_start:.1f}s)"
            )

    # ── Speaker reference matching (optional) ──────────────────────────
    speaker_name_map: dict[str, str] = {}
    if config.speaker_references and not config.diarize:
        console.print(
            "[bold yellow]警告: --speaker-ref 需要 --diarize（默认启用），"
            "当前 --no-diarize 已关闭说话人识别，声纹匹配将被跳过[/bold yellow]"
        )
    if config.speaker_references and diarization:
        from transcribe.models.matcher import SpeakerMatcher

        ref_matcher = SpeakerMatcher(device=device)
        try:
            ref_matcher.register_speakers(config.speaker_references)
            speaker_name_map = ref_matcher.match_speakers_to_references(
                audio, diarization
            )
            if verbose and speaker_name_map:
                console.print(
                    "  说话人匹配: "
                    + ", ".join(
                        f"{sid} → {name}"
                        for sid, name in speaker_name_map.items()
                    )
                )
        except (FileNotFoundError, ValueError) as e:
            if verbose:
                console.print(
                    f"[bold yellow]警告: 说话人声样加载失败: {e}[/bold yellow]"
                )
        finally:
            ref_matcher.cleanup()

    # ── Stage 3: ASR ───────────────────────────────────────────────────
    step += 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 语音转文字 ...", end=" ")
    transcriber = ASRTranscriber(device=device, hotword_path=config.hotwords)
    all_segments: list[TranscriptSegment] = []
    _min_samples = int(0.3 * audio.sample_rate)

    if diarization is None:
        # --no-diarize mode: transcribe full audio as single speaker
        transcripts = transcriber.transcribe(audio)
        for t in transcripts:
            all_segments.append(
                TranscriptSegment(
                    speaker_id="SPEAKER_00",
                    start_time=t.start_time,
                    end_time=t.end_time,
                    text=t.text,
                )
            )
    else:
        # Default mode: send every diarization segment directly to ASR
        for spk_seg in diarization.segments:
            start_sample = int(
                (spk_seg.start_time - audio.start_time) * audio.sample_rate
            )
            end_sample = int(
                (spk_seg.end_time - audio.start_time) * audio.sample_rate
            )
            start_sample = max(0, start_sample)
            end_sample = min(len(audio.waveform), end_sample)
            if end_sample - start_sample < _min_samples:
                continue

            segment_audio = AudioSegment(
                waveform=audio.waveform[start_sample:end_sample],
                sample_rate=audio.sample_rate,
                start_time=spk_seg.start_time,
                end_time=spk_seg.end_time,
            )

            transcripts = transcriber.transcribe(segment_audio)
            for t in transcripts:
                all_segments.append(
                    TranscriptSegment(
                        speaker_id=spk_seg.speaker_id,
                        start_time=t.start_time,
                        end_time=t.end_time,
                        text=t.text,
                    )
                )

    transcriber.cleanup()

    if verbose:
        console.print(
            f"识别 {len(all_segments)} 个片段 ... 完成 ({time.time() - step_start:.1f}s)"
        )

    # ── Stage 4: SRT generation ────────────────────────────────────────
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

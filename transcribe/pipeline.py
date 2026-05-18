"""Pipeline orchestrator — serial dispatch of transcription stages."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as TA_F
from rich.console import Console

from transcribe.config import load_config, resolve_device
from transcribe.data.types import (
    AudioSegment,
    DiarizationResult,
    PipelineConfig,
    SpeakerSegment,
    TranscriptSegment,
)
from transcribe.models.audio_extractor import AudioExtractor
from transcribe.models.asr import ASRTranscriber
from transcribe.models.denoiser import DEFAULT_SNR_THRESHOLD, Denoiser, estimate_snr
from transcribe.models.diarizer import Diarizer
from transcribe.models.separator import Separator
from transcribe.models.srt_writer import SrtWriter

console = Console()

# ASR models expect 16 kHz input
_ASR_SAMPLE_RATE = 16_000


def _default_output_path(input_path: str) -> str:
    return str(Path(input_path).with_suffix(".srt"))


def _non_overlap_ranges(
    seg_start: float,
    seg_end: float,
    overlap_regions: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Compute the portions of [seg_start, seg_end] not covered by any overlap region."""
    ranges = [(seg_start, seg_end)]
    for o_start, o_end in overlap_regions:
        new_ranges: list[tuple[float, float]] = []
        for r_start, r_end in ranges:
            if o_start < r_end and o_end > r_start:
                # Overlap intersects this range — split around it
                if r_start < o_start:
                    new_ranges.append((r_start, o_start))
                if o_end < r_end:
                    new_ranges.append((o_end, r_end))
            else:
                new_ranges.append((r_start, r_end))
        ranges = new_ranges
    return ranges


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
    # Stages: extract, (denoise), (diarize), (separate), asr, srt
    total_stages = (
        2  # extract + srt
        + (1 if config.denoise else 0)
        + (1 if config.diarize else 0)
        + (1 if config.separate and config.diarize else 0)
        + 1  # asr
    )

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

    # Stage 2: Noise suppression (optional, SNR-gated)
    if config.denoise:
        step += 1
        step_start = time.time()
        snr = estimate_snr(audio)
        if snr >= DEFAULT_SNR_THRESHOLD:
            if verbose:
                console.print(
                    f"[{step}/{total_stages}] 噪声抑制 ... "
                    f"跳过 (SNR={snr:.1f}dB >= {DEFAULT_SNR_THRESHOLD:.0f}dB，音频较干净)"
                )
        else:
            if verbose:
                console.print(
                    f"[{step}/{total_stages}] 噪声抑制 ... "
                    f"SNR={snr:.1f}dB < {DEFAULT_SNR_THRESHOLD:.0f}dB，需要降噪 ...",
                    end=" ",
                )
            denoiser = Denoiser(device=device)
            audio = denoiser.process(audio)
            denoiser.cleanup()
            if verbose:
                console.print(f"完成 ({time.time() - step_start:.1f}s)")

    # Resample to ASR sample rate if needed
    if audio.sample_rate != _ASR_SAMPLE_RATE:
        audio = _resample(audio, _ASR_SAMPLE_RATE)

    # Stage 3: Speaker diarization (default on, disable with --no-diarize)
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

    # Stage 4: Speech separation (optional, only with --separate + diarization)
    overlap_separated: dict[tuple[float, float], list[AudioSegment]] = {}
    if config.separate and diarization and diarization.overlap_regions:
        step += 1
        step_start = time.time()
        if verbose:
            console.print(
                f"[{step}/{total_stages}] 重叠语音分离 ... "
                f"处理 {len(diarization.overlap_regions)} 个重叠片段 ...",
                end=" ",
            )
        separator = Separator(device=device)
        separated_audios = separator.separate_overlaps(audio, diarization)

        # Group separated audio by overlap region (2 sources per region)
        sep_idx = 0
        for overlap_start, overlap_end in diarization.overlap_regions:
            region_segments = []
            for _ in range(2):
                if sep_idx < len(separated_audios):
                    region_segments.append(separated_audios[sep_idx])
                    sep_idx += 1
            overlap_separated[(overlap_start, overlap_end)] = region_segments

        separator.cleanup()
        if verbose:
            console.print(f"完成 ({time.time() - step_start:.1f}s)")

    # Stage 5: ASR
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
    elif overlap_separated:
        # --separate mode: overlap regions via separated tracks,
        # non-overlap portions from original audio
        for o_start, o_end in diarization.overlap_regions:
            if (o_start, o_end) not in overlap_separated:
                continue
            separated_tracks = overlap_separated[(o_start, o_end)]
            unique_speakers: dict[str, SpeakerSegment] = {}
            for s in diarization.segments:
                if s.start_time < o_end and s.end_time > o_start:
                    if s.speaker_id not in unique_speakers:
                        unique_speakers[s.speaker_id] = s
            speaker_ids = list(unique_speakers.keys())
            for idx, spk_id in enumerate(speaker_ids):
                if idx < len(separated_tracks):
                    track = separated_tracks[idx]
                    if len(track.waveform) < _min_samples:
                        continue
                    transcripts = transcriber.transcribe(track)
                    for t in transcripts:
                        all_segments.append(
                            TranscriptSegment(
                                speaker_id=spk_id,
                                start_time=t.start_time,
                                end_time=t.end_time,
                                text=t.text,
                            )
                        )
        # Non-overlap portions from original audio
        for spk_seg in diarization.segments:
            non_overlap = _non_overlap_ranges(
                spk_seg.start_time, spk_seg.end_time, diarization.overlap_regions
            )
            for rng_start, rng_end in non_overlap:
                start_sample = int(
                    (rng_start - audio.start_time) * audio.sample_rate
                )
                end_sample = int(
                    (rng_end - audio.start_time) * audio.sample_rate
                )
                start_sample = max(0, start_sample)
                end_sample = min(len(audio.waveform), end_sample)
                if end_sample - start_sample < _min_samples:
                    continue
                segment_audio = AudioSegment(
                    waveform=audio.waveform[start_sample:end_sample],
                    sample_rate=audio.sample_rate,
                    start_time=rng_start,
                    end_time=rng_end,
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

    # Stage 6: SRT generation
    step += 1
    step_start = time.time()
    if verbose:
        console.print(f"[{step}/{total_stages}] 生成 SRT ...", end=" ")
    writer = SrtWriter(speaker_label=config.diarize)
    writer.write(all_segments, output)
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

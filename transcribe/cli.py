"""CLI argument parsing for the transcription pipeline."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Offline multi-speaker audio transcription for Chinese Mandarin",
    )
    parser.add_argument("input", help="Input video or audio file path")
    parser.add_argument(
        "-o", "--output", help="Output SRT file path (default: input_name.srt)"
    )
    parser.add_argument(
        "--hotwords", help="Hotword file path (one word per line)"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Known number of speakers (auto-detect if omitted)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization (all audio attributed to one speaker)",
    )
    parser.add_argument(
        "--speaker-ref",
        metavar="DIR",
        help="Directory of speaker audio samples (filename without extension = speaker name)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Compute device",
    )
    parser.add_argument(
        "--backend",
        choices=["Fun-ASR-Paraformer", "Fun-ASR-Nano", "Qwen3-ASR"],
        default="Fun-ASR-Nano",
        help="ASR backend (default: Fun-ASR-Nano)",
    )
    parser.add_argument(
        "--cache-dir", default=".cache", help="Intermediate cache directory"
    )
    parser.add_argument("--config", help="YAML config file path")
    parser.add_argument(
        "--keep-cache", action="store_true", help="Keep intermediate artifacts"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)

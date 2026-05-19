"""CLI argument parsing for the transcription pipeline."""

from __future__ import annotations

import argparse
import sys


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
        "--denoise", action="store_true", help="Enable noise suppression"
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization (all audio attributed to one speaker)",
    )
    parser.add_argument(
        "--separate", action="store_true", help="Enable overlap speech separation"
    )
    parser.add_argument(
        "--tse",
        action="store_true",
        help="Enable target speaker extraction using video face tracking (requires video input, mutually exclusive with --separate)",
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
    args = build_parser().parse_args(argv)

    # Mutual exclusion: --tse and --separate
    if args.tse and args.separate:
        print("错误: --tse 和 --separate 互斥，请选择其一", file=sys.stderr)
        sys.exit(1)

    return args

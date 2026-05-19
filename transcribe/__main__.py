"""CLI entry point for the transcription pipeline."""

from transcribe.cli import parse_args
from transcribe.config import load_config
from transcribe.pipeline import run_pipeline


def main() -> None:
    args = parse_args()
    config = load_config(
        config_path=args.config,
        cli_overrides={
            "device": args.device,
            "denoise": args.denoise,
            "diarize": not args.no_diarize,
            "separate": args.separate,
            "tse": args.tse,
            "hotwords": args.hotwords,
            "num_speakers": args.num_speakers,
            "cache_dir": args.cache_dir,
            "speaker_references": args.speaker_ref,
        },
    )
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        config=config,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

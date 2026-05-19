"""Stage 4A: Target Speaker Extraction using ClearVoice AV_MossFormer2_TSE_16K.

Extracts individual speaker audio from overlapping speech using video face
tracking as visual cues. Requires video input (not pure audio).

ClearVoice TSE operates in IO mode: it takes a video file path, runs face
detection + tracking internally, and writes per-face-track extracted audio
to an output directory. We then read those files back as AudioSegments.
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from transcribe.data.types import AudioSegment, DiarizationResult


class TargetSpeakerExtractor:
    """Target speaker extraction using ClearVoice TSE.

    Uses AV_MossFormer2_TSE_16K to extract individual speakers from mixed
    audio by leveraging visual face tracking from the input video.
    """

    def __init__(
        self,
        device: str = "cpu",
        model: str = "AV_MossFormer2_TSE_16K",
        min_track_frames: int = 50,
    ) -> None:
        self._device = device
        self._model_name = model
        self._min_track_frames = min_track_frames
        self._model = self._load_model()

    @staticmethod
    def _patch_tse_cuda_bug() -> None:
        """Patch ClearVoice TSE hardcoded .cuda() for CPU compatibility.

        ClearVoice's av_mossformer2.py has a hardcoded .cuda() call in
        the `overlap_and_add` function (line 177). This crashes on systems
        without NVIDIA GPU. We monkey-patch it to use the signal's device.
        """
        try:
            import clearvoice.models.av_mossformer2_tse.av_mossformer2 as tse_mod

            def _overlap_and_add_cpu_safe(signal, frame_step):
                """CPU-safe replacement that uses signal.device instead of .cuda()."""
                import math

                outer_dimensions = signal.size()[:-2]
                frames, frame_length = signal.size()[-2:]

                subframe_length = math.gcd(frame_length, frame_step)
                subframe_step = frame_step // subframe_length
                subframes_per_frame = frame_length // subframe_length
                output_size = frame_step * (frames - 1) + frame_length
                output_subframes = output_size // subframe_length

                subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

                frame = torch.arange(0, output_subframes).unfold(
                    0, subframes_per_frame, subframe_step
                )
                # Use signal's device instead of hardcoded .cuda()
                frame = signal.new_tensor(frame).long().to(signal.device)
                frame = frame.contiguous().view(-1)

                result = signal.new_zeros(
                    *outer_dimensions, output_subframes, subframe_length
                )
                result.index_add_(-2, frame, subframe_signal)
                result = result.view(*outer_dimensions, -1)
                return result

            tse_mod.overlap_and_add = _overlap_and_add_cpu_safe
        except ImportError:
            pass

    @staticmethod
    def _patch_tse_visualization() -> None:
        """Replace ClearVoice TSE visualization with audio-only write.

        The original `visualization` function writes extracted audio as WAV
        files then does expensive per-track video encoding (libx264 1080p).
        Since we only need the audio, we replace it with a version that
        writes WAV files and skips all video processing.
        """
        try:
            import clearvoice.utils.video_process as vp_mod

            def _visualization_audio_only(tracks, est_sources, video_args):
                """Write extracted audio as WAV files, skip video encoding."""
                for idx, audio in enumerate(est_sources):
                    max_value = np.max(np.abs(audio))
                    if max_value > 1:
                        audio /= max_value
                    sf.write(
                        video_args.pycropPath + "/est_%s.wav" % idx, audio, 16000
                    )

            vp_mod.visualization = _visualization_audio_only
        except ImportError:
            pass

    def _load_model(self):
        """Load ClearVoice TSE model."""
        from clearvoice import ClearVoice

        from transcribe.models.rocm_compat import patch_clearvoice_for_rocm

        patch_clearvoice_for_rocm()
        self._patch_tse_cuda_bug()
        self._patch_tse_visualization()
        cv = ClearVoice(task="target_speaker_extraction", model_names=[self._model_name])
        return cv

    def extract(
        self,
        video_path: str,
        audio: AudioSegment,
        diarization: DiarizationResult,
    ) -> dict[str, AudioSegment]:
        """Extract each speaker's audio from video using face tracking.

        ClearVoice TSE requires IO mode (online_write=True): it reads the
        video file, performs face detection/tracking, and writes extracted
        per-face-track audio as WAV files to an output directory.

        Args:
            video_path: Path to the input video file.
            audio: Denoised full audio segment (16kHz), used for sample rate.
            diarization: Speaker diarization result.

        Returns:
            {face_track_id: AudioSegment} mapping. Each face track's
            extracted speaker audio. face_track_id is a string like "face_0".
        """
        # TSE requires online_write=True and operates on the video file directly.
        # It creates a temp directory structure:
        #   output_path/AV_MossFormer2_TSE_16K/<video_name>/py_faceTracks/est_*.wav
        tmp_dir = tempfile.mkdtemp(prefix="tse_")

        # ClearVoice TSE internally calls argparse.ArgumentParser().parse_args()
        # which reads sys.argv and crashes when our CLI args are present.
        # Temporarily replace sys.argv to avoid the conflict.
        original_argv = sys.argv
        sys.argv = [original_argv[0]] if original_argv else []
        try:
            self._model(video_path, online_write=True, output_path=tmp_dir)
        except Exception as e:
            raise RuntimeError(
                f"TSE extraction failed. Ensure video file contains visible faces: {e}"
            ) from e
        finally:
            sys.argv = original_argv

        # Find the extracted audio files written by TSE
        face_tracks: dict[str, AudioSegment] = {}

        # TSE writes to: <tmp_dir>/<model_name>/<video_stem>/py_faceTracks/est_<N>.wav
        # Also check the final output: <tmp_dir>/<model_name>/<video_stem>/py_faceTracks/est_<N>.wav
        video_stem = Path(video_path).stem
        search_patterns = [
            os.path.join(tmp_dir, "**", "est_*.wav"),
        ]

        wav_files: list[str] = []
        for pattern in search_patterns:
            wav_files.extend(glob.glob(pattern, recursive=True))
        wav_files.sort()

        if not wav_files:
            # No faces detected — return empty dict (pipeline will fall back)
            return face_tracks

        for idx, wav_file in enumerate(wav_files):
            waveform, sr = sf.read(wav_file, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)

            # Derive face track ID from filename (e.g. est_0.wav -> face_0)
            basename = os.path.basename(wav_file)
            stem = os.path.splitext(basename)[0]  # "est_0"
            track_id = stem.replace("est_", "face_")

            face_tracks[track_id] = AudioSegment(
                waveform=waveform,
                sample_rate=audio.sample_rate,
                start_time=audio.start_time,
                end_time=audio.end_time,
            )

        return face_tracks

    def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

"""ClearVoice ROCm compatibility patch.

ClearVoice's SpeechModel.__init__ uses nvidia-smi to detect GPU availability.
On AMD ROCm, nvidia-smi doesn't exist, causing models to silently fall back to CPU.

This patch overrides the device detection to use torch.cuda.is_available() instead.
"""

from __future__ import annotations

import torch


def patch_clearvoice_for_rocm() -> None:
    """Apply ROCm compatibility patch for ClearVoice.

    Safe to call multiple times (idempotent). No-op if ClearVoice is not installed.
    """
    try:
        from clearvoice.clearvoice.networks import SpeechModel
    except ImportError:
        return

    if getattr(SpeechModel, "_rocm_patched", False):
        return

    _original_init = SpeechModel.__init__

    def _patched_init(self, args) -> None:
        _original_init(self, args)
        if torch.cuda.is_available() and self.device.type == "cpu":
            import torch as _torch
            args.use_cuda = 1
            self.device = _torch.device("cuda")

    SpeechModel.__init__ = _patched_init
    SpeechModel._rocm_patched = True

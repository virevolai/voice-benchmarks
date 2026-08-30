"""Shared audio constants and fixture loading.

Every harness streams the same PCM format so a fixture recorded once drives
all vendors. Vendors that reject 16 kHz upsample at their own edge; the
fixtures on disk stay canonical.
"""

from __future__ import annotations

import wave
from pathlib import Path

INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
CHUNK_MS = 20
CHUNK_BYTES = INPUT_RATE * 2 * CHUNK_MS // 1_000


def read_pcm16_mono(path: Path) -> bytes:
    """Load a fixture, rejecting anything that is not canonical format.

    Strict on purpose: a silently resampled or stereo fixture produces
    plausible-looking numbers that are not comparable across vendors.
    """
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getframerate() != INPUT_RATE
        ):
            raise ValueError(f"expected 16 kHz mono PCM16 WAV: {path}")
        pcm = wav.readframes(wav.getnframes())
        if not pcm:
            raise ValueError(f"audio fixture contains no samples: {path}")
        return pcm


# Backwards-compatible alias: the harnesses were written against this name.
_read_pcm16_mono = read_pcm16_mono

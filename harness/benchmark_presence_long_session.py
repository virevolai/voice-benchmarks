"""Measure Presence latency on one socket over a long session.

Third line of the long-session benchmark. Mirrors the Gemini and OpenAI
harnesses: same fixtures, same turn structure, same latency boundary (final
input sample sent -> first response audio frame).

Cost is not measured per turn here, because there is nothing per-turn to
measure: capacity is billed by occupied time rather than by conversation
history, so a later turn costs no more than an earlier one. What this harness
establishes is the other half of that claim — that latency does not drift as
the session grows.

Set PRESENCE_WS_URL to a session websocket, or pass --url.

Usage:
    uv run python scripts/benchmark_presence_long_session.py --turns 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from pathlib import Path

import websockets

from audio import (
    CHUNK_BYTES,
    CHUNK_MS,
    INPUT_RATE,
    _read_pcm16_mono,
)

from presence_session import resolve_ws_url

# No endpoint is baked in. A session is created through the public API unless
# --url or $PRESENCE_WS_URL points at an existing one.
DEFAULT_URL = os.environ.get("PRESENCE_WS_URL", "")

def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


async def wait_ready(ws, timeout: float = 90.0) -> dict:
    """Block until the gateway reports the session is warm."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
        if isinstance(raw, bytes):
            continue
        event = json.loads(raw)
        if event.get("type") == "ready":
            return event
    raise RuntimeError("session never reported ready")


async def run_turn(ws, pcm: bytes, number: int) -> dict:
    """Stream one fixture, measure to first response audio, wait for idle.

    The receiver runs for the whole turn, including while audio is still being
    sent: the gateway can begin responding before the fixture finishes, and a
    receiver started afterwards would both miss those frames and read a stale
    `face:idle` heartbeat left over from the previous turn.
    """
    first_audio_at: float | None = None
    completed_at: float | None = None
    speech_end_at: float | None = None
    output_bytes = 0
    events_seen: list[str] = []

    async def send() -> None:
        nonlocal speech_end_at
        # Let the session settle before streaming; sending immediately on
        # `ready` can race session start-up.
        await asyncio.sleep(float(os.environ.get("BENCH_SETTLE_S", "0.5")))
        next_send = time.monotonic()
        for offset in range(0, len(pcm), CHUNK_BYTES):
            await ws.send(pcm[offset:offset + CHUNK_BYTES])
            next_send += CHUNK_MS / 1_000
            await asyncio.sleep(max(0, next_send - time.monotonic()))
        speech_end_at = time.monotonic()

        # Trailing silence so local VAD endpoints on its own, exactly as the
        # hosted runs let server VAD close the turn.
        silence = bytes(CHUNK_BYTES)
        for _ in range(400):
            await ws.send(silence)
            next_send += CHUNK_MS / 1_000
            await asyncio.sleep(max(0, next_send - time.monotonic()))

    async def receive() -> None:
        nonlocal first_audio_at, completed_at, output_bytes
        while completed_at is None:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                if raw:
                    first_audio_at = first_audio_at or time.monotonic()
                    output_bytes += len(raw)
                continue
            event = json.loads(raw)
            kind = event.get("type", "")
            if kind not in events_seen:
                events_seen.append(kind)
            if os.environ.get("BENCH_DEBUG"):
                print(f"    rx {kind} {event.get('state','')} audio={output_bytes}",
                      flush=True)
            # `expression` frames stream alongside audio and `face:idle`
            # repeats as a ~20s heartbeat, so completion is the first idle
            # that follows THIS turn's audio.
            if kind == "face" and event.get("state") == "idle" and first_audio_at:
                completed_at = time.monotonic()
                return

    sender = asyncio.create_task(send())
    receiver = asyncio.create_task(receive())
    try:
        await asyncio.wait_for(receiver, timeout=90)
    finally:
        if not sender.done():
            sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)

    if first_audio_at is None or completed_at is None or speech_end_at is None:
        raise RuntimeError(f"turn {number} did not complete; saw {events_seen}")

    detail = {
        "turn": number,
        "first_audio_ms": round((first_audio_at - speech_end_at) * 1_000),
        "complete_ms": round((completed_at - speech_end_at) * 1_000),
        "input_audio_ms": round(len(pcm) / 2 / INPUT_RATE * 1_000),
        "output_audio_ms": round(output_bytes / 2 / INPUT_RATE * 1_000),
        "events": events_seen,
    }
    print(json.dumps({"progress": detail}), flush=True)
    return detail


async def run(args) -> dict:
    first_pcm = _read_pcm16_mono(Path(args.first_audio))
    middle_pcm = _read_pcm16_mono(Path(args.middle_audio))
    final_pcm = _read_pcm16_mono(Path(args.final_audio))

    started = time.monotonic()
    turns: list[dict] = []

    url = await resolve_ws_url(args.url)
    async with websockets.connect(
        url, max_size=None, open_timeout=90, ping_interval=20
    ) as ws:
        ready = await wait_ready(ws)
        connected_ms = round((time.monotonic() - started) * 1_000)
        session_started = time.monotonic()

        turns.append(await run_turn(ws, first_pcm, 1))
        for number in range(2, args.turns):
            turns.append(await run_turn(ws, middle_pcm, number))
        turns.append(await run_turn(ws, final_pcm, len(turns) + 1))

        occupied_s = time.monotonic() - session_started

    latencies = [float(t["first_audio_ms"]) for t in turns]

    return {
        "model": "presence-local-stack",
        # Deliberately not the resolved URL: it carries a session-scoped
        # token, and results in this repo are published.
        "url": "<session websocket>",
        "comparability": {
            "latency_boundary": (
                "wall-clock instant the final speech sample was sent -> first "
                "response audio frame; identical definition to the Gemini and "
                "OpenAI runs"
            ),
            "input_rate_hz": INPUT_RATE,
            "turn_end_detection": "local Silero VAD + endpointing, vendor default",
            "cost_model": (
                "occupied session time, not conversation history. There is no "
                "per-turn token charge, so no per-turn cost is reported."
            ),
            "session_start_note": (
                "connected_ms includes session start-up. Turn 1 measures "
                "materially slower than a warm one; report both rather than "
                "quoting the warm number alone."
            ),
        },
        "connected_ms": connected_ms,
        "ready": ready,
        "turns_completed": len(turns),
        "occupied_seconds": round(occupied_s, 1),
        "cost_note": "billed by occupied session time; flat regardless of turn count",
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "first_five_mean": round(statistics.mean(latencies[:5]), 1),
            "last_five_mean": round(statistics.mean(latencies[-5:]), 1),
        },
        "turns": turns,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="session websocket; defaults to $PRESENCE_WS_URL")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--first-audio", default="/tmp/bohita-gemini-memory-first.wav")
    ap.add_argument("--middle-audio", default="/tmp/bohita-parakeet-smoke.wav")
    ap.add_argument("--final-audio", default="/tmp/bohita-gemini-memory-final.wav")
    ap.add_argument("--out", default="results/presence-long-session.json")
    args = ap.parse_args()

    result = asyncio.run(run(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "turns"}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Measure OpenAI Realtime latency and context billing on one socket.

Mirrors scripts/benchmark_gemini_live_long_session.py so the two are directly
comparable: same fixtures, same turn structure, same recorded fields, same
cost-guard behaviour.

The question this answers: Gemini Live reported **zero cached tokens on every
turn**, so its per-turn cost grew 17.3x across 40 turns. OpenAI documents
automatic prompt caching for the growing conversational prefix at roughly a
99% discount, but describes it as best-effort. Does the cache actually hit in
a live session, and what does that do to the cost curve?

Usage:
    make voice-openai-long-fixtures      # once; reuses the Gemini fixtures
    uv run python scripts/benchmark_openai_realtime_long_session.py \
        --turns 40 --max-estimated-usd 5

Requires OPENAI_API_KEY (repo .env is loaded automatically).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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

REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"

# gpt-realtime-2.1 list price, USD per million tokens, checked 2026-08-17.
# Cached input is the whole point of this benchmark: a ~99% discount on the
# repeated conversational prefix, which Gemini Live did not offer at all.
PRICING = {
    "gpt-realtime-2.1": {
        "text_in": 4.00,
        "text_in_cached": 0.40,
        "audio_in": 32.00,
        "audio_in_cached": 0.40,
        "text_out": 16.00,
        "audio_out": 64.00,
    },
    "gpt-realtime-2.1-mini": {
        "text_in": 1.00,
        "text_in_cached": 0.10,
        "audio_in": 10.00,
        "audio_in_cached": 0.10,
        "text_out": 4.00,
        "audio_out": 20.00,
    },
}

OUTPUT_RATE = 24_000  # Realtime returns pcm16 at 24 kHz.

# Realtime rejects input below 24 kHz, while the Gemini run and our own
# pipeline use 16 kHz. The fixtures stay identical in content; only the
# sample rate is lifted, so the comparison is like-for-like.
REALTIME_INPUT_RATE = 24_000
REALTIME_CHUNK_BYTES = CHUNK_BYTES * REALTIME_INPUT_RATE // INPUT_RATE


def upsample_to_24k(pcm16_16k: bytes) -> bytes:
    """Linear resample 16 kHz -> 24 kHz PCM16 mono."""
    import array

    src = array.array("h")
    src.frombytes(pcm16_16k)
    if not src:
        return b""
    ratio = INPUT_RATE / REALTIME_INPUT_RATE
    out = array.array("h")
    count = int(len(src) / ratio)
    for i in range(count):
        pos = i * ratio
        left = int(pos)
        right = min(left + 1, len(src) - 1)
        frac = pos - left
        out.append(int(src[left] * (1 - frac) + src[right] * frac))
    return out.tobytes()


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                name, _, value = line.partition("=")
                if name.strip() == "OPENAI_API_KEY":
                    key = value.strip().strip("'\"")
                    break
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return key


def split_usage(usage: dict | None) -> dict:
    """Normalise Realtime usage into the fields the cost model needs.

    Realtime reports input_token_details.cached_tokens plus a per-modality
    breakdown; the cached split is what makes or breaks the long-session
    story, so it is recorded explicitly rather than folded into a total.
    """
    if not usage:
        return {}
    inp = usage.get("input_token_details") or {}
    cached_detail = inp.get("cached_tokens_details") or {}
    out = usage.get("output_token_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "input_text": int(inp.get("text_tokens", 0)),
        "input_audio": int(inp.get("audio_tokens", 0)),
        "input_cached": int(inp.get("cached_tokens", 0)),
        "input_cached_text": int(cached_detail.get("text_tokens", 0)),
        "input_cached_audio": int(cached_detail.get("audio_tokens", 0)),
        "output_text": int(out.get("text_tokens", 0)),
        "output_audio": int(out.get("audio_tokens", 0)),
    }


def usage_cost_usd(usage: dict, model: str) -> float:
    """Price a turn. Cached tokens are billed at the cached rate and removed
    from the uncached buckets, which is where the compounding is avoided."""
    if not usage:
        return 0.0
    rates = PRICING.get(model) or PRICING["gpt-realtime-2.1"]

    cached_text = usage.get("input_cached_text", 0)
    cached_audio = usage.get("input_cached_audio", 0)
    # When only a total cached count is reported, attribute it to audio, which
    # is the expensive modality — the conservative choice for our own claim.
    if not cached_text and not cached_audio:
        cached_audio = usage.get("input_cached", 0)

    fresh_text = max(0, usage.get("input_text", 0) - cached_text)
    fresh_audio = max(0, usage.get("input_audio", 0) - cached_audio)

    total = (
        fresh_text * rates["text_in"]
        + cached_text * rates["text_in_cached"]
        + fresh_audio * rates["audio_in"]
        + cached_audio * rates["audio_in_cached"]
        + usage.get("output_text", 0) * rates["text_out"]
        + usage.get("output_audio", 0) * rates["audio_out"]
    ) / 1_000_000
    return total


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


async def run_turn(ws, pcm: bytes, number: int, model: str) -> dict:
    """One turn: stream the fixture, let server VAD close it, await audio."""
    speech_end_at: float | None = None
    first_audio_at: float | None = None
    completed_at: float | None = None
    output_bytes = 0
    output_text: list[str] = []
    input_text: list[str] = []
    usage_detail: dict = {}

    async def send() -> None:
        nonlocal speech_end_at
        next_send = time.monotonic()
        for offset in range(0, len(pcm), REALTIME_CHUNK_BYTES):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(
                    pcm[offset:offset + REALTIME_CHUNK_BYTES]
                ).decode(),
            }))
            next_send += CHUNK_MS / 1_000
            await asyncio.sleep(max(0, next_send - time.monotonic()))
        # Wall-clock instant the last speech sample was sent. Both benchmarks
        # measure from here, so the boundary is identical even though the
        # sample rates differ.
        speech_end_at = time.monotonic()

        # Trailing silence so server VAD detects end-of-speech on its own,
        # matching the Gemini run rather than committing the buffer manually.
        silence = base64.b64encode(bytes(REALTIME_CHUNK_BYTES)).decode()
        for _ in range(150):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append", "audio": silence,
            }))
            next_send += CHUNK_MS / 1_000
            await asyncio.sleep(max(0, next_send - time.monotonic()))

    async def receive() -> None:
        nonlocal first_audio_at, completed_at, output_bytes, usage_detail
        # recv() rather than `async for`: the websocket iterator is consumed
        # once per connection, so a per-turn `async for` leaves later turns
        # with an exhausted iterator.
        #
        # Events from the PREVIOUS turn can still be in flight when this turn
        # starts — notably a trailing response.done, which would otherwise end
        # this turn instantly with turn-1 usage. Bind to the response id
        # created after our audio, and ignore anything belonging to another.
        this_response: str | None = None
        while True:
            raw = await ws.recv()
            event = json.loads(raw)
            kind = event.get("type", "")

            if kind == "response.created":
                this_response = (event.get("response") or {}).get("id")
                continue
            if kind == "error":
                raise RuntimeError(f"realtime error: {event.get('error')}")

            # Drop stragglers addressed to an earlier response.
            event_response = event.get("response_id") or (
                (event.get("response") or {}).get("id")
                if isinstance(event.get("response"), dict) else None
            )
            if (
                this_response is not None
                and event_response is not None
                and event_response != this_response
            ):
                continue
            if this_response is None and kind.startswith("response."):
                continue

            if kind == "response.output_audio.delta" and event.get("delta"):
                first_audio_at = first_audio_at or time.monotonic()
                output_bytes += len(base64.b64decode(event["delta"]))
            elif kind == "response.output_audio_transcript.delta":
                output_text.append(event.get("delta", ""))
            elif kind == "conversation.item.input_audio_transcription.completed":
                input_text.append(event.get("transcript", ""))
            elif kind == "response.done":
                usage_detail = split_usage(
                    (event.get("response") or {}).get("usage")
                )
                completed_at = time.monotonic()
                return

    sender = asyncio.create_task(send())
    receiver = asyncio.create_task(receive())
    try:
        await asyncio.wait_for(receiver, timeout=60)
    finally:
        if not sender.done():
            sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)

    if speech_end_at is None or first_audio_at is None or completed_at is None:
        raise RuntimeError(f"turn {number} did not complete")

    detail = {
        "turn": number,
        "first_audio_ms": round((first_audio_at - speech_end_at) * 1_000),
        "complete_ms": round((completed_at - speech_end_at) * 1_000),
        "input_audio_ms": round(len(pcm) / 2 / REALTIME_INPUT_RATE * 1_000),
        "output_audio_ms": round(output_bytes / 2 / OUTPUT_RATE * 1_000),
        "input_transcript": "".join(input_text).strip(),
        "output_transcript": "".join(output_text).strip(),
        "usage": usage_detail,
        "cached_tokens": usage_detail.get("input_cached", 0),
        "estimated_usd": round(usage_cost_usd(usage_detail, model), 6),
    }
    print(json.dumps({"progress": detail}), flush=True)
    return detail


async def run(args) -> dict:
    first_pcm = upsample_to_24k(_read_pcm16_mono(Path(args.first_audio)))
    middle_pcm = upsample_to_24k(_read_pcm16_mono(Path(args.middle_audio)))
    final_pcm = upsample_to_24k(_read_pcm16_mono(Path(args.final_audio)))

    headers = {"Authorization": f"Bearer {_api_key()}"}
    started = time.monotonic()
    turns: list[dict] = []
    estimated_total = 0.0
    stop_reason = "requested_turns"

    async with websockets.connect(
        REALTIME_URL.format(model=args.model),
        additional_headers=headers,
        max_size=None,
    ) as ws:
        connected_ms = round((time.monotonic() - started) * 1_000)
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": (
                    "You are in a long-running voice conversation. Reply "
                    "naturally in no more than eight words. Do not ask "
                    "follow-up questions. Remember facts the user explicitly "
                    "asks you to remember."
                ),
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_INPUT_RATE},
                        "turn_detection": {"type": "server_vad"},
                        "transcription": {"model": "whisper-1"},
                    },
                    "output": {"voice": args.voice},
                },
            },
        }))

        turns.append(await run_turn(ws, first_pcm, 1, args.model))
        estimated_total += turns[-1]["estimated_usd"]

        # Reserve the final recall turn; stop before the guard rather than
        # discovering an expensive turn after it is billed.
        for number in range(2, args.turns):
            reserve = max(0.05, turns[-1]["estimated_usd"] * 2)
            if estimated_total + reserve >= args.max_estimated_usd:
                stop_reason = "cost_guard"
                break
            turns.append(await run_turn(ws, middle_pcm, number, args.model))
            estimated_total += turns[-1]["estimated_usd"]

        final_number = len(turns) + 1
        turns.append(await run_turn(ws, final_pcm, final_number, args.model))
        estimated_total += turns[-1]["estimated_usd"]

    latencies = [float(t["first_audio_ms"]) for t in turns]
    cached = [t["cached_tokens"] for t in turns]
    memory_answer = turns[-1]["output_transcript"]

    return {
        "model": args.model,
        # Envelope differences between vendors are the easiest way to publish
        # a wrong comparison. Recorded here so the writeup states them rather
        # than implying the runs are identical.
        "comparability": {
            "latency_boundary": (
                "wall-clock instant the final speech sample was sent -> first "
                "output audio byte received; identical definition to the "
                "Gemini run"
            ),
            "input_rate_hz": REALTIME_INPUT_RATE,
            "input_resampled_from_hz": INPUT_RATE,
            "resample_note": (
                "Realtime rejects input below 24 kHz. Fixture content is "
                "identical to the Gemini run; only the sample rate is lifted. "
                "Audio token counts are therefore NOT directly comparable to "
                "Gemini's, because tokenisation is rate-dependent. Compare "
                "cost curves and their shape, not absolute token counts."
            ),
            "turn_end_detection": "server_vad (vendor default), same as Gemini run",
            "endpointing_note": (
                "Each vendor closes the turn with its own VAD, so a slice of "
                "the measured latency is their endpointing decision rather "
                "than generation. This is the honest way to measure a hosted "
                "product as a user would experience it, but it means the "
                "number is not purely time-to-first-token."
            ),
            "cost_source": (
                "token counts are vendor-reported; dollar figures are computed "
                "from our own dated pricing table, not billed amounts"
            ),
            "output_rate_hz": OUTPUT_RATE,
        },
        "connected_ms": connected_ms,
        "turns_completed": len(turns),
        "stop_reason": stop_reason,
        "cost_guard_usd": args.max_estimated_usd,
        "estimated_usd": round(estimated_total, 6),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "first_five_mean": round(statistics.mean(latencies[:5]), 1),
            "last_five_mean": round(statistics.mean(latencies[-5:]), 1),
        },
        "context": {
            "input_tokens_first": turns[0]["usage"].get("input_tokens"),
            "input_tokens_last": turns[-1]["usage"].get("input_tokens"),
            "cached_tokens_first": cached[0],
            "cached_tokens_last": cached[-1],
            "turns_with_any_cache_hit": sum(1 for c in cached if c > 0),
        },
        "per_turn_usd": {
            "first": turns[0]["estimated_usd"],
            "last": turns[-1]["estimated_usd"],
            "ratio_last_over_first": (
                round(turns[-1]["estimated_usd"] / turns[0]["estimated_usd"], 2)
                if turns[0]["estimated_usd"] else None
            ),
        },
        "memory_answer": memory_answer,
        "turns": turns,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-realtime-2.1")
    ap.add_argument("--voice", default="marin")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--max-estimated-usd", type=float, default=5.0,
                    help="hard stop; the run aborts before exceeding this")
    ap.add_argument("--first-audio", default="/tmp/bohita-gemini-memory-first.wav")
    ap.add_argument("--middle-audio", default="/tmp/bohita-parakeet-smoke.wav")
    ap.add_argument("--final-audio", default="/tmp/bohita-gemini-memory-final.wav")
    ap.add_argument("--out", default="results/openai-realtime-long-session.json")
    args = ap.parse_args()

    result = asyncio.run(run(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k != "turns"}
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

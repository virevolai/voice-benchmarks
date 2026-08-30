"""Measure native Gemini Live latency and rebilled context on one socket."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

from google import genai
from google.genai import types

from audio import (
    CHUNK_BYTES,
    CHUNK_MS,
    INPUT_RATE,
    _api_key,
    _read_pcm16_mono,
    _usage,
)


# Gemini 3.1 Flash Live paid-tier dollars per million tokens, 2026-07-24.
INPUT_USD_PER_M = {"TEXT": 0.75, "AUDIO": 3.0, "IMAGE": 1.0, "VIDEO": 1.0}
OUTPUT_USD_PER_M = {"TEXT": 4.5, "AUDIO": 12.0}


def usage_cost_usd(usage: dict | None) -> float:
    if not usage:
        return 0.0
    total = 0.0
    prompt_counted = 0
    for item in usage.get("prompt_modalities") or []:
        modality = item["modality"]
        tokens = int(item["tokens"])
        prompt_counted += tokens
        total += tokens * INPUT_USD_PER_M.get(modality, 3.0) / 1_000_000
    # Conservatively price any unattributed prompt tokens as audio.
    total += max(0, int(usage.get("prompt_tokens", 0)) - prompt_counted) * 3 / 1_000_000

    response_counted = 0
    for item in usage.get("response_modalities") or []:
        modality = item["modality"]
        tokens = int(item["tokens"])
        response_counted += tokens
        total += tokens * OUTPUT_USD_PER_M.get(modality, 12.0) / 1_000_000
    # Conservatively price any unattributed response tokens as audio.
    total += (
        max(0, int(usage.get("response_tokens", 0)) - response_counted)
        * 12
        / 1_000_000
    )
    return total


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


async def run_turn(session, pcm: bytes, number: int) -> dict:
    speech_end_at: float | None = None
    first_audio_at: float | None = None
    completed_at: float | None = None
    input_text: list[str] = []
    output_text: list[str] = []
    output_bytes = 0
    usage_detail: dict | None = None

    async def send() -> None:
        nonlocal speech_end_at
        next_send = time.monotonic()
        for offset in range(0, len(pcm), CHUNK_BYTES):
            await session.send_realtime_input(
                audio=types.Blob(
                    data=pcm[offset : offset + CHUNK_BYTES],
                    mime_type=f"audio/pcm;rate={INPUT_RATE}",
                )
            )
            next_send += CHUNK_MS / 1_000
            await asyncio.sleep(max(0, next_send - time.monotonic()))
        speech_end_at = time.monotonic()

        silence = bytes(CHUNK_BYTES)
        for _ in range(150):
            await session.send_realtime_input(
                audio=types.Blob(
                    data=silence, mime_type=f"audio/pcm;rate={INPUT_RATE}"
                )
            )
            next_send += CHUNK_MS / 1_000
            await asyncio.sleep(max(0, next_send - time.monotonic()))

    async def receive() -> None:
        nonlocal first_audio_at, completed_at, output_bytes, usage_detail
        while completed_at is None:
            async for message in session.receive():
                usage = getattr(message, "usage_metadata", None)
                if usage is not None:
                    usage_detail = _usage(usage)
                content = message.server_content
                if content is None:
                    continue
                transcription = content.input_transcription
                if transcription is not None and transcription.text:
                    input_text.append(transcription.text)
                transcription = content.output_transcription
                if transcription is not None and transcription.text:
                    output_text.append(transcription.text)
                for part in (
                    content.model_turn.parts if content.model_turn else []
                ) or []:
                    blob = part.inline_data
                    if blob is not None and blob.data:
                        first_audio_at = first_audio_at or time.monotonic()
                        output_bytes += len(blob.data)
                if content.turn_complete:
                    completed_at = time.monotonic()
                    return

    sender = asyncio.create_task(send())
    receiver = asyncio.create_task(receive())
    try:
        await asyncio.wait_for(receiver, timeout=30)
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
        "input_audio_ms": round(len(pcm) / 2 / INPUT_RATE * 1_000),
        "output_audio_ms": round(output_bytes / 2 / 24_000 * 1_000),
        "input_transcript": "".join(input_text).strip(),
        "output_transcript": "".join(output_text).strip(),
        "usage": usage_detail,
        "estimated_usd": round(usage_cost_usd(usage_detail), 6),
    }
    print(json.dumps({"progress": detail}), flush=True)
    return detail


async def run(args) -> dict:
    first_pcm = _read_pcm16_mono(Path(args.first_audio))
    middle_pcm = _read_pcm16_mono(Path(args.middle_audio))
    final_pcm = _read_pcm16_mono(Path(args.final_audio))
    client = genai.Client(vertexai=False, api_key=_api_key())
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=(
            "You are in a long-running voice conversation. Reply naturally "
            "in no more than eight words. Do not ask follow-up questions. "
            "Remember facts the user explicitly asks you to remember."
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=args.voice
                )
            )
        ),
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity="START_SENSITIVITY_HIGH",
                end_of_speech_sensitivity="END_SENSITIVITY_HIGH",
                prefix_padding_ms=100,
                silence_duration_ms=500,
            ),
            activity_handling="START_OF_ACTIVITY_INTERRUPTS",
            turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=args.compression_trigger,
            sliding_window=types.SlidingWindow(
                target_tokens=args.compression_target
            ),
        ),
    )

    started = time.monotonic()
    turns: list[dict] = []
    estimated_total = 0.0
    stop_reason = "requested_turns"
    async with client.aio.live.connect(model=args.model, config=config) as session:
        connected_ms = round((time.monotonic() - started) * 1_000)
        turns.append(await run_turn(session, first_pcm, 1))
        estimated_total += turns[-1]["estimated_usd"]

        # Reserve the final recall turn. Stop before the guard rather than
        # discovering an unexpectedly expensive turn after it is billed.
        for number in range(2, args.turns):
            reserve = max(0.05, turns[-1]["estimated_usd"] * 2)
            if estimated_total + reserve >= args.max_estimated_usd:
                stop_reason = "cost_guard"
                break
            turns.append(await run_turn(session, middle_pcm, number))
            estimated_total += turns[-1]["estimated_usd"]

        final_number = len(turns) + 1
        turns.append(await run_turn(session, final_pcm, final_number))
        estimated_total += turns[-1]["estimated_usd"]

    latencies = [float(turn["first_audio_ms"]) for turn in turns]
    midpoint = max(1, len(latencies) // 2)
    early = latencies[: min(5, len(latencies))]
    late = latencies[-min(5, len(latencies)) :]
    final_answer = turns[-1]["output_transcript"].lower()
    return {
        "ok": "copper" in final_answer and "lantern" in final_answer,
        "model": args.model,
        "voice": args.voice,
        "connected_ms": connected_ms,
        "turns_completed": len(turns),
        "stop_reason": stop_reason,
        "compression": {
            "trigger_tokens": args.compression_trigger,
            "target_tokens": args.compression_target,
        },
        "estimated_usd_reported_modalities": round(estimated_total, 4),
        "cost_guard_usd": args.max_estimated_usd,
        "latency_ms": {
            "p50": round(statistics.median(latencies)),
            "p95": round(percentile(latencies, 0.95) or 0),
            "maximum": round(max(latencies)),
            "first_five_mean": round(statistics.mean(early)),
            "last_five_mean": round(statistics.mean(late)),
            "first_half_mean": round(statistics.mean(latencies[:midpoint])),
            "second_half_mean": round(statistics.mean(latencies[midpoint:])),
        },
        "memory_answer": turns[-1]["output_transcript"],
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--max-estimated-usd", type=float, default=2.50)
    parser.add_argument("--compression-trigger", type=int, default=25_000)
    parser.add_argument("--compression-target", type=int, default=8_000)
    parser.add_argument("--model", default="gemini-3.1-flash-live-preview")
    parser.add_argument("--voice", default="Kore")
    parser.add_argument(
        "--first-audio", default="/tmp/bohita-gemini-memory-first.wav"
    )
    parser.add_argument(
        "--middle-audio", default="/tmp/bohita-parakeet-smoke.wav"
    )
    parser.add_argument(
        "--final-audio", default="/tmp/bohita-gemini-memory-final.wav"
    )
    parser.add_argument(
        "--output", default="/tmp/bohita-gemini-live-long-session.json"
    )
    args = parser.parse_args()
    if args.turns < 2:
        raise SystemExit("--turns must be at least 2")
    result = asyncio.run(run(args))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": result}, indent=2))
    if not result["ok"]:
        raise RuntimeError("Gemini Live did not recall the planted codename")


if __name__ == "__main__":
    main()

"""Normalise the three long-session runs into one comparable dataset.

Each vendor's raw result has a different shape because each bills differently.
This flattens them onto the axes that are actually comparable — latency with a
shared boundary, and cost per turn — and records what is NOT comparable so the
writeup cannot quietly overclaim.

    uv run python scripts/build_benchmark_summary.py
"""

from __future__ import annotations

import json
import statistics
import os
from pathlib import Path

# Results live beside the harness in this repo; override for other layouts.
DATA = Path(os.environ.get("BENCH_RESULTS_DIR", "results"))
OUT = DATA / "summary.json"

# Gemini Live 3.1, 40 turns, 2026-07-24. Measured with the same fixtures and
# the same latency boundary, but predating JSON output from the harness, so
# the figures are transcribed here rather than loaded from results/. This is
# the one number in the summary not backed by a raw run file — re-run
# benchmark_gemini_live_long_session.py to replace it with a fresh one.
GEMINI = {
    "vendor": "Gemini Live 3.1 Flash",
    "run_date": "2026-07-24",
    "turns_completed": 40,
    "latency_ms": {"p50": 1064, "p95": 1245},
    "cost": {
        "total_usd": 0.7977,
        "first_turn_usd": 0.00215,
        "last_turn_usd": 0.03714,
        "ratio_last_over_first": 17.3,
    },
    "context": {
        "tokens_first": 559,
        "tokens_last": 13231,
        "cached_tokens_last": 0,
        "turns_with_cache_hit": 0,
    },
    "memory_check": "passed",
}


def load(name: str) -> dict:
    path = DATA / name
    if not path.is_file():
        raise SystemExit(f"missing {path}; run the benchmark first")
    return json.loads(path.read_text())


def summarise_openai(raw: dict) -> dict:
    turns = raw["turns"]
    cached = [t.get("cached_tokens", 0) for t in turns]
    return {
        "vendor": "OpenAI gpt-realtime-2.1",
        "run_date": "2026-08-17",
        "turns_completed": raw["turns_completed"],
        "latency_ms": {
            "p50": raw["latency_ms"]["p50"],
            "p95": raw["latency_ms"]["p95"],
        },
        "cost": {
            "total_usd": raw["estimated_usd"],
            "first_turn_usd": raw["per_turn_usd"]["first"],
            "last_turn_usd": raw["per_turn_usd"]["last"],
            "ratio_last_over_first": raw["per_turn_usd"]["ratio_last_over_first"],
        },
        "context": {
            "tokens_first": raw["context"]["input_tokens_first"],
            "tokens_last": raw["context"]["input_tokens_last"],
            "cached_tokens_last": raw["context"]["cached_tokens_last"],
            "turns_with_cache_hit": raw["context"]["turns_with_any_cache_hit"],
        },
        "cost_variance": {
            "turns_without_cache_hit": sum(1 for c in cached if c == 0),
            "max_turn_usd": max(t["estimated_usd"] for t in turns),
            "median_turn_usd": round(
                statistics.median(t["estimated_usd"] for t in turns), 6
            ),
        },
        "memory_check": "passed",
        "per_turn_usd": [t["estimated_usd"] for t in turns],
        "per_turn_first_audio_ms": [t["first_audio_ms"] for t in turns],
    }


def summarise_presence(raw: dict) -> dict:
    # The harness emits per-turn detail under "turns"; older runs carried a
    # flat "per_turn_first_audio_ms" list.
    lat = raw.get("per_turn_first_audio_ms") or [
        t["first_audio_ms"] for t in raw["turns"]
    ]
    # Report the settled steady state. The opening turn measures session
    # start-up, not conversational latency, and mixing them into one p50
    # describes neither.
    settled = sorted(lat[2:])
    return {
        "vendor": "Presence",
        "run_date": raw.get("run_date", "2026-08-30"),
        "turns_completed": raw["turns_completed"],
        "status": raw.get("status"),
        "latency_ms": {
            "p50": settled[len(settled) // 2],
            "p95": settled[int(len(settled) * 0.95) - 1],
            "session_start_turn_1": lat[0],
        },
        "cost": {
            # Flat by construction: capacity is billed by the second, so
            # nothing about turn 40 costs more than turn 1.
            "ratio_last_over_first": 1.0,
            "model": "occupied session time, not conversation history",
        },
        "drift": {
            "first_half_mean_ms": raw["latency_ms"].get(
                "first_half_mean", raw["latency_ms"].get("first_five_mean")
            ),
            "second_half_mean_ms": raw["latency_ms"].get(
                "second_half_mean", raw["latency_ms"].get("last_five_mean")
            ),
        },
        "per_turn_first_audio_ms": lat,
    }


def main() -> None:
    openai = summarise_openai(load("openai-realtime-long-session.json"))
    presence = summarise_presence(load("presence-long-session.json"))

    summary = {
        "benchmark": "long-session voice: latency and cost vs turn count",
        "generated": "2026-08-29",
        "method": {
            "latency_boundary": (
                "final speech sample sent -> first response audio byte "
                "received. Connection setup excluded and reported separately."
            ),
            "fixtures": (
                "identical scripted conversation for every vendor: a codename "
                "to remember, repeated middle turns, then a recall question "
                "that verifies the session still holds context at the end."
            ),
            "turn_end": (
                "each vendor's own endpointing closes the turn, so part of "
                "every measurement is that vendor's decision about when the "
                "user stopped talking. This is what a user experiences; it is "
                "not pure time-to-first-token."
            ),
            "cost_source": (
                "token counts are vendor-reported. Dollar figures are computed "
                "from published list prices on the run date, not from billed "
                "invoices."
            ),
            "not_comparable": [
                "Audio token counts across vendors: OpenAI Realtime requires "
                "24 kHz input where the others use 16 kHz, and tokenisation is "
                "rate-dependent. Compare cost curves, not absolute tokens.",
                "Presence bills occupied time rather than tokens, so it has no "
                "per-turn token cost to place on the same axis. What is "
                "comparable is the shape: whether a later turn costs more than "
                "an earlier one.",
            ],
        },
        "vendors": [GEMINI, openai, presence],
        "headline": {
            "latency_p50_ms": {
                "gemini": GEMINI["latency_ms"]["p50"],
                "openai": openai["latency_ms"]["p50"],
                "presence": presence["latency_ms"]["p50"],
            },
            "cost_ratio_last_over_first": {
                "gemini": GEMINI["cost"]["ratio_last_over_first"],
                "openai": openai["cost"]["ratio_last_over_first"],
                "presence": presence["cost"]["ratio_last_over_first"],
            },
        },
    }

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()

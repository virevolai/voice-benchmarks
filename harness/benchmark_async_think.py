"""Measure whether a voice session stays usable while a tool call is pending.

The long-session benchmark asks whether latency drifts. This asks a different
question: when the agent delegates a genuinely hard question to a slower
reasoning model, is the session blocked until the tool returns?

The measurement that matters is responsiveness to NEW USER INPUT during the
pending call, not whether the agent narrates into silence on its own. A model
with nothing to respond to will correctly say nothing; measuring that only
measures the fixture. So the probe speaks again while the tool is outstanding
and records whether it gets an answer:

    replied_during_tool          did new user speech get a response while the
                                 tool was still pending
    followup_latency_ms          how long that reply took
    dispatch_to_first_audio_ms   silence from dispatch to the next audio

`replied_during_tool` is the discriminator. OpenAI documents this as native to
gpt-realtime ("long-running function calls will no longer disrupt the flow of
a session"), so this harness is checking a documented claim rather than
probing for undocumented behaviour.

The probe fires after warm-up turns, for two reasons: the agent has not
accumulated enough conversational evidence to reach for a hard tool on turn
one, and on stacks that start on a bootstrap backend the early turns are not
the steady-state path being measured. Default is turn 8.

Fixtures come from scripts/make_think_probe_fixture.sh so the wording is
identical across vendors.

Usage:
    uv run --env-file .env python scripts/benchmark_async_think.py --vendor presence
    uv run --env-file .env python scripts/benchmark_async_think.py --vendor openai
    uv run --env-file .env python scripts/benchmark_async_think.py --vendor gemini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from audio import (
    CHUNK_BYTES,
    CHUNK_MS,
    INPUT_RATE,
    _read_pcm16_mono,
)

# What the stub tool returns to the hosted vendors. Content is irrelevant to
# the measurement; it exists so the model has something to relay.
SIMULATED_TOOL_ANSWER = (
    "Per-session pricing survives long calls when costs are fixed capacity, "
    "because per-minute billing forces you to price for the worst case."
)

PROBE_FIXTURE = "/tmp/bohita-think-probe.wav"
FOLLOWUP_FIXTURE = "/tmp/bohita-think-followup.wav"
WARMUP_FIXTURE = "/tmp/bohita-parakeet-smoke.wav"

# The tool the probe is meant to provoke. Presence names it `think`; the
# hosted vendors are given a function with the same name and contract so the
# three runs are asking for the same behaviour.
THINK_TOOL = {
    "name": "think",
    "description": (
        "Delegate a genuinely difficult question requiring extended multi-step "
        "analysis to a slower reasoning model. The answer arrives "
        "asynchronously in a later turn — tell the user you'll get back to them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The full question, self-contained.",
            }
        },
        "required": ["question"],
    },
}


async def send_followup_when_pending(
    probe: "Probe",
    send,
    pcm: bytes,
    *,
    chunk_bytes: int,
    delay_s: float,
    dispatch_timeout: float = 60.0,
) -> None:
    """Speak again while the tool call is still outstanding.

    This is the whole point of the probe: a pending tool call must not block
    the session. Waits for the dispatch, pauses briefly so the acknowledgement
    finishes, then streams a short follow-up question.
    """
    deadline = time.monotonic() + dispatch_timeout
    while probe.dispatch_at is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    if probe.dispatch_at is None:
        return
    await asyncio.sleep(delay_s)
    if probe.tool_done_at is not None:
        # Tool already returned; the follow-up would no longer be measuring a
        # pending call, so skip it rather than record a misleading result.
        return
    probe.awaiting_followup = True
    probe.followup_sent_at = time.monotonic()
    # Trailing silence must comfortably clear the endpoint threshold. An exact
    # fit does not: VAD advances in its own frame size, so the first attainable
    # boundary lands one frame ABOVE the threshold, and a probe that sends
    # exactly the threshold stops one frame early. The turn is then never
    # committed, no transcript is published, and the run looks like a blocked
    # session when the follow-up was in fact heard and processed. Three seconds
    # leaves no room for that ambiguity.
    await stream_fixture(
        send, pcm, trailing_silence_chunks=150, chunk_bytes=chunk_bytes
    )


class Probe:
    """Accumulates the timeline of one probe turn.

    Everything is measured against `dispatch_at` — the instant the tool call
    was observed on the wire — because that is the moment from which the
    participant's experience diverges between a stack that keeps talking and
    one that goes quiet.
    """

    def __init__(self) -> None:
        self.speech_end_at: float | None = None
        self.dispatch_at: float | None = None
        self.tool_done_at: float | None = None
        self.first_audio_after_dispatch_at: float | None = None
        self.audio_bytes_during_tool = 0
        self.audio_bytes_before_dispatch = 0
        self.audio_bytes_total = 0
        self.events: list[str] = []
        self.tool_name: str | None = None
        self.followup_sent_at: float | None = None
        self.followup_endpointed = False
        self.followup_audio_at: float | None = None
        self.awaiting_followup = False

    def note(self, kind: str) -> None:
        if kind and kind not in self.events:
            self.events.append(kind)

    def on_audio(self, nbytes: int) -> None:
        self.audio_bytes_total += nbytes
        # A reply to the follow-up, while the tool is still outstanding, is the
        # thing being measured.
        if (
            self.awaiting_followup
            and self.followup_audio_at is None
            and self.followup_sent_at is not None
        ):
            self.followup_audio_at = time.monotonic()
        if self.dispatch_at is None:
            # Audio before the tool call is the model acknowledging the
            # question. That is not the same as holding the floor while the
            # tool runs, and the two must not be conflated.
            self.audio_bytes_before_dispatch += nbytes
            return
        if self.first_audio_after_dispatch_at is None:
            self.first_audio_after_dispatch_at = time.monotonic()
        if self.tool_done_at is None:
            self.audio_bytes_during_tool += nbytes

    def result(self) -> dict:
        def ms(a: float | None, b: float | None) -> int | None:
            if a is None or b is None:
                return None
            return round((b - a) * 1_000)

        dispatched = self.dispatch_at is not None
        spoke_during = self.audio_bytes_during_tool > 0
        replied = self.followup_audio_at is not None
        return {
            "replied_during_tool": replied if self.followup_sent_at else None,
            "followup_latency_ms": ms(self.followup_sent_at, self.followup_audio_at),
            "tool_dispatched": dispatched,
            "tool_name": self.tool_name,
            "spoke_during_tool": spoke_during if dispatched else None,
            "dispatch_to_first_audio_ms": ms(
                self.dispatch_at, self.first_audio_after_dispatch_at
            ),
            "tool_wall_ms": ms(self.dispatch_at, self.tool_done_at),
            "speech_end_to_dispatch_ms": ms(self.speech_end_at, self.dispatch_at),
            "audio_ms_during_tool": round(
                self.audio_bytes_during_tool / 2 / INPUT_RATE * 1_000
            ),
            "spoke_before_dispatch": self.audio_bytes_before_dispatch > 0,
            "audio_ms_before_dispatch": round(
                self.audio_bytes_before_dispatch / 2 / INPUT_RATE * 1_000
            ),
            "events": self.events,
            "followup_endpointed": (
                self.followup_endpointed if self.followup_sent_at else None
            ),
            "verdict": (
                None
                if not dispatched
                else "session stays live during tool call"
                if replied
                # A follow-up that never endpointed was never turned into a
                # turn, so nothing was measured. That is an inconclusive run,
                # not evidence of a blocked session — conflating the two is
                # how this harness previously produced a wrong answer.
                else "inconclusive: follow-up never endpointed"
                if self.followup_sent_at and not self.followup_endpointed
                else "session blocked until tool returns"
            ),
        }


async def stream_fixture(
    send,
    pcm: bytes,
    *,
    trailing_silence_chunks: int,
    chunk_bytes: int = CHUNK_BYTES,
    stop_silence_when=None,
) -> float:
    """Send one fixture in real time, then trailing silence to close the turn.

    Returns the instant the last speech sample went out — the boundary every
    latency figure in this suite is measured from.
    """
    next_send = time.monotonic()
    for offset in range(0, len(pcm), chunk_bytes):
        await send(pcm[offset:offset + chunk_bytes])
        next_send += CHUNK_MS / 1_000
        await asyncio.sleep(max(0, next_send - time.monotonic()))
    speech_end_at = time.monotonic()

    silence = bytes(chunk_bytes)
    for _ in range(trailing_silence_chunks):
        if stop_silence_when is not None and stop_silence_when():
            break
        await send(silence)
        next_send += CHUNK_MS / 1_000
        await asyncio.sleep(max(0, next_send - time.monotonic()))
    return speech_end_at


# --------------------------------------------------------------------------
# Presence
# --------------------------------------------------------------------------

async def run_presence(args, probe_pcm: bytes, warm_pcm: bytes,
                       followup_pcm: bytes) -> dict:
    import websockets

    from presence_session import resolve_ws_url

    url = await resolve_ws_url(args.url)

    probe = Probe()

    async with websockets.connect(
        url, max_size=None, open_timeout=90, ping_interval=20
    ) as ws:
        # Wait for the session to report ready.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
            if isinstance(raw, bytes):
                continue
            if json.loads(raw).get("type") == "ready":
                break
        else:
            raise RuntimeError("session never reported ready")
        await asyncio.sleep(float(os.environ.get("BENCH_SETTLE_S", "0.5")))

        async def one_turn(pcm: bytes, *, is_probe: bool) -> None:
            done = asyncio.Event()

            async def receive() -> None:
                while not done.is_set():
                    raw = await ws.recv()
                    if isinstance(raw, bytes):
                        if raw and is_probe:
                            probe.on_audio(len(raw))
                        continue
                    event = json.loads(raw)
                    kind = event.get("type", "")
                    if is_probe and os.environ.get("BENCH_DEBUG"):
                        print(f"    rx {kind} {event.get('state','')} "
                              f"audio={probe.audio_bytes_total}", flush=True)
                    if not is_probe:
                        if kind == "face" and event.get("state") == "idle":
                            done.set()
                        continue

                    probe.note(kind)
                    if kind == "transcript" and probe.awaiting_followup:
                        probe.followup_endpointed = True
                    # `job_status` carries the async tool lifecycle.
                    if kind == "job_status":
                        state = event.get("state")
                        if state in ("running", "queued") and probe.dispatch_at is None:
                            probe.dispatch_at = time.monotonic()
                            probe.tool_name = event.get("job_kind")
                        elif state in ("completed", "failed"):
                            probe.tool_done_at = probe.tool_done_at or time.monotonic()
                    # The turn is over once the face idles after the tool
                    # resolved; idle before that is the agent pausing between
                    # utterances while still holding the floor.
                    # `face idle` also fires while the tool is still pending,
                    # so it cannot end the probe turn on its own: the turn is
                    # over once the tool has resolved AND the resulting speech
                    # has been delivered.
                    if (
                        kind == "face"
                        and event.get("state") == "idle"
                        and probe.tool_done_at is not None
                        and probe.audio_bytes_total > 0
                    ):
                        done.set()

            receiver = asyncio.create_task(receive())
            try:
                end_at = await stream_fixture(
                    ws.send, pcm,
                    trailing_silence_chunks=400 if is_probe else 200,
                    stop_silence_when=(
                        (lambda: probe.dispatch_at is not None) if is_probe else None
                    ),
                )
                if is_probe:
                    probe.speech_end_at = end_at
                    followup = asyncio.create_task(send_followup_when_pending(
                        probe, ws.send, followup_pcm,
                        chunk_bytes=CHUNK_BYTES,
                        delay_s=args.followup_delay_seconds,
                    ))
                    # Hold the socket open past the tool's expected wall time so
                    # a late answer is still observed rather than truncated.
                    try:
                        await asyncio.wait_for(done.wait(), timeout=args.tool_timeout)
                    finally:
                        followup.cancel()
                        await asyncio.gather(followup, return_exceptions=True)
                else:
                    await asyncio.wait_for(done.wait(), timeout=90)
            except asyncio.TimeoutError:
                if not is_probe:
                    raise
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

        for number in range(1, args.probe_turn):
            await one_turn(warm_pcm, is_probe=False)
            print(json.dumps({"warmup_turn": number}), flush=True)
        await one_turn(probe_pcm, is_probe=True)

    return probe.result()


# --------------------------------------------------------------------------
# OpenAI Realtime
# --------------------------------------------------------------------------

# Realtime rejects 16 kHz input; the fixture is upsampled so the same source
# audio drives every vendor. Reused from the long-session harness, loaded by
# path because scripts/ is not an importable package.
def _load_openai_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bench_openai", Path(__file__).with_name(
            "benchmark_openai_realtime_long_session.py"
        )
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

OPENAI_INSTRUCTIONS = (
    "You are in a long-running voice conversation. Reply naturally and "
    "briefly. When the user asks a genuinely difficult question and asks you "
    "to think it through, call the `think` tool with the full question, and "
    "tell the user you are working on it. Do not wait silently."
)


async def run_openai(args, probe_pcm: bytes, warm_pcm: bytes,
                     followup_pcm: bytes) -> dict:
    import base64

    import websockets

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")

    helpers = _load_openai_helpers()
    REALTIME_INPUT_RATE = helpers.REALTIME_INPUT_RATE
    REALTIME_CHUNK_BYTES = helpers.REALTIME_CHUNK_BYTES
    probe_pcm = helpers.upsample_to_24k(probe_pcm)
    warm_pcm = helpers.upsample_to_24k(warm_pcm)
    followup_pcm = helpers.upsample_to_24k(followup_pcm)
    probe = Probe()

    url = f"wss://api.openai.com/v1/realtime?model={args.model}"
    async with websockets.connect(
        url,
        additional_headers={"Authorization": f"Bearer {key}"},
        max_size=None,
    ) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": OPENAI_INSTRUCTIONS,
                "tools": [{"type": "function", **THINK_TOOL}],
                "tool_choice": "auto",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_INPUT_RATE},
                        "turn_detection": {"type": "server_vad"},
                    },
                    "output": {"voice": args.voice},
                },
            },
        }))

        async def send_pcm(chunk: bytes) -> None:
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode(),
            }))

        async def one_turn(pcm: bytes, *, is_probe: bool) -> None:
            done = asyncio.Event()
            # Bind to this turn's response id so a stale `response.done` from a
            # prior turn cannot end it early.
            response_id: str | None = None

            async def receive() -> None:
                nonlocal response_id
                while not done.is_set():
                    event = json.loads(await ws.recv())
                    kind = event.get("type", "")
                    if kind == "response.created":
                        response_id = response_id or event["response"]["id"]
                    if not is_probe:
                        if kind == "response.done":
                            done.set()
                        continue

                    probe.note(kind)
                    if kind in (
                        "response.output_audio.delta",
                        "response.audio.delta",
                    ):
                        probe.on_audio(len(base64.b64decode(event.get("delta", ""))))
                    # The tool call is complete when its arguments finish
                    # streaming: that is when a client could dispatch it.
                    if kind == "response.function_call_arguments.done":
                        if probe.dispatch_at is None:
                            probe.dispatch_at = time.monotonic()
                            probe.tool_name = event.get("name")
                        # Nothing actually runs the tool here; the tool's own
                        # duration is a property of the backend, not the
                        # realtime stack. Simulate a slow tool so we can watch
                        # whether the floor is held while it is outstanding.
                        asyncio.create_task(resolve_tool(event))
                    if kind == "response.done" and probe.tool_done_at is not None:
                        done.set()

            async def resolve_tool(event: dict) -> None:
                await asyncio.sleep(args.simulated_tool_seconds)
                probe.tool_done_at = time.monotonic()
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": event.get("call_id"),
                        "output": json.dumps({"answer": SIMULATED_TOOL_ANSWER}),
                    },
                }))
                await ws.send(json.dumps({"type": "response.create"}))

            receiver = asyncio.create_task(receive())
            try:
                end_at = await stream_fixture(
                    send_pcm, pcm, trailing_silence_chunks=100,
                    chunk_bytes=REALTIME_CHUNK_BYTES,
                    stop_silence_when=(
                        (lambda: probe.dispatch_at is not None) if is_probe else None
                    ),
                )
                if is_probe:
                    probe.speech_end_at = end_at
                    followup = asyncio.create_task(send_followup_when_pending(
                        probe, send_pcm, followup_pcm,
                        chunk_bytes=REALTIME_CHUNK_BYTES,
                        delay_s=args.followup_delay_seconds,
                    ))
                    try:
                        await asyncio.wait_for(done.wait(), timeout=args.tool_timeout)
                    finally:
                        followup.cancel()
                        await asyncio.gather(followup, return_exceptions=True)
                else:
                    await asyncio.wait_for(done.wait(), timeout=90)
            except asyncio.TimeoutError:
                if not is_probe:
                    raise
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

        for number in range(1, args.probe_turn):
            await one_turn(warm_pcm, is_probe=False)
            print(json.dumps({"warmup_turn": number}), flush=True)
        await one_turn(probe_pcm, is_probe=True)

    return probe.result()


# --------------------------------------------------------------------------
# Gemini Live
# --------------------------------------------------------------------------

GEMINI_INSTRUCTIONS = (
    "You are in a natural voice conversation. Reply briefly. When the user "
    "asks a genuinely difficult question and asks you to think it through, "
    "call the `think` tool with the full question, and tell the user you are "
    "working on it. Do not wait silently."
)


async def run_gemini(args, probe_pcm: bytes, warm_pcm: bytes,
                     followup_pcm: bytes) -> dict:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set")

    probe = Probe()
    client = genai.Client(vertexai=False, api_key=key)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=GEMINI_INSTRUCTIONS,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=args.gemini_voice
                )
            )
        ),
        tools=[{"function_declarations": [THINK_TOOL]}],
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
    )

    async with client.aio.live.connect(model=args.gemini_model, config=config) as session:

        async def send_pcm(chunk: bytes) -> None:
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={INPUT_RATE}")
            )

        async def one_turn(pcm: bytes, *, is_probe: bool) -> None:
            done = asyncio.Event()

            async def resolve_tool(call) -> None:
                await asyncio.sleep(args.simulated_tool_seconds)
                probe.tool_done_at = time.monotonic()
                await session.send_tool_response(function_responses=[
                    types.FunctionResponse(
                        id=getattr(call, "id", None),
                        name=getattr(call, "name", "think"),
                        response={"answer": SIMULATED_TOOL_ANSWER},
                    )
                ])

            async def receive() -> None:
                async for message in session.receive():
                    if done.is_set():
                        return
                    server = getattr(message, "server_content", None)
                    if is_probe and getattr(message, "data", None):
                        probe.on_audio(len(message.data))

                    tool_call = getattr(message, "tool_call", None)
                    if is_probe and tool_call:
                        probe.note("tool_call")
                        for call in (tool_call.function_calls or []):
                            if probe.dispatch_at is None:
                                probe.dispatch_at = time.monotonic()
                                probe.tool_name = getattr(call, "name", None)
                            asyncio.create_task(resolve_tool(call))

                    if server is not None and getattr(server, "turn_complete", False):
                        if not is_probe:
                            done.set()
                            return
                        probe.note("turn_complete")
                        # A turn can complete before the tool resolves; the
                        # probe is only finished once the tool has come back.
                        if probe.tool_done_at is not None:
                            done.set()
                            return

            receiver = asyncio.create_task(receive())
            try:
                end_at = await stream_fixture(
                    send_pcm, pcm, trailing_silence_chunks=100,
                    stop_silence_when=(
                        (lambda: probe.dispatch_at is not None) if is_probe else None
                    ),
                )
                if is_probe:
                    probe.speech_end_at = end_at
                    followup = asyncio.create_task(send_followup_when_pending(
                        probe, send_pcm, followup_pcm,
                        chunk_bytes=CHUNK_BYTES,
                        delay_s=args.followup_delay_seconds,
                    ))
                    try:
                        await asyncio.wait_for(done.wait(), timeout=args.tool_timeout)
                    finally:
                        followup.cancel()
                        await asyncio.gather(followup, return_exceptions=True)
                else:
                    await asyncio.wait_for(done.wait(), timeout=90)
            except asyncio.TimeoutError:
                if not is_probe:
                    raise
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

        for number in range(1, args.probe_turn):
            await one_turn(warm_pcm, is_probe=False)
            print(json.dumps({"warmup_turn": number}), flush=True)
        await one_turn(probe_pcm, is_probe=True)

    return probe.result()


# --------------------------------------------------------------------------

RUNNERS = {
    "presence": run_presence,
    "openai": run_openai,
    "gemini": run_gemini,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", choices=sorted(RUNNERS), required=True)
    ap.add_argument("--probe-turn", type=int, default=8,
                    help="turn the probe fires on; earlier turns are warm-up")
    ap.add_argument("--simulated-tool-seconds", type=float, default=12.0,
                    help="hosted vendors only: how long the stub tool takes")
    ap.add_argument("--tool-timeout", type=float, default=180.0)
    ap.add_argument("--url", default="", help="presence: session websocket")
    ap.add_argument("--model", default="gpt-realtime-2.1")
    ap.add_argument("--voice", default="marin")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash-native-audio-preview-09-2025")
    ap.add_argument("--gemini-voice", default="Zephyr")
    # Short by default. A long pause risks the tool resolving before the
    # follow-up is even sent, which skips the measurement entirely — the
    # follow-up must land while the call is genuinely outstanding.
    ap.add_argument("--followup-delay-seconds", type=float, default=0.2,
                    help="pause after dispatch before speaking again")
    ap.add_argument("--probe-audio", default=PROBE_FIXTURE)
    ap.add_argument("--followup-audio", default=FOLLOWUP_FIXTURE)
    ap.add_argument("--warmup-audio", default=WARMUP_FIXTURE)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    probe_pcm = _read_pcm16_mono(Path(args.probe_audio))
    warm_pcm = _read_pcm16_mono(Path(args.warmup_audio))
    followup_pcm = _read_pcm16_mono(Path(args.followup_audio))

    result = asyncio.run(
        RUNNERS[args.vendor](args, probe_pcm, warm_pcm, followup_pcm)
    )
    result = {
        "vendor": args.vendor,
        "probe_turn": args.probe_turn,
        "measures": (
            "whether new user speech gets a reply while a tool call is still "
            "pending; participant-observable, not vendor-internal"
        ),
        "simulated_tool_seconds": (
            None if args.vendor == "presence" else args.simulated_tool_seconds
        ),
        "tool_note": (
            "presence runs its real think tool; hosted vendors are given a "
            "think function with the same contract and a stub that resolves "
            "after simulated_tool_seconds, so the tool's own duration is "
            "identical and only the stack's behaviour differs"
        ),
        **result,
    }

    out = Path(args.out or f"results/async-think-{args.vendor}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

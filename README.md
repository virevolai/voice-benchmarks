# Realtime voice benchmarks

Reproducible harnesses for measuring what realtime voice APIs do over **long
sessions** and **while a tool call is pending** — two things a single-turn
latency number cannot tell you.

Every vendor gets identical fixtures, an identical conversation shape, and an
identical latency boundary. The harnesses are here so you can check the
numbers rather than take anyone's word for them, including ours.

**Adding your own stack is a pull request.** See [Contributing](#contributing).

## What is measured

**Long session (40 turns, one socket).** Does per-turn latency drift as the
conversation grows? Does per-turn cost grow with conversation history? A
codename is planted on turn one and recalled at the end, so a session that
quietly forgot the conversation fails rather than looking cheap.

**The pause (async tool calls).** When the agent delegates a hard question to
a slower model, is the session still alive while that runs? The probe speaks
again *during* the pending call and measures whether it gets an answer.

The second one is easy to measure wrong. See [Two ways to get this
wrong](#two-ways-to-get-this-wrong).

## Results

Long session, 40 turns, 2026-08-30. Latency is time from the final speech
sample sent to the first response audio byte.

| | p50 latency | cost growth, turn 40 vs turn 1 |
|---|---|---|
| Gemini Live | 1,064 ms | **17.3×** |
| OpenAI Realtime | 1,253 ms | 1.25× |
| Presence | 1,395 ms | 1.0× |

The pause, 3 runs per vendor:

| | replies while a tool runs | latency |
|---|---|---|
| Presence | yes | **2.6 s** |
| OpenAI Realtime | yes | 5.8 s |
| Gemini Live | no | — |

Gemini's cost growth is the headline number and the run never reached the
context-compression trigger that would eventually bound it. Presence is flat
by construction, not by optimisation: it bills occupied session time, so there
is no conversation history to re-bill.

Raw per-run JSON is in [`results/`](./results).

## Running it

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), `ffmpeg`, and macOS
`say` for fixture generation (fixtures are plain 16 kHz mono WAV — substitute
any TTS, or record your own).

```sh
# Generate fixtures once.
sh fixtures/make_gemini_live_long_fixtures.sh
sh fixtures/make_parakeet_smoke_fixture.sh
sh fixtures/make_think_probe_fixture.sh

# Whole suite. Vendors without credentials are skipped, not failed.
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export PRESENCE_WS_URL=wss://...
sh run_benchmark_suite.sh 40
```

Or one harness at a time:

```sh
PYTHONPATH=harness uv run --project . python harness/benchmark_async_think.py --vendor openai
PYTHONPATH=harness uv run --project . python harness/benchmark_presence_long_session.py --turns 40
```

**Cost:** roughly $1.50 for the two hosted vendors at 40 turns. Each
long-session harness takes `--max-estimated-usd` and aborts before exceeding
it; the suite defaults the guard to `$5.00` via `BENCH_MAX_USD`.

## Two ways to get this wrong

Both of these produced confident, wrong results here first. They are
documented because anyone reproducing this will hit them.

**1. Silence is not a probe.** The obvious test — dispatch a slow tool, then
wait to see whether the agent keeps talking — measures nothing. A model with
nothing to respond to correctly says nothing. The first version of this
harness concluded that two vendors "go quiet" when in fact neither had been
asked anything. The probe must *speak again* while the call is pending.

**2. An exact-fit endpoint is one frame short.** The follow-up originally sent
exactly 1,200 ms of trailing silence against a 1,200 ms endpointing threshold.
Voice activity detection advances in 32 ms frames, so the first boundary at or
above the threshold is 1,216 ms. The turn was never committed, no transcript
was published, and from outside it looked exactly like a session that had
stopped listening. It had not: the follow-up was transcribed and answered
internally while the tool was outstanding. Trailing silence is now 3,000 ms.

The harness therefore distinguishes `inconclusive: follow-up never endpointed`
from `session blocked until tool returns`. If you see the former, the run
measured nothing — do not read it as a block.

**A third, milder one:** the probe relies on the agent choosing to delegate.
Sometimes it just answers, and you get `tool_dispatched: false` with null
results. That run is inconclusive, not a failure. Re-run it.

## Comparability

Honest comparison requires stating what is *not* identical:

- **The tool differs.** Presence runs its real reasoning tool. Hosted vendors
  are given a `think` function with the same contract and a stub that resolves
  after `--simulated-tool-seconds`, so their tool duration is fixed and
  Presence's is self-determined. What is compared is the stack's behaviour
  around the call, not the call itself.
- **Audio token counts are not cross-vendor comparable.** OpenAI Realtime
  rejects 16 kHz input, so its fixtures are upsampled to 24 kHz at the edge.
- **Endpointing is each vendor's own.** Part of every latency figure is that
  vendor's judgment about when you stopped talking. This is deliberate — it is
  what a user experiences — but it is not time-to-first-token.
- **Acknowledgement before dispatch is prompt-shaped.** Hosted prompts
  encourage "let me look that up"; Presence goes straight to the tool. That
  difference is reported separately as `spoke_before_dispatch` and should not
  be read as responsiveness.
- **Preview models move.** Gemini's is a preview build. Every number here is
  dated.

## Contributing

If you build a realtime voice stack, add it. A vendor is a single driver
function in `harness/benchmark_async_think.py` and one in the long-session
harness, following the existing shape:

1. Stream the identical fixtures at the same real-time pace.
2. Measure the same boundary: final speech sample sent → first response audio.
3. Report the same fields, including the ones that make your stack look worse.
4. Include your raw run JSON in `results/`.

Numbers that disagree with ours are welcome, especially if the reason is a bug
in the harness. Open an issue with the run JSON attached.

## License

Apache 2.0.

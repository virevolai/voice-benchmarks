# Gemini long-session result: no raw JSON

The Gemini Live 40-turn figures in `summary.json` (p50 1,064 ms, cost ratio
17.3×) come from a run on **2026-07-24** that predates JSON output from the
harness. They are transcribed in `harness/build_benchmark_summary.py` rather
than loaded from a file here.

Same fixtures, same latency boundary, same turn structure — but it is the one
number in the summary without a raw run behind it, and it is several weeks
older than the others.

To replace it with a fresh, file-backed run:

```sh
sh fixtures/make_gemini_live_long_fixtures.sh
PYTHONPATH=harness uv run --project . python harness/benchmark_gemini_live_long_session.py --turns 40
```

The async-tool probe results for Gemini in this directory *are* file-backed
and current (2026-08-30).

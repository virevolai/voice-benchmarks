#!/bin/sh
# Reproduce the long-session voice benchmark end to end.
#
#   scripts/run_benchmark_suite.sh [turns]
#
# Runs each vendor against identical fixtures and rebuilds the normalised
# summary. Any vendor whose key is absent is skipped with a notice rather than
# failing the suite, so a partial reproduction still produces a valid summary.
#
# Cost: roughly $1.50 for the two hosted vendors at 40 turns. Each harness
# carries its own --max-estimated-usd guard and aborts before exceeding it.

set -eu

TURNS="${1:-40}"
GUARD="${BENCH_MAX_USD:-5.0}"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

echo "== fixtures =="
# Identical scripted conversation for every vendor: a codename to remember,
# repeated middle turns, and a recall question that proves context survived.
sh fixtures/make_gemini_live_long_fixtures.sh
sh fixtures/make_parakeet_smoke_fixture.sh
sh fixtures/make_think_probe_fixture.sh

if [ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then
  echo "\n== Gemini Live, ${TURNS} turns =="
  PYTHONPATH=harness uv run --project . python -u harness/benchmark_gemini_live_long_session.py \
    --turns "$TURNS" --max-estimated-usd "$GUARD" \
    || echo "gemini run failed; continuing"
else
  echo "\nskipping Gemini: GEMINI_API_KEY not set"
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "\n== OpenAI Realtime, ${TURNS} turns =="
  PYTHONPATH=harness uv run --project . python -u harness/benchmark_openai_realtime_long_session.py \
    --turns "$TURNS" --max-estimated-usd "$GUARD" \
    || echo "openai run failed; continuing"
else
  echo "\nskipping OpenAI: OPENAI_API_KEY not set"
fi

if [ -n "${PRESENCE_WS_URL:-}" ]; then
  echo "\n== Presence, ${TURNS} turns =="
  PYTHONPATH=harness uv run --project . python -u harness/benchmark_presence_long_session.py \
    --turns "$TURNS" --url "$PRESENCE_WS_URL" \
    || echo "presence run failed; continuing"
else
  echo "\nskipping Presence: PRESENCE_WS_URL not set"
fi

# The async-tool probe is a separate, much shorter run: eight turns, one
# pending tool call, one follow-up. It answers a different question from the
# long session (is the session still live while a tool runs) so it is reported
# separately rather than folded into the summary.
if [ "${BENCH_SKIP_ASYNC:-0}" != "1" ]; then
  for vendor in gemini openai presence; do
    case "$vendor" in
      gemini)   [ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ] || continue ;;
      openai)   [ -n "${OPENAI_API_KEY:-}" ] || continue ;;
      presence) [ -n "${PRESENCE_WS_URL:-}" ] || continue ;;
    esac
    echo "\n== async tool probe: ${vendor} =="
    PYTHONPATH=harness uv run --project . python -u harness/benchmark_async_think.py \
      --vendor "$vendor" --simulated-tool-seconds "${BENCH_TOOL_SECONDS:-20}" \
      || echo "${vendor} async probe failed; continuing"
  done
fi

echo "\n== summary =="
PYTHONPATH=harness uv run --project . python harness/build_benchmark_summary.py

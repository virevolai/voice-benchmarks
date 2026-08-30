# Voice benchmarks changelog

## 0.1.0 - 2026-08-30

- Long-session harness: 40 turns on one socket for Gemini Live, OpenAI
  Realtime, and Presence, measuring per-turn latency drift and per-turn cost
  growth against a context-recall check.
- Async tool probe: measures whether a session still answers new speech while
  a tool call is pending.
- Documented two harness bugs that produced wrong results before being fixed:
  probing with silence, and an exact-fit endpoint threshold landing one VAD
  frame short.
- Normalised summary builder and reproducible fixture generators.

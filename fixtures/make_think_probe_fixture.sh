#!/bin/sh
# Regenerate the async-think probe fixture.
#
# The question is deliberately (a) genuinely hard, so delegating is the right
# call rather than a contrivance, and (b) explicit that the agent should think
# first. Explicitness matters: a probe that relies on the agent spontaneously
# reaching for a tool can silently measure nothing on a run where it just
# answers, and a no-dispatch run tells you nothing about what happens during a
# tool call.
#
# macOS `say` keeps this reproducible with no GPU or API dependency. The exact
# voice does not matter; the identical wording across vendors does.
set -eu

OUT="${1:-/tmp/bohita-think-probe.wav}"
FOLLOWUP="${2:-/tmp/bohita-think-followup.wav}"
AIFF="$(dirname "$OUT")/$(basename "$OUT" .wav).aiff"

QUESTION="I want you to think really carefully about this one, it's genuinely hard. Suppose we price per session while every competitor prices per token, and a customer runs three hour calls every single day. Work through the second order effects on our margins, on their incentives to keep sessions open, on what happens when they compare invoices with a competitor, and on whether the pricing survives a finance review. Take your time and reason it through properly before you answer."

# The follow-up is deliberately short. It is spoken while the tool call is
# still outstanding, so it has to fit inside the tool's own duration.
FOLLOWUP_TEXT="How long will that take?"

say -v Samantha -r 180 -o "$AIFF" "$QUESTION"
ffmpeg -nostdin -loglevel error -y -i "$AIFF" -ac 1 -ar 16000 -c:a pcm_s16le "$OUT"
rm -f "$AIFF"

FAIFF="$(dirname "$FOLLOWUP")/$(basename "$FOLLOWUP" .wav).aiff"
say -v Samantha -r 190 -o "$FAIFF" "$FOLLOWUP_TEXT"
ffmpeg -nostdin -loglevel error -y -i "$FAIFF" -ac 1 -ar 16000 -c:a pcm_s16le "$FOLLOWUP"
rm -f "$FAIFF"

echo "wrote $OUT"
echo "wrote $FOLLOWUP"

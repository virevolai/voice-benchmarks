#!/bin/sh
set -eu

output=${1:-/tmp/bohita-parakeet-smoke.wav}
work_dir=$(mktemp -d /tmp/bohita-parakeet-fixture.XXXXXX)
trap 'rm -r "$work_dir"' EXIT

say -v Samantha \
  'Bohita is testing the speech recognition pipeline.' \
  -o "$work_dir/samantha.aiff"
say -v Daniel \
  'Can you hear both voices clearly on Modal?' \
  -o "$work_dir/daniel.aiff"

ffmpeg -loglevel error -y \
  -i "$work_dir/samantha.aiff" \
  -i "$work_dir/daniel.aiff" \
  -filter_complex '[0:a][1:a]concat=n=2:v=0:a=1[out]' \
  -map '[out]' -ar 16000 -ac 1 -c:a pcm_s16le \
  "$output"

duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$output")
awk -v duration="$duration" 'BEGIN { exit !((duration + 0) > 0.1) }' || {
  echo "generated smoke fixture contains no audio: $output" >&2
  exit 1
}

echo "wrote two-voice Parakeet smoke fixture: $output"

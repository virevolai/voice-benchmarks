#!/bin/sh
set -eu

first_output=${1:-/tmp/bohita-gemini-memory-first.wav}
final_output=${2:-/tmp/bohita-gemini-memory-final.wav}
work_dir=$(mktemp -d /tmp/bohita-gemini-memory.XXXXXX)
trap 'rm -r "$work_dir"' EXIT

say -v Samantha \
  'The launch codename is copper lantern. Please remember that for later and acknowledge in five words.' \
  -o "$work_dir/first.aiff"
say -v Samantha \
  'What was the launch codename I asked you to remember at the beginning?' \
  -o "$work_dir/final.aiff"

ffmpeg -loglevel error -y \
  -i "$work_dir/first.aiff" -ar 16000 -ac 1 -c:a pcm_s16le \
  "$first_output"
ffmpeg -loglevel error -y \
  -i "$work_dir/final.aiff" -ar 16000 -ac 1 -c:a pcm_s16le \
  "$final_output"

for output in "$first_output" "$final_output"; do
  duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$output")
  awk -v duration="$duration" 'BEGIN { exit !(duration > 0.1) }' || {
    echo "generated memory fixture contains no audio: $output" >&2
    exit 1
  }
done

echo "wrote Gemini Live memory fixtures: $first_output, $final_output"

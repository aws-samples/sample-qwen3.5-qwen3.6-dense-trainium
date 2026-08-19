#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Concurrent-request check against a running `vllm serve` instance.
#
# Fires 4 completions SIMULTANEOUSLY so they share decode batches, then
# eyeball-checks each output stays on its own topic (the state-crossing
# bug signature was one prompt's content appearing in another's output —
# docs/implementation-notes.md).
#
# Usage:  ./test/concurrent_api_test.sh [PORT] [MODEL_NAME]
#   PORT       default 8000
#   MODEL_NAME default: queried from the server's /v1/models
#
# Greedy (temperature 0) so results are deterministic. Expected:
#   counting  -> continues "6 7 8 9 ..."
#   France    -> mentions Paris
#   story     -> story-like continuation
#   fibonacci -> python code
# Any output continuing a DIFFERENT prompt's topic = the bug.

set -euo pipefail

PORT="${1:-8000}"
BASE="http://localhost:${PORT}"

MODEL_NAME="${2:-$(curl -s "${BASE}/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')}"
echo "Server model: ${MODEL_NAME}"
echo

PROMPTS=(
  "I am gonna keep counting forever, 1 2 3 4 5 "
  "The capital of France is"
  "Once upon a time, there was a"
  "def fibonacci(n):"
)

TMPDIR_OUT="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_OUT}"' EXIT

for i in "${!PROMPTS[@]}"; do
  (
    curl -s "${BASE}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "$(python3 -c "
import json, sys
print(json.dumps({
    'model': '${MODEL_NAME}',
    'prompt': '''${PROMPTS[$i]}''',
    'max_tokens': 48,
    'temperature': 0,
}))")" > "${TMPDIR_OUT}/out_${i}.json"
  ) &
done
wait

echo "=== Results (each must stay on its own topic) ==="
for i in "${!PROMPTS[@]}"; do
  echo
  echo "PROMPT: ${PROMPTS[$i]}"
  python3 -c "
import json
r = json.load(open('${TMPDIR_OUT}/out_${i}.json'))
print('OUTPUT:', repr(r['choices'][0]['text'][:120]))
"
done

#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
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
# If the server uses API-key authentication, export VLLM_API_KEY before
# running this script. The key is read from the environment and never logged.
#
# Greedy (temperature 0) so results are deterministic. Expected:
#   counting  -> continues "6 7 8 9 ..."
#   France    -> mentions Paris
#   story     -> story-like continuation
#   fibonacci -> python code
# Any output continuing a DIFFERENT prompt's topic = the bug.

set -euo pipefail

PORT="${1:-8000}"
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || ((${#PORT} > 5)); then
  echo "ERROR: PORT must be an integer between 1 and 65535, got ${PORT}" >&2
  exit 2
fi
PORT="$((10#${PORT}))"
if ((PORT < 1 || PORT > 65535)); then
  echo "ERROR: PORT must be between 1 and 65535, got ${PORT}" >&2
  exit 2
fi
BASE="http://localhost:${PORT}"

# Requests are intentionally local: bypass inherited proxy settings so an
# Authorization header cannot be sent to an HTTP proxy. Bound all requests so
# a stopped or unhealthy server cannot hang the smoke test indefinitely.
CURL_OPTIONS=(
  --disable
  --fail
  --silent
  --show-error
  --noproxy "*"
  --connect-timeout 5
  --max-time 180
  --header "Content-Type: application/json"
)

# Keep the key out of curl's argument vector and child-process environments.
# curl reads the authorization header from stdin when authentication is used.
API_KEY="${VLLM_API_KEY:-}"
export -n API_KEY
unset VLLM_API_KEY
if [[ "${API_KEY}" == *$'\r'* || "${API_KEY}" == *$'\n'* ]]; then
  echo "ERROR: VLLM_API_KEY must not contain newline characters" >&2
  exit 2
fi
curl_local() {
  if [[ -n "${API_KEY}" ]]; then
    printf 'Authorization: Bearer %s\n' "${API_KEY}" |
      curl "${CURL_OPTIONS[@]}" --header @- "$@"
  else
    curl "${CURL_OPTIONS[@]}" "$@"
  fi
}

MODEL_NAME="${2:-$(
  curl_local "${BASE}/v1/models" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])'
)}"
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

pids=()
for i in "${!PROMPTS[@]}"; do
  (
    payload="$(
      MODEL_NAME="${MODEL_NAME}" PROMPT="${PROMPTS[$i]}" python3 -c '
import json
import os

print(json.dumps({
    "model": os.environ["MODEL_NAME"],
    "prompt": os.environ["PROMPT"],
    "max_tokens": 48,
    "temperature": 0,
}))
'
    )"
    curl_local "${BASE}/v1/completions" \
      -d "${payload}" > "${TMPDIR_OUT}/out_${i}.json"
  ) &
  pids+=("$!")
done

request_failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    request_failed=1
  fi
done
if ((request_failed)); then
  echo "ERROR: one or more concurrent requests failed" >&2
  exit 1
fi

echo "=== Results (each must stay on its own topic) ==="
for i in "${!PROMPTS[@]}"; do
  echo
  echo "PROMPT: ${PROMPTS[$i]}"
  python3 -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as response_file:
    response = json.load(response_file)
print("OUTPUT:", repr(response["choices"][0]["text"][:120]))
' "${TMPDIR_OUT}/out_${i}.json"
done

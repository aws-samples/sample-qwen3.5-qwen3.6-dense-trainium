#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Launch vllm serve for a Qwen3.5/3.6 dense model with the registry patch.
#
# Usage (from a vllm-neuron environment with overrides/ applied —
# see overrides/README.md):
#
#   ./serve.sh                                        # 4B, TP=4, 4K context
#   MODEL="${HOME}/models/Qwen3.5-27B" TP=4 ./serve.sh  # 27B
#   MAX_LEN=256 BUCKET=256 MAX_NUM_SEQS=4 ./serve.sh     # quick smoke config
#
# MODEL defaults to "${HOME}/models/Qwen3.5-4B".
#
# The default HOST=127.0.0.1 accepts requests only from this instance. Set
# VLLM_API_KEY externally from a protected secret source to require API-key
# authentication. See the security guidance in the repository README before
# enabling access from another host.
#
# BUCKET must be strictly increasing and end at MAX_LEN. The Python launcher
# sets max_num_batched_tokens to that final value, explicitly disables chunked
# prefill/APC, and derives VISION_ATTN_BLOCK_SIZE as min(MAX_LEN, 2048) unless
# it is set here.
#
# This script puts src/ (this directory) on PYTHONPATH; sitecustomize.py
# then auto-registers the model architecture in every Python process,
# including vLLM worker subprocesses.

set -euo pipefail

export MODEL="${MODEL:-${HOME}/models/Qwen3.5-4B}"
export TP=${TP:-4}
export MAX_LEN=${MAX_LEN:-4096}
export PORT=${PORT:-8000}

# SECURITY: Keep the server bound to localhost for normal testing.
export HOST="${HOST:-127.0.0.1}"

# Supply this externally from a protected secret source. Never hardcode an API
# key in this file or commit one to the repository.
export VLLM_API_KEY="${VLLM_API_KEY:-}"

# DEMO ONLY — DO NOT UNCOMMENT AS-IS.
# Before binding to all interfaces, apply the controls documented in README.md:
# restrict Security Group ingress to trusted source IPs, use a private VPC,
# and/or enable API-key authentication. Never expose an unauthenticated endpoint
# to 0.0.0.0/0 or ::/0.
# export HOST="0.0.0.0"

export BUCKET=${BUCKET:-}
export VISION_ATTN_BLOCK_SIZE=${VISION_ATTN_BLOCK_SIZE:-}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}

if [[ ! -d "${MODEL}" ]]; then
  echo "ERROR: local model directory does not exist: ${MODEL}" >&2
  echo "       Download the checkpoint first or set MODEL to its local path." >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

if [[ -n "${VLLM_API_KEY}" ]]; then
  AUTH_STATUS="enabled"
else
  AUTH_STATUS="disabled"
fi

echo "==============================="
echo "Qwen3.5/3.6 dense vLLM serve"
echo "  Model:              ${MODEL}"
echo "  TP:                 ${TP}"
echo "  max_len:            ${MAX_LEN}"
echo "  bucket:             ${BUCKET:-<MAX_LEN as single bucket>}"
echo "  vision_block:       ${VISION_ATTN_BLOCK_SIZE:-<min(MAX_LEN, 2048)>}"
echo "  max_num_seqs:       ${MAX_NUM_SEQS}"
echo "  kv_cache_dtype:     ${KV_CACHE_DTYPE}"
echo "  host:               ${HOST}"
echo "  port:               ${PORT}"
echo "  API authentication: ${AUTH_STATUS}"
echo "  PYTHONPATH:         ${PYTHONPATH}"
echo "==============================="

exec python "${HERE}/_serve_main.py"

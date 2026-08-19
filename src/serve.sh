#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch vllm serve for a Qwen3.5/3.6 dense model with the registry patch.
#
# Usage (from a vllm-neuron environment with overrides/ applied —
# see overrides/README.md):
#
#   ./serve.sh                                        # 4B, TP=4, 4K context
#   MODEL=/root/models/Qwen3.5-27B TP=4 ./serve.sh    # 27B
#   MAX_LEN=256 BUCKET=256 MAX_NUM_SEQS=4 ./serve.sh  # quick smoke config
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

export MODEL=${MODEL:-/root/models/Qwen3.5-4B}
export TP=${TP:-4}
export MAX_LEN=${MAX_LEN:-4096}
export PORT=${PORT:-8000}
export BUCKET=${BUCKET:-}
export VISION_ATTN_BLOCK_SIZE=${VISION_ATTN_BLOCK_SIZE:-}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}

HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

echo "==============================="
echo "Qwen3.5/3.6 dense vLLM serve"
echo "  Model:              ${MODEL}"
echo "  TP:                 ${TP}"
echo "  max_len:            ${MAX_LEN}"
echo "  bucket:             ${BUCKET:-<MAX_LEN as single bucket>}"
echo "  vision_block:       ${VISION_ATTN_BLOCK_SIZE:-<min(MAX_LEN, 2048)>}"
echo "  max_num_seqs:       ${MAX_NUM_SEQS}"
echo "  kv_cache_dtype:     ${KV_CACHE_DTYPE}"
echo "  port:               ${PORT}"
echo "  PYTHONPATH:         ${PYTHONPATH}"
echo "==============================="

exec python "${HERE}/_serve_main.py"

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Wrapper that applies the Qwen3.5/3.6 registry patch and execs `vllm serve`.

Configured entirely via env vars to keep the launcher shell simple.
Called from `serve.sh`.

ENV VARS
--------
- MODEL: path to local HF weights (default: ~/models/Qwen3.5-4B).
  Any Qwen3.5/3.6 DENSE checkpoint works — 4B, 9B, 27B (config-driven).
- TP: tensor parallel size (default: 4 — the verified configuration).
  Constraints (validated at model init with clear errors):
    * GQA:      TP must divide num_attention_heads (16 on 4B/9B, 24 on 27B)
    * DeltaNet: TP must divide num_v_heads AND num_k_heads
                (32/16 on 4B, 48/16 on 27B) — head-sharded, not replicated.
- MAX_LEN: max model context (default: 4096). DeltaNet chunked prefill is
  not implemented, so the launcher requires single-shot prefill:
  max_num_batched_tokens and the last BUCKET entry both equal MAX_LEN.
- BUCKET: comma-separated, strictly increasing prefill bucket sizes for
  num_batched_tokens_buckets (default: MAX_LEN as a single bucket).
- VISION_ATTN_BLOCK_SIZE: compatibility block for the vision config that the
  Neuron platform creates from the checkpoint's `vision_config`, even for
  text-only requests (default: min(MAX_LEN, 2048)). The launcher declares the
  matching `num_vision_tokens_buckets` explicitly — see the note by `addl`.
- MAX_NUM_SEQS: decode batch size cap (default: 4). Continuous batching is
  fully supported — the state-crossing fix in overrides/ is REQUIRED for
  MAX_NUM_SEQS > 1 (see docs/implementation-notes.md).
- KV_CACHE_DTYPE: "auto" (BF16, verified) or "fp8" (code path present for
  GQA layers, not verified end-to-end).
- HOST: HTTP bind address (default: 127.0.0.1). Keep the loopback default for
  local testing. See README.md before binding to an external interface.
- PORT: HTTP port (default: 8000).
- VLLM_API_KEY: optional API key for authenticated requests. Supply it through
  a protected environment variable; never hardcode or log it.
"""

import json
import os
import sys


def main() -> int:
    # 0. Host-environment guards (EFA affinity on EFA-less instances).
    from neuron_host_env import configure_efa_affinity

    configure_efa_affinity()

    # 1. Patch the registries FIRST, before vllm sees the model.
    from qwen3_6.register import register, install_post_plugin_hook

    register()
    install_post_plugin_hook()

    # 2. Build sys.argv from env vars.
    model = os.path.abspath(
        os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.5-4B"))
    )
    tp = os.environ.get("TP", "4")
    max_len = int(os.environ.get("MAX_LEN", "4096"))
    host = os.environ.get("HOST", "127.0.0.1").strip()
    port = os.environ.get("PORT", "8000")
    api_key = os.environ.get("VLLM_API_KEY", "")
    bucket = os.environ.get("BUCKET", "").strip()
    max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", "4"))
    kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE", "auto").strip()

    if not os.path.isdir(model):
        raise FileNotFoundError(
            f"Local model directory does not exist: {model}. "
            "Download the checkpoint first or set MODEL to its local path."
        )

    if max_len <= 0:
        raise ValueError(f"MAX_LEN must be positive, got {max_len}")
    if max_num_seqs <= 0:
        raise ValueError(f"MAX_NUM_SEQS must be positive, got {max_num_seqs}")
    if not host:
        raise ValueError("HOST must not be empty")

    if host in {"0.0.0.0", "::", "[::]"}:
        print(
            "[serve] WARNING: vLLM is binding to all network interfaces. "
            "Confirm the Security Group, VPC, and authentication controls "
            "documented in README.md are in place.",
            file=sys.stderr,
        )

    if bucket:
        try:
            bucket_list = [int(entry.strip()) for entry in bucket.split(",")]
        except ValueError as exc:
            raise ValueError(
                f"BUCKET must be a comma-separated list of integers, got {bucket!r}"
            ) from exc
    else:
        bucket_list = [max_len]

    if any(value <= 0 for value in bucket_list):
        raise ValueError(f"BUCKET entries must be positive, got {bucket_list}")
    if any(
        current >= following
        for current, following in zip(bucket_list, bucket_list[1:])
    ):
        raise ValueError(
            f"BUCKET entries must be strictly increasing, got {bucket_list}"
        )

    # vllm-neuron validates that the final bucket equals this scheduler value.
    # Equality with MAX_LEN keeps this model on the single-shot prefill path;
    # DeltaNet does not yet preserve recurrent state across prompt chunks.
    max_num_batched_tokens = bucket_list[-1]
    if max_num_batched_tokens != max_len:
        raise ValueError(
            f"The last BUCKET entry ({max_num_batched_tokens}) must equal "
            f"MAX_LEN ({max_len}) for DeltaNet single-shot prefill."
        )

    vision_block_value = os.environ.get("VISION_ATTN_BLOCK_SIZE", "").strip()
    vision_attention_block_size = (
        int(vision_block_value) if vision_block_value else min(max_len, 2048)
    )
    if not 0 < vision_attention_block_size <= max_len:
        raise ValueError(
            "VISION_ATTN_BLOCK_SIZE must be positive and no larger than "
            f"MAX_LEN={max_len}, got {vision_attention_block_size}"
        )

    addl = {
        "neuron_config": {
            "num_batched_tokens_buckets": bucket_list,
            "num_seqs_buckets": [max_num_seqs],
            "on_device_sampling_config": {"all_greedy": True},
        },
        # This is top-level in additional_config, not inside neuron_config.
        # A single vision bucket equal to one attention block prevents the
        # plugin's 2048-token default from conflicting with short text
        # contexts. We state the bucket EXPLICITLY rather than capping the
        # auto-generated list with `max_vision_seq_len`: on 0.24,
        # `NeuronPlatform._resolve_vision_auto_config` writes its derived
        # `num_vision_tokens_buckets` back into this dict, and
        # `check_and_update_config` runs again in the EngineCore process on
        # the already-mutated copy — where `max_vision_seq_len` plus the
        # injected buckets trips its own "cannot set both" guard. The explicit
        # form takes the early-return branch and is therefore idempotent.
        # [block_size] is exactly what the auto path derived from
        # max_vision_seq_len == block_size, so behavior is unchanged.
        "vision_neuron_config": {
            "vision_attention_block_size": vision_attention_block_size,
            "num_vision_tokens_buckets": [vision_attention_block_size],
        },
    }

    sys.argv = [
        "vllm",
        "serve",
        model,
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(max_len),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-num-seqs", str(max_num_seqs),
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--kv-cache-dtype", kv_cache_dtype,
        # SECURITY: 127.0.0.1 is the safe default for local testing. Binding
        # to 0.0.0.0 or :: exposes the listener to reachable interfaces. Do
        # not enable an external bind without the controls in README.md,
        # including restricted Security Group sources, private networking,
        # and/or authentication.
        "--host", host,
        "--port", str(port),
        # Zero limits keep requests text-only. They do not suppress Neuron's
        # vision auto-configuration, so addl above also supplies a bounded
        # vision_neuron_config that the text-only factory accepts and ignores.
        "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),
        "--additional-config", json.dumps(addl),
    ]

    if api_key:
        sys.argv.extend(["--api-key", api_key])

    print("[serve] launching vllm with argv:")
    for index, arg in enumerate(sys.argv):
        display_arg = (
            "<redacted>"
            if index > 0 and sys.argv[index - 1] == "--api-key"
            else arg
        )
        print(f"  {display_arg}")

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

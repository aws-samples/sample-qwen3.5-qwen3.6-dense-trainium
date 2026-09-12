# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Offline (LLM.generate) example for the Qwen3.5/3.6 dense family.

Submits 4 prompts SIMULTANEOUSLY so they decode in the same continuous
batch — this exercises the state-slot machinery (see
docs/implementation-notes.md); each output must stay on its own topic.

Usage (with src/ on PYTHONPATH so the registry patch applies, and the
overrides/ files in place — see overrides/README.md):

    PYTHONPATH=src python src/run_offline_example.py \
        --model-checkpoint ~/models/Qwen3.5-4B [--tp 4] [--kv-cache-dtype auto]
"""

import argparse
import os

# The DeltaNet recurrent state is kept in fp32 for correctness, which uses
# more HBM and leaves a smaller KV-cache budget. For this small smoke test
# (256 tokens, 4 seqs) a sub-1 GiB KV budget is fine, so lower the safety
# floor below its 1.0 GiB default.
os.environ.setdefault("VLLM_NEURON_MIN_KV_BUDGET_GIB", "0.5")

# Hosts without EFA (e.g. trn2.3xlarge) must skip EFA affinity or the worker
# raises during startup — see src/neuron_host_env.py.
from neuron_host_env import configure_efa_affinity

configure_efa_affinity()

import qwen3_6  # noqa: F401  (import registers the architecture)
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="~/models/Qwen3.5-4B",
        help="Path to a Qwen3.5/3.6 dense HF checkpoint (4B/9B/27B)",
    )
    parser.add_argument(
        "--tp", type=int, default=4, help="Tensor parallel size (default 4)"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp8"],
        help="KV cache dtype ('auto' = BF16, verified; 'fp8' untested e2e)",
    )
    args = parser.parse_args()
    args.model_checkpoint = os.path.abspath(
        os.path.expanduser(args.model_checkpoint)
    )
    if not os.path.isdir(args.model_checkpoint):
        parser.error(
            "local model directory does not exist: "
            f"{args.model_checkpoint}. Download the checkpoint first or pass "
            "--model-checkpoint with the correct path."
        )

    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        tensor_parallel_size=args.tp,
        kv_cache_dtype=args.kv_cache_dtype,
        # Text-only, plus the bounded vision_neuron_config the platform needs
        # anyway — see src/_serve_main.py's docstring for why.
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [256],
                "num_seqs_buckets": [4],
            },
            # Top-level, NOT nested under neuron_config. One explicit bucket
            # (not `max_vision_seq_len`) — see src/_serve_main.py for why the
            # explicit form is required on 0.24.
            "vision_neuron_config": {
                "vision_attention_block_size": 256,
                "num_vision_tokens_buckets": [256],
            },
        },
    )

    prompts = [
        "I am gonna keep counting forever, 1 2 3 4 5 ",
        "The capital of France is ",
        "Once upon a time, there was a ",
        "def fibonacci(n):",
    ]
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0, top_p=1.0)

    outputs = llm.generate(prompts, sampling_params)

    for out in outputs:
        print(f"PROMPT: {out.prompt!r}")
        print(f"OUTPUT: {out.outputs[0].text!r}")
        print()


if __name__ == "__main__":
    main()

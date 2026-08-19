# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched-consistency ("churn") test — on-device, pass/fail.

Verifies the continuous-batching state-slot fix end-to-end
(docs/implementation-notes.md): every prompt generated in a BATCH must
be token-exact vs the same prompt generated ALONE (greedy sampling makes
this a strict equality check).

The prompt set is adversarial by construction:
  - staggered max_tokens → requests finish at different times → the batch
    condenses mid-run (exercises request→slot stability across row swaps);
  - 6 prompts > max_num_seqs=4 → requests queue and state slots get REUSED
    by later requests.

Before the fix, outputs crossed between sequences (e.g. the France prompt
completed with fibonacci code). Exit code 0 = all token-exact, 1 = mismatch.

Usage (device required; overrides/ applied per overrides/README.md):

    PYTHONPATH=src python test/batched_consistency_test.py \
        [--model /root/models/Qwen3.5-4B] [--tp 4]
"""

import argparse
import os
import sys

# Lower the KV-budget floor for this small config (fp32 DeltaNet state leaves
# a smaller budget) — see src/run_offline_example.py.
os.environ.setdefault("VLLM_NEURON_MIN_KV_BUDGET_GIB", "0.5")

# Hosts without EFA (e.g. trn2.3xlarge) must skip EFA affinity or the worker
# raises during startup — see src/neuron_host_env.py.
from neuron_host_env import configure_efa_affinity

configure_efa_affinity()

import qwen3_6  # noqa: F401  (import registers the architecture)
from vllm import LLM, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen3.5-4B")
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        tensor_parallel_size=args.tp,
        # Text-only + bounded vision config; see src/_serve_main.py's docstring.
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [256],
                "num_seqs_buckets": [4],
            },
            "vision_neuron_config": {
                "vision_attention_block_size": 256,
                "num_vision_tokens_buckets": [256],
            },
        },
    )

    # Staggered lengths force mid-run batch condensation; 6 > max_num_seqs=4
    # forces slot reuse.
    prompts_and_lens = [
        ("I am gonna keep counting forever, 1 2 3 4 5 ", 40),
        ("The capital of France is", 8),
        ("Once upon a time, there was a", 24),
        ("def fibonacci(n):", 16),
        ("The first president of the United States was", 12),
        ("Roses are red, violets are", 8),
    ]

    # Singles = per-request golden reference (single-sequence batches are
    # unaffected by the state-crossing bug class).
    refs = {}
    for prompt, n in prompts_and_lens:
        sp = SamplingParams(max_tokens=n, temperature=0.0, top_p=1.0)
        refs[prompt] = list(llm.generate([prompt], sp)[0].outputs[0].token_ids)

    # All together: batched, staggered, over-subscribed.
    batch_sps = [
        SamplingParams(max_tokens=n, temperature=0.0, top_p=1.0)
        for _, n in prompts_and_lens
    ]
    bouts = llm.generate([p for p, _ in prompts_and_lens], batch_sps)

    ok = True
    for out in bouts:
        got = list(out.outputs[0].token_ids)
        want = refs[out.prompt]
        match = got == want
        ok &= match
        status = "MATCH   " if match else "MISMATCH"
        print(f"{status} :: {out.prompt!r}")
        if not match:
            div = next(
                (i for i, (a, b) in enumerate(zip(want, got)) if a != b),
                min(len(want), len(got)),
            )
            print(f"    diverges at token {div}; ref {len(want)} toks, got {len(got)}")

    print(f"\nBATCHED-CONSISTENCY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

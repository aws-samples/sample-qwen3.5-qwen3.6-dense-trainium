# Known Boundaries — where the tested envelope ends

The enablement is complete and verified within the envelope below. Items under
"untested / not implemented" are scope boundaries, not known defects — they are
the first things to check or build for whoever extends this work.

## Verified envelope

Everything below was verified on device on the vllm-neuron `0.24.0.1.1.0`
stack this repo targets (see the README's Compatibility Matrix).

- **Models**: all four published dense checkpoints — Qwen3.5-4B, Qwen3.5-9B,
  Qwen3.5-27B and Qwen3.6-27B. Together they span the config-driven dimensions
  (24+8 vs 48+16 layers, hidden 2560/4096/5120, 16 vs 24 Q heads, 32/16 vs
  48/16 DeltaNet heads, tied vs untied lm_head).
- **Parallelism**: TP=1 and TP=4 on trn2.3xlarge (LNC=2); the 27B fits the
  KV budget at TP=4 (the goal of the head-sharding work).
- **Paths**: prefill + multi-token decode; single-prompt and continuous
  batching (state-slot fix verified by a churn test with staggered lengths,
  mid-run batch condensation, and slot reuse).
- **Modes**: greedy sampling, with batched output token-exact vs single-prompt
  references at a fixed TP (not across TP degrees — see the README's Validation
  Summary); both online (OpenAI-compatible `vllm serve`) and offline
  (`LLM.generate()`) — same engine underneath.

## Untested / not implemented

### 1. Chunked prefill with DeltaNet — NOT IMPLEMENTED

All verification used prompts that fit a single prefill bucket.
`_forward_prefill` starts the DeltaNet recurrence from zero state on every
call; a prompt chunked across multiple prefill steps would silently drop the
earlier chunks' recurrence history. Implementing it needs a state-resume path:
read the request's slot state at chunk start (the state-slot machinery gives
you the addressing; the missing piece is prefill *reading* its slot, plus the
conv-state stitch across the chunk boundary).

Until then, use single-shot prefill: `max_num_batched_tokens` and the final
`num_batched_tokens_buckets` entry must both equal `MAX_LEN`. The launcher
(`src/_serve_main.py`) enforces this invariant and explicitly disables chunked
prefill and automatic prefix caching rather than allowing a silently incorrect
DeltaNet prompt split.

### 2. FP8 KV cache — code present, unverified

`--kv-cache-dtype fp8` has a write/read path in the GQA layers (scale folding
on K, explicit dequant on V) but every verified run used `auto` (BF16).
DeltaNet layers are unaffected: their state lives in side-channel buffers, not
the paged KV cache, and keeps its own dtypes regardless of this flag
(recurrent state fp32, conv state bf16).

### 3. Preemption under memory pressure — implemented, unexercised

State-slot release on `preempted_req_ids` is implemented in the runner
override, but no test drove the scheduler into real preemption. A preempted
request re-prefills from scratch on resume (correct by design — its slot is
freed and a fresh one acquired), but this path has not been observed live.

### 4. The smaller family members (0.8B / 2B) — expected-fine, not run

All four larger dense checkpoints are validated (see "Verified envelope"), but
the 0.8B and 2B members have not been run. Every dimension is read from the HF
config and the divisibility guard raises a clear error on non-divisible head
counts, so they should work by construction at TP values that divide their head
counts. No run has confirmed it.

### 5. Performance — correctness-first enablement

The per-`(batch, head)` sequential NKI kernel loop in prefill is the obvious
optimization target (48 calls/layer on the 27B before sharding; TP=4 head
sharding already cuts it to 12 calls/rank). Batching the kernel across heads,
or a multi-head kernel variant, is the next big win. No throughput / TTFT
benchmarking was done as part of this enablement.

### 6. Cold-compile time

First serve compiles one NEFF per bucket per graph — measured at ~10 minutes
for the 4B and ~15 for the 27B at the small smoke config, and longer with
multiple buckets. Warm restarts hit the NEFF cache and take 2-3 minutes; clear
that cache after any edit to the model or override files (`overrides/apply.sh`
does it, and `overrides/README.md` documents the path).
